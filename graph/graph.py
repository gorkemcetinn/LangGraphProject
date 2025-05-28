from dotenv import load_dotenv

from langgraph.graph import END, StateGraph
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from graph.node_constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()

def decide_to_generate(state):
    print("---DEĞERLENDİRİLEN DOKÜMANLARI ANALİZ ET---")

    if state["web_search"]:
        print(
            "---KARAR: TÜM DOKÜMANLAR SORUYLA İLGİLİ DEĞİL, WEB ARAMASI EKLENİYOR---"
        )
        return WEBSEARCH
    else:
        print("---KARAR: MODEL YANIT OLUŞTURUYOR (GENERATE)---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---HALLÜSİNASYON KONTROLÜ YAPILIYOR---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---KARAR: MODELİN CEVABI BELGELERLE UYUMLU (GROUNDED)---")
        print("---YANIT, SORU İLE NE KADAR UYUMLU DEĞERLENDİRİLİYOR---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---KARAR: MODELİN CEVABI SORUYU DOĞRU YANITLIYOR---")
            return "useful"
        else:
            print("---KARAR: MODELİN CEVABI SORUYU DOĞRU YANITLAMIYOR---")
            return "not useful"
    else:
        print("---KARAR: MODELİN CEVABI BELGELERLE DESTEKLENMİYOR, YENİDEN DENE---")
        return "not supported"


def route_question(state: GraphState) -> str:
    print("---SORU YÖNLENDİRİLİYOR---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---SORU WEB ARAMASINA YÖNLENDİRİLDİ---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---SORU RAG (Vektör Veritabanı) YÖNTEMİNE YÖNLENDİRİLDİ---")
        return RETRIEVE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
