from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello RAG")
    print(app.invoke(input={"question": "IGA (Istanbul Grand Airport) havalimanı otoparkında Hızlı Geçiş Sistemi (HGS) ile ödeme seçeneği nerede bulunuyor?"}))
