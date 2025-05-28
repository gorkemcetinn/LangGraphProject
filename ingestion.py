from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Belgelerin linklerini alıyoruz
urls = ["https://www.igairport.aero/surdurulebilirlik/",
        "https://www.igairport.aero/iga-dunyasi/hakkimizda/sertifika-ve-oduller/",
        "https://www.igairport.aero/havacilik/",
        "https://www.istairport.com/hizmetler/kesfet/iga-pass-deneyimleri/?tab=1#gelen-yolcu-hizmetleri",
        "https://www.istairport.com/hizmetler/kesfet/iga-pass-deneyimleri/?tab=2#iga-pass-premium-servisler",
        "https://www.istairport.com/hizmetler/kesfet/iga-pass-deneyimleri/?tab=3#iga-pass-premium-paketler",
        "https://www.istairport.com/hizmetler/kesfet/havalimani-olanaklari/istanbul-airport-mobil-uygulamasi/",
        "https://www.istairport.com/hizmetler/kesfet/konaklama/istanbul-havalimani-oteli/",
        "https://www.istairport.com/havalimani/havalimani-ulasim/?tab=1#sehir-ici",
        "https://www.istairport.com/havalimani/havalimani-ulasim/havalimani-otopark/",
        "https://www.istairport.com/hizmetler/yeme-icme/restoran-ve-kafeler/burger-king-d-bolgesine-yakin/",
        "https://www.istairport.com/hizmetler/kesfet/kayip-ve-buluntu-esya-hizmeti/#https://www.istairport.com/hizmetler/deneyim/kayip-ve-buluntu-esya-hizmeti/",
        ]


# Dökümanları yüklüyoruz
docs = [WebBaseLoader(url).load() for url in urls]

# Dökümanları sublist yaparak birleştiriyoruz
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0,
)

# Dökümanları parçalara ayırıyoruz
splits = text_splitter.split_documents(docs_list)

# Parçalara ayrılan dökümanları vektörize ediyoruz
vector_store = Chroma.from_documents(
    documents=splits,
    collection_name="ist-airport",
    embedding=OpenAIEmbeddings(),
    persist_directory="./.airport"
)

# Vektörler üzerinden arama yapabilmek için retriever oluşturuyoruz.
# Retriever, kullanıcının sorduğu soruya en yakın vektörleri bulup geri döndürür.
retriever = Chroma(
    collection_name="ist-airport",
    persist_directory="./.airport",
    embedding_function=OpenAIEmbeddings()
).as_retriever()


if __name__ == "__main__":
    print("Vektörler oluşturuldu ve arama yapmaya hazır hale getirildi.")


