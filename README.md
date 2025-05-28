# LangGraphProject

LangGraphProject, LangChain ve LangGraph altyapılarını kullanarak geliştirilmiş, çok adımlı kontrol mekanizmalarına sahip bir **RAG (Retrieval-Augmented Generation)** sistemidir. Bu yapı, kullanıcının sorusuna göre ilgili belgeleri bulur, gerekirse web araması yapar, yanıttaki hallüsinasyonları tespit eder ve sadece güvenilir bilgilerle oluşturulmuş cevaplar üretir.

---

## 🚀 Özellikler

- 🔎 **Otomatik veri yönlendirme**: Sorunun içeriğine göre vektör veritabanı veya web aramasına yönlendirme yapılır.
- 📚 **Belge skorlama**: İlgisiz belgeler filtrelenir, yalnızca alakalı olanlarla yanıt üretilir.
- 🧠 **Hallüsinasyon kontrolü**: Üretilen cevaplar, belgelerle tutarlılık açısından kontrol edilir.
- 💬 **Yararlılık derecelendirmesi**: Yanıt soruyu gerçekten yanıtlıyor mu? LLM ile analiz edilir.
- 🌐 **Web araması (Tavily API)**: Belgeler yetersizse dış kaynaklardan destek alınır.
- 🧰 **LLM Zincirleri ve Akış Diyagramı**: LangGraph ile oluşturulmuş, node bazlı akış mimarisi.
- 🔐 **.env ile güvenli API entegrasyonu**: OpenAI, Tavily ve LangSmith API anahtarları çevresel değişkenlerde tutulur.

---

## 📂 Proje Yapısı

```text
LangGraphProject/
│
├── .airport/                    # Vektör veritabanı (ChromaDB) dizini
├── .gitignore                   # Git'e dahil edilmemesi gereken dosyalar
├── .env                         # Gerçek API anahtarlarını içeren dosya (asla commit etme)
├── .env.example                 # Ortam değişkenleri için örnek dosya (kullanıcılar için)
│
├── graph.png                    # LangGraph akış yapısını görselleştiren diyagram
├── requirements.txt             # Projeye ait tüm Python bağımlılıkları
├── main.py                      # Uygulamanın giriş noktası, `graph.py`'i çağırır
├── ingestion.py                 # Web'den veri çekimi ve vektör oluşturma işlemleri
│
├── graph/                       # LangGraph'e özel tüm iş akışı modülleri
│   ├── graph.py                 # Akış mimarisinin oluşturulduğu dosya
│   ├── state.py                 # Akış sırasında taşınan veri yapısının tanımı
│   ├── node_constants.py        # Akış düğümleri için sabit isimler
│   ├── __init__.py              # Modül tanımlayıcı
│   │
│   ├── chains/                  # LLM tabanlı derecelendirme zincirleri
│   │   ├── answer_grader.py         # Yanıtın soruyu karşılama kontrolü
│   │   ├── hallucination_grader.py  # Yanıtın belge tabanlı olup olmadığını ölçer
│   │   ├── retrieval_grader.py      # Belge ile soru ilgisi kontrolü
│   │   ├── generation.py            # LLM ile metin üretimi
│   │   └── router.py                # Soru yönlendirme (vektör mü web mi?)
│   │
│   └── nodes/                   # LangGraph node’larının tanımı
│       ├── retrieve.py              # Vektör veritabanından belge alma
│       ├── grade_documents.py       # Belge ile soruyu eşleştirme
│       ├── generate.py              # Cevap üretimi (generation_chain)
│       └── web_search.py           # Tavily API ile dış web araması

