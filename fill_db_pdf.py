import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()
documents = {
     "doc1" : {
         "title" : "Einführung in die Verfahrenstechnik",
         "path" : "sources/ZOGG-kurz.pdf"
     }
 }
# # PDF laden
pages = []
for key, doc in documents.items():
     reader = PdfReader(doc["path"])
     for page_number, page in enumerate(reader.pages, start=85):
         text = page.extract_text()
         if text:
             pages.append({
                 "page": page_number,
                 "text": text,
                 "title": doc["title"]
             })

print(f"Loaded PDF with {len(pages)} pages")

# Text in kleinere Bestandteile (Chunks) aufteilen
text_splitter = RecursiveCharacterTextSplitter(
     chunk_size=350,
     chunk_overlap=150,
 )
#
# # Vektordatenbank einrichten und Kollektion erstellen
client = chromadb.PersistentClient(path="./chroma_neu")
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name="jinaai/jina-embeddings-v2-base-de",
     device="cpu"
)
collection = client.get_or_create_collection(
     "verfahrenstechnik",
     embedding_function=emb,
     metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 32,
        "hnsw:construction_ef": 400,
        "hnsw:search_ef": 200
     }
)
#
ids = []
metadatas = []
chunks = []
for page in pages:
     page_chunks = text_splitter.split_text(page["text"])

     for chunk in page_chunks:
         chunks.append(chunk)
         metadatas.append({
             "source": page["title"],
             "page": page["page"]
         })
         ids.append(str(uuid4()))

collection.add(
     documents=chunks,
     metadatas=metadatas,
     ids=ids
)

print("Stored PDF in Chroma.")

# Die Datenbank abfragen
query = "Welche anderen Filtrationsmethoden gibt es außer Druckfiltration?"
results = collection.query(
    query_texts=[query],
    n_results=10,
)

print(results)
