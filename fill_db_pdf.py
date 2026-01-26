import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()
documents = {
     "doc1" : {
         "title" : "Zogg, Martin: Einführung in die Verfahrenstechnik",
         "path" : "sources/ZOGG-noch-kürzer.pdf"
     },
     "doc2" : {
         "title" : "Bedienungsanleitung Ilmvac",
         "path" : "sources/Bedienungsanleitung.pdf"
     },
    "doc3" : {
         "title" : "Betriebsanweisung Druckfilter",
         "path" : "sources/Betriebsanweisung_Druckfilter.pdf"
     },
    "doc4" : {
         "title" : "Betriebsanweisung Vakuumfilter",
         "path" : "sources/Betriebsanweisung_Vakuumfilter.pdf"
     },
    "doc5" : {
         "title" : "Beispiel-Protokolle",
         "path" : "sources/Bsp_Protokolle.pdf"
     },
    "doc6" : {
         "title" : "Protokollinhalte",
         "path" : "sources/Protokollinhalte.pdf"
     }

     

     
 }
# PDF laden
pages = []
for key, doc in documents.items():
    reader = PdfReader(doc["path"])

    if key == "doc1":
        start_page_number = 95
    else:
        start_page_number = 1

    for page_number, page in enumerate(reader.pages, start=start_page_number):
        text = page.extract_text()
        if text:
            pages.append({
                "page": page_number,
                "text": text,
                "title": doc["title"]
            })
from collections import defaultdict

per_doc = defaultdict(list)
for p in pages:
    per_doc[p["title"]].append(p["page"])

for title, pg in per_doc.items():
    print(title, "Seiten:", min(pg), "bis", max(pg))


# Text in kleinere Bestandteile (Chunks) aufteilen
text_splitter = RecursiveCharacterTextSplitter(
     chunk_size=800,
     chunk_overlap=100,
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
     embedding_function=emb
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
