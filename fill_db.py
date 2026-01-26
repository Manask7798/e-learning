from chromadb.utils import embedding_functions
import os
from uuid import uuid4
from langchain_core.documents import Document
import chromadb
import langgraph



# Vekordatenbank einrichten
client = chromadb.Client()
client = chromadb.PersistentClient(path="./chroma_db")


# Deutsche Embeddings von JinAI benutzen (https://huggingface.co/jinaai/jina-embeddings-v2-base-de)
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="jinaai/jina-embeddings-v2-base-de"
)

# Kollektion erstellen
collection = client.get_or_create_collection(
    name="studienberatung",
    embedding_function=emb,
)

# Dokumente in die Datenbank laden
documents = []
directory_path = 'documents'
identifier=0
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    with open(file_path, 'r') as file:
        content = file.read()
    identifier += 1
    document = Document(
        page_content=content,
        metadata={"title": filename},
        id=identifier
    )
    documents.append(document)

uuids = [str(uuid4()) for _ in range(len(documents))]
collection.add(
    documents=[doc.page_content for doc in documents],
    metadatas=[doc.metadata for doc in documents],
    ids=uuids
)

# Datenbank abfragen
results = collection.query(
    query_texts=["Ist es sinnvoll, zu studieren?"],
    n_results=2,
)

# Treffer ausgeben
docs = results["documents"][0]
metas = results["metadatas"][0]

for doc, meta in zip(docs, metas):
    print("----")
    print(doc)
    print(meta)