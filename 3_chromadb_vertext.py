import chromadb
import os
from google import genai

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GCP_KEY/orchestrator-lab-dq-202601-cred.json'
gcp_client = genai.Client( vertexai=True )
embeddingModel = "text-embedding-005"


# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.PersistentClient(path="./chroma_vertex_db") # for persistence, will create a folder called chroma_db in the current directory

sentences = [
    'This is the first document for python learning.',
    'This document is about using python as a tool to automate tasks.',
    'And this is the third one.',
    'Is this the first document?',
]

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection("all-my-documents")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!

for i, sentence in enumerate(sentences):
    
    response = gcp_client.models.embed_content(
        model=embeddingModel,
        contents=sentence
    )
    
    embedding = response.embeddings[0].values
    
    collection.add(
        documents=[sentence], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        metadatas=[{"source": "Vertex ChromaDB Demo"}], # filter on these!
        ids=[f"doc_{i}"], # unique for each doc
        embeddings=[embedding]
    )