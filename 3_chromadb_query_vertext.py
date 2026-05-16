import chromadb
from chromadb import Collection
import os
from google import genai

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GCP_KEY/orchestrator-lab-dq-202601-cred.json'
gcp_client = genai.Client( vertexai=True )
embeddingModel = "text-embedding-005"



def setup_chromadb() -> Collection:
    client = chromadb.PersistentClient(path="./chroma_de_knowledge_db")
    try:
        collection = client.get_collection("knowledge-collection")
        print(collection.count())
        return collection
    except Exception as e:
        print("Collection not found. Please run ingest.py first!")
        return None
    

def query_database(collection, search_query, num_results=2):
    # 1. Point to the exact same persistent directory
    # 2. Fetch the existing collection (will raise an error if it doesn't exist)

    query_response = gcp_client.models.embed_content(
        model=embeddingModel,
        contents=search_query
    )

    query_embedding = query_response.embeddings[0].values
       
    # 3. Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results
    )
    
    return results

if __name__ == "__main__":
    
    
    collection = setup_chromadb()
    user_query = "business data analysis"
    results = query_database(collection, user_query)
    print(f"Query results: {results}")
