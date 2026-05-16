import chromadb
from chromadb import Collection



def setup_chromadb() -> Collection:
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = client.get_collection("all-my-documents")
        return collection
    except Exception as e:
        print("Collection not found. Please run ingest.py first!")
        return None
    

def query_database(collection, search_query, num_results=2):
    # 1. Point to the exact same persistent directory
    # 2. Fetch the existing collection (will raise an error if it doesn't exist)

        
    # 3. Query the collection
    results = collection.query(
        query_texts=[search_query],
        n_results=num_results
    )
    
    return results

if __name__ == "__main__":
    
    
    collection = setup_chromadb()
    user_query = "where is the document for study programming language?"
    results = query_database(collection, user_query)
    print(f"Query results: {results}")