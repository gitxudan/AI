import chromadb
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
#client = chromadb.Client()
client = chromadb.PersistentClient(path="./chroma_db") # for persistence, will create a folder called chroma_db in the current directory

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
    collection.add(
        documents=[sentence], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        metadatas=[{"source": "DQ Generated ChromaDB Demo"}], # filter on these!
        ids=[f"doc_{i}"], # unique for each doc
    )


'''
# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["where is the document for study programming language?"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

print(results)
'''
