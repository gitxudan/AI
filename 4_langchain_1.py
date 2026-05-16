from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Defining raw data source
raw_documents = [
    {
        "text": "This is the first document for python learning.",
        "source": "DQ Generated ChromaDB Demo",
        "doc_id": "doc_0"
    },
    {
        "text": "This document is about using python as a tool to automate tasks.",
        "source": "from internet - ",
        "doc_id": "doc_1"
    },
    {
        "text": "And this is the third one.",
        "source": "vertex ai demo",
        "doc_id": "doc_2"
    }
]

# 2. Transforming raw data into LangChain Document objects
langchain_docs = []

for doc in raw_documents:
    lc_doc = Document(
        page_content=doc["text"],      # The actual text content to be embedded
        metadata={                    # Dictionary for filtering/scoping queries
            "source": doc["source"],
            "length": len(doc["text"])
        },
        id=doc["doc_id"]              # Unique identifier string (supported in modern LangChain)
    )
    langchain_docs.append(lc_doc)




txt_loader = TextLoader("docs/python_intro.txt",encoding="utf-8")
langchain_docs.extend(txt_loader.load())


# 3. Initialize and extend with the PDF file
pdf_loader = PyPDFLoader("docs/guide-to-business-data-analytics.pdf")

# .load() splits the PDF automatically, returning 1 Document per page
pdf_pages = pdf_loader.load()
for page in pdf_pages:
    # metadata['page'] is 0-indexed (e.g., 0, 1, 2...)
    page_num = page.metadata.get("page", 0) 
    page.id = f"sample_pdf_page_{page_num}"


langchain_docs.extend(pdf_pages)




# 1. Initialize DirectoryLoader 
# We point to the directory, target only .docx extensions, and pass the Docx2txtLoader
dir_loader = DirectoryLoader(
    path="docs/",
    glob="**/*.docx",
    loader_cls=Docx2txtLoader
)

# 2. Load all matching Word documents
docx_documents = dir_loader.load()

# 3. Iterate and assign clean, deterministic IDs before extending
for i, doc in enumerate(docx_documents):
    # Extract file name from metadata path to make the ID meaningful
    file_path = doc.metadata.get("source", "word_doc")
    # Clean up the path format for the ID string (e.g., "docs/plan.docx" -> "plan_docx")
    clean_name = file_path.replace("docs/", "").replace(".", "_")
    
    doc.id = f"{clean_name}_{i}"

# 4. Extend the central list
langchain_docs.extend(docx_documents)

# --- Inspect the Result ---
for doc in langchain_docs:
    print('-----------------------------------')
    print(f"Content Preview: {doc.page_content[:100]}...")
    print(f"Default Metadata: {doc.metadata}")
    print(f"ID:       {doc.id}")
    print(f"Source:   {doc.metadata.get('source', 'N/A')}")
    print(f"Length:   {doc.metadata.get('length', 'N/A')}")




# =====================================================================
# STAGE 2: CHUNKING PROCESS (The Missing Piece)
# =====================================================================
print("\n--- Splitting Documents into Chunks ---")

# 1000 characters chunk size with 200 characters overlap handles semantic context beautifully
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# This breaks down your large files (PDFs, text, docx) into smaller pieces
final_chunks = text_splitter.split_documents(langchain_docs)
print(f"Original items: {len(langchain_docs)} -> Split into {len(final_chunks)} total chunks.")

'''
if final_chunks:
    print("\n--- Sample Chunk ---")
    sample_chunk = final_chunks[0]
    print(f"Chunk Content Preview: {sample_chunk.page_content[:200]}...")
    print(f"Chunk Metadata: {sample_chunk.metadata}")
    print(f"Chunk ID: {sample_chunk.id}")
'''




import chromadb
import os
from google import genai

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GCP_KEY/orchestrator-lab-dq-202601-cred.json'
gcp_client = genai.Client( vertexai=True )
embeddingModel = "text-embedding-005"


# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.PersistentClient(path="./chroma_de_knowledge_db") # for persistence, will create a folder called chroma_db in the current directory
print("ChromaDB setup complete. Ready to add documents.")

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.get_or_create_collection("knowledge-collection")
print(f"Collection 'knowledge-collection' is ready. Current document count: {collection.count()}")  
# Add docs to the collection. Can also update and delete. Row-based API coming soon!

for i, chunk in enumerate(final_chunks):
    
    response = gcp_client.models.embed_content(
        model=embeddingModel,
        contents=chunk.page_content
    )
    
    embedding = response.embeddings[0].values
    '''
    collection.add(
        documents=[sentence.page_content], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        metadatas=[{"source": "Vertex ChromaDB Demo"}], # filter on these!
        ids=[f"doc_{i}"], # unique for each doc
        embeddings=[embedding]
    )'''
    
    
    # Construct distinct chunk metadata based on its parent loader context
    metadata = chunk.metadata if chunk.metadata else {}
    metadata["chunk_index"] = i  # Tracks the chunk positioning layout
    
    # Build a deterministic, unique ID for every single split chunk
    parent_id = chunk.id if chunk.id else f"doc_{i}"
    chunk_id = f"{parent_id}_chunk_{i}"
    
    # Store clean primitive data types safely into ChromaDB
    collection.add(
        documents=[chunk.page_content], 
        metadatas=[metadata], 
        ids=[chunk_id], 
        embeddings=[embedding]
    )



print(f'number of chunks added: {i+1}/{len(final_chunks)}', end='\r')
print(f'collection count: {collection.count()}  ', end='\r')

