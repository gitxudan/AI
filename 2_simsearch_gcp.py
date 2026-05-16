import os
from google import genai
import numpy as np
sentences = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GCP_KEY/orchestrator-lab-dq-202601-cred.json'
client = genai.Client( vertexai=True )

embeddingModel = "text-embedding-005"

embeddings=[]
for sentence in sentences:
    response = client.models.embed_content(
        model=embeddingModel,
        contents=sentence
    )
    embedding = response.embeddings[0].values
    embeddings.append(embedding)
    
print(len(embeddings))

if len(embeddings) > 0:
    print(f"Embedding dimension: {len(embeddings[0])}")
    

def normalize(vec):
    return vec / np.linalg.norm(vec)

def dot_similarity(vecA, vecB):
    return np.dot(vecA, vecB)

query = "documents"
response = client.models.embed_content(
        model=embeddingModel,
        contents=query
    )
query_embedding = response.embeddings[0].values

for emb in embeddings:
    score = dot_similarity(normalize(query_embedding), normalize(emb))
    print(f"Similarity score: {score:.4f}")

print(f"Similarity score between doc 1 and doc 2: {score:.4f}")



for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        score = dot_similarity(normalize(embeddings[i]), normalize(embeddings[j]))
        print(f"Similarity between '{sentences[i]}' and '{sentences[j]}': {score:.4f}")
