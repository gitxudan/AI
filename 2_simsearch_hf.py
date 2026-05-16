from sentence_transformers import SentenceTransformer

sentences = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings.shape)


similarity_matrix = model.similarity(embeddings, embeddings)
print(similarity_matrix)

query = ['document']
query_embedding = model.encode(query)
similarity_scores = model.similarity(query_embedding, embeddings).flatten()

# Find most relevant document
best_idx = similarity_scores.argmax()
best_score = similarity_scores[best_idx]
best_sentence = sentences[best_idx]

print("Similarity scores:", similarity_scores)
print(f"Most relevant doc: '{best_sentence}' with score {best_score:.4f}")

print(similarity_scores)