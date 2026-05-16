from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

sentences = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

# keyword query
query = ['document']

# --- TF-IDF Vectorizer ---
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(sentences)
query_vec = tfidf_vectorizer.transform(query)
tfidf_scores = cosine_similarity(query_vec, X).flatten()

# --- BM25 ---
tokenized_corpus = [s.lower().split() for s in sentences]
bm25 = BM25Okapi(tokenized_corpus)
bm25_scores = bm25.get_scores(query[0].split())

# --- Weighted combination ---
alpha = 0.5
beta = 0.5
combined_scores = alpha * tfidf_scores + beta * bm25_scores

# Results
for i, sentence in enumerate(sentences):
    print(f"Sentence: {sentence}")
    print(f"TF-IDF Score: {tfidf_scores[i]:.4f}")
    print(f"BM25 Score: {bm25_scores[i]:.4f}")
    print(f"Combined Score: {combined_scores[i]:.4f}")
    print("-" * 50)

        