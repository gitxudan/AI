from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

sentences = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

tokenized_sentences = [doc.lower().split() for doc in sentences]

# 3. Initialize the BM25 object with the tokenized sentences
bm25 = BM25Okapi(tokenized_sentences)

print(tokenized_sentences)

# 4. Define a query and tokenize it
query = 'document'
tokenized_query = query.lower().split(" ")


# 5. Get document scores for the query
doc_scores = bm25.get_scores(tokenized_query)
print(f'scores for each document: - {doc_scores}')

# 6. Retrieve thetop N documents
top_n = bm25.get_top_n(tokenized_query, sentences, n=2)
print(f'Top 2 documents: - {top_n}')
