from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)
feature_names = vectorizer.get_feature_names_out()
print(X.shape)
print(feature_names)
print(X.toarray())

keywords = ['document']
keyword_vec = vectorizer.transform(keywords)
print(f'vectore of {keywords[0]} is - {keyword_vec.toarray()}')

similarities = cosine_similarity(keyword_vec, X)
print(f'similiarities are - {similarities}')