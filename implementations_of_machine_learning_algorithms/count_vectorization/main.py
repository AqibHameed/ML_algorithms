
from sklearn.feature_extraction.text import CountVectorizer
if __name__ == '__main__':
    # list of text documents
    text = ["John is a good boy. John watches basketball"]

    vectorizer = CountVectorizer()
    # tokenize and build vocab
    vectorizer.fit(text)

    print(vectorizer.vocabulary_)

    # encode document
    vector = vectorizer.transform(text)
    # summarize encoded vector
    print(vector.shape)
    print(vector.toarray())
