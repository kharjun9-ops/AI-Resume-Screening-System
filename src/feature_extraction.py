from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(texts):
    """
    Converts a list of texts into a matrix of TF-IDF features.
    
    Args:
        texts (list of str): A list of documents to vectorize.
        
    Returns:
        tuple: A tuple containing:
            - scipy.sparse.csr_matrix: The TF-IDF matrix.
            - TfidfVectorizer: The fitted vectorizer instance.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

if __name__ == '__main__':
    # Example Usage
    from text_preprocessing import preprocess_text

    documents = [
        "This is the first document. It contains some words.",
        "This document is the second document. It has different words.",
        "And this is the third one. It is unique.",
        "Is this the first document again? Yes, it is."
    ]

    processed_docs = [preprocess_text(doc) for doc in documents]
    
    tfidf_matrix, vectorizer = vectorize_text(processed_docs)
    
    print("Shape of TF-IDF Matrix:", tfidf_matrix.shape)
    
    # To see the feature names (the vocabulary)
    # print("Feature Names:", vectorizer.get_feature_names_out())
    
    # To see the matrix in a dense format (for small matrices)
    # print("TF-IDF Matrix (dense):\n", tfidf_matrix.toarray())
