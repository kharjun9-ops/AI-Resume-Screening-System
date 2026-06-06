import os

import spacy


def _load_spacy_model():
    """Loads spaCy pipeline once, with heavy components disabled for speed."""
    try:
        # We only need tokenization + lemmatization + stopword flags.
        # Disabling parser/NER makes preprocessing much faster for long resumes.
        model = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        from spacy.cli import download

        download("en_core_web_sm")
        model = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Avoid errors for longer documents.
    model.max_length = max(model.max_length, int(os.getenv("SPACY_MAX_LENGTH", "2000000")))
    return model


nlp = _load_spacy_model()


def _preprocess_doc(doc):
    return " ".join(
        token.lemma_.lower().strip()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    )

def preprocess_text(text):
    """Cleans and preprocesses the input text using spaCy."""
    if not text:
        return ""
    return _preprocess_doc(nlp(text))


def preprocess_texts(texts, batch_size=64):
    """Batch version of preprocess_text for much faster throughput."""
    if not texts:
        return []

    safe_texts = [(text or "") for text in texts]
    docs = nlp.pipe(safe_texts, batch_size=batch_size)
    return [_preprocess_doc(doc) for doc in docs]

if __name__ == '__main__':
    # Example Usage
    sample_text = "This is an example of text preprocessing for a resume screening system. It involves cleaning, tokenizing, and lemmatizing the text."
    
    processed_text = preprocess_text(sample_text)
    print("Original Text:", sample_text)
    print("Processed Text:", processed_text)
