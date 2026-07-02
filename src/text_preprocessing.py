import re

def preprocess_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_texts(texts):
    return [preprocess_text(text) for text in texts]


if __name__ == '__main__':
    sample_text = "This is an example of text preprocessing for a resume screening system."

    processed_text = preprocess_text(sample_text)

    print("Original Text:", sample_text)
    print("Processed Text:", processed_text)
