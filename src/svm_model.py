import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

try:
    from src.text_extraction import extract_text_from_file
    from src.text_preprocessing import preprocess_text
except ModuleNotFoundError:
    from text_extraction import extract_text_from_file
    from text_preprocessing import preprocess_text

def train_svm_classifier(data, labels):
    """
    Trains an SVM classifier on the given text data and labels.

    Args:
        data (list of str): A list of preprocessed text documents (resumes).
        labels (list of int): A list of labels (1 for good match, 0 for poor match).

    Returns:
        tuple: A trained SVM classifier and the TfidfVectorizer instance.
    """
    # Vectorize the text data
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)

    # Train the SVM classifier
    # Using class_weight='balanced' helps with imbalanced datasets
    classifier = SVC(kernel='linear', probability=True, class_weight='balanced')
    classifier.fit(X, labels)

    return classifier, vectorizer

def _label_from_filename(filename, partial_label):
    name = filename.lower()
    if "good" in name:
        return 1
    if "poor" in name:
        return 0
    if "partial" in name:
        if partial_label == "exclude":
            return None
        return 1 if partial_label == "good" else 0
    return None

def _combine_jd_resume(jd_text, resume_text):
    return f"jd {jd_text} resume {resume_text}"

def train_svm_from_folder(jd_path, resumes_folder, partial_label="exclude"):
    if partial_label not in {"exclude", "good", "poor"}:
        raise ValueError("partial_label must be 'exclude', 'good', or 'poor'.")

    jd_text = extract_text_from_file(jd_path)
    if "Unsupported file type" in jd_text:
        raise ValueError("Training job description file type is not supported.")

    processed_jd = preprocess_text(jd_text)

    training_texts = []
    labels = []

    for filename in os.listdir(resumes_folder):
        file_path = os.path.join(resumes_folder, filename)
        if os.path.isdir(file_path):
            continue

        label = _label_from_filename(filename, partial_label)
        if label is None:
            continue

        resume_text = extract_text_from_file(file_path)
        if "Unsupported file type" in resume_text:
            continue

        processed_resume = preprocess_text(resume_text)
        training_texts.append(_combine_jd_resume(processed_jd, processed_resume))
        labels.append(label)

    if not training_texts:
        raise ValueError("No labeled resumes found for training.")

    return train_svm_classifier(training_texts, labels)

def predict_match(classifier, vectorizer, jd_text, resume_text):
    """
    Predicts if a resume is a good match using the trained SVM classifier.

    Args:
        classifier (SVC): The trained SVM model.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        jd_text (str): The job description text.
        resume_text (str): The resume text.

    Returns:
        tuple: The prediction label (1 or 0) and the prediction probability.
    """
    processed_jd = preprocess_text(jd_text)
    processed_resume = preprocess_text(resume_text)
    combined_text = _combine_jd_resume(processed_jd, processed_resume)

    # Vectorize the combined text
    resume_vector = vectorizer.transform([combined_text])

    # Predict the class
    prediction = classifier.predict(resume_vector)
    
    # Get the probability of the prediction
    probability = classifier.predict_proba(resume_vector)

    return prediction[0], probability[0]

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    jd_path = os.path.join(base_dir, "data", "job_descriptions", "job_description.txt")
    resumes_folder = os.path.join(base_dir, "data", "resumes")

    print("--- Training SVM Classifier ---")
    svm_classifier, tfidf_vectorizer = train_svm_from_folder(
        jd_path,
        resumes_folder,
        partial_label="exclude"
    )
    print("Training complete.")
    print("-----------------------------\n")

    print("--- Making Predictions on New Resumes ---")
    jd_text = extract_text_from_file(jd_path)

    new_resume_good = "A machine learning engineer with python and scikit-learn experience."
    prediction, probability = predict_match(svm_classifier, tfidf_vectorizer, jd_text, new_resume_good)
    print(f"Resume: '{new_resume_good}'")
    print(f"Prediction: {'Good Match' if prediction == 1 else 'Poor Match'}")
    print(f"Confidence: {probability[prediction]:.2%}\n")

    new_resume_poor = "I am a chef with ten years of experience in French cuisine."
    prediction, probability = predict_match(svm_classifier, tfidf_vectorizer, jd_text, new_resume_poor)
    print(f"Resume: '{new_resume_poor}'")
    print(f"Prediction: {'Good Match' if prediction == 1 else 'Poor Match'}")
    print(f"Confidence: {probability[prediction]:.2%}\n")

    print("---------------------------------------")
