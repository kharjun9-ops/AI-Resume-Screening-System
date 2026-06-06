import os
import time
from functools import partial
from multiprocessing import Pool
from text_extraction import extract_text_from_file
from text_preprocessing import preprocess_text
from feature_extraction import vectorize_text
from sklearn.metrics.pairwise import cosine_similarity

def process_resume(resumes_dir, resume_file):
    """
    Extracts and preprocesses text from a single resume file.

    Args:
        resumes_dir (str): The path to the directory containing resumes.
        resume_file (str): The filename of the resume.

    Returns:
        tuple: A tuple containing the resume filename and the processed text.
               Returns (None, None) if the file cannot be processed.
    """
    resume_path = os.path.join(resumes_dir, resume_file)
    resume_text = extract_text_from_file(resume_path)
    if "File not found" in resume_text or "Unsupported file type" in resume_text:
        return None, None
    return resume_file, preprocess_text(resume_text)

def screen_resumes(job_description_path, resumes_dir):
    """
    Screens resumes against a job description and ranks them by similarity.

    Args:
        job_description_path (str): The path to the job description file.
        resumes_dir (str): The path to the directory containing resumes.

    Returns:
        list of tuple: A list of (resume_filename, similarity_score) tuples,
                       sorted in descending order of similarity.
    """
    # 1. Load and process the job description
    job_description_text = extract_text_from_file(job_description_path)
    if "File not found" in job_description_text or "Unsupported file type" in job_description_text:
        print(f"Error with job description file: {job_description_text}")
        return []
    
    processed_jd = preprocess_text(job_description_text)

    # 2. Load and process resumes in parallel
    resume_files = [f for f in os.listdir(resumes_dir) if os.path.isfile(os.path.join(resumes_dir, f))]
    if not resume_files:
        print(f"No resumes found in the directory: {resumes_dir}")
        return []

    # Use a pool of worker processes to parallelize resume processing
    with Pool() as pool:
        # We use partial to create a version of process_resume with the resumes_dir argument fixed
        process_func = partial(process_resume, resumes_dir)
        # Map the processing function to all resume files
        results = pool.map(process_func, resume_files)

    # Filter out any resumes that failed to process and create a dictionary
    processed_resumes = {file: text for file, text in results if file is not None}

    # 3. Vectorize all texts together
    all_texts = [processed_jd] + list(processed_resumes.values())
    tfidf_matrix, vectorizer = vectorize_text(all_texts)

    # 4. Calculate cosine similarity
    # The first vector is the job description, the rest are resumes
    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    
    similarity_scores = cosine_similarity(jd_vector, resume_vectors)

    # 5. Rank resumes
    ranked_resumes = sorted(zip(processed_resumes.keys(), similarity_scores[0]), key=lambda item: item[1], reverse=True)

    return ranked_resumes

if __name__ == '__main__':
    # Define paths
    # Note: These paths are relative to the 'src' directory where this script is located.
    # We navigate up one level ('..') to get to the project root.
    JOB_DESCRIPTION_PATH = os.path.join('..', 'data', 'job_descriptions', 'job_description.txt')
    RESUMES_DIR = os.path.join('..', 'data', 'resumes')

    # Create dummy files if they don't exist
    if not os.path.exists(JOB_DESCRIPTION_PATH):
        os.makedirs(os.path.dirname(JOB_DESCRIPTION_PATH), exist_ok=True)
        with open(JOB_DESCRIPTION_PATH, 'w') as f:
            f.write("We are looking for a Python developer with experience in machine learning and natural language processing. Key skills include scikit-learn, pandas, and NLTK.")

    if not os.path.exists(os.path.join(RESUMES_DIR, 'resume1_good_match.txt')):
        os.makedirs(RESUMES_DIR, exist_ok=True)
        with open(os.path.join(RESUMES_DIR, 'resume1_good_match.txt'), 'w') as f:
            f.write("Experienced Python developer with a background in machine learning. Proficient in scikit-learn, NLTK, and pandas for natural language processing tasks.")
    
    if not os.path.exists(os.path.join(RESUMES_DIR, 'resume2_partial_match.txt')):
        with open(os.path.join(RESUMES_DIR, 'resume2_partial_match.txt'), 'w') as f:
            f.write("Software engineer with a focus on web development. Some experience with Python and pandas.")

    if not os.path.exists(os.path.join(RESUMES_DIR, 'resume3_poor_match.txt')):
        with open(os.path.join(RESUMES_DIR, 'resume3_poor_match.txt'), 'w') as f:
            f.write("Graphic designer with skills in Adobe Photoshop and Illustrator. No programming experience.")

    # Run the screening process and time it
    start_time = time.time()
    ranked_candidates = screen_resumes(JOB_DESCRIPTION_PATH, RESUMES_DIR)
    end_time = time.time()

    # Print the results
    if ranked_candidates:
        print("--- Resume Screening Results ---")
        for i, (resume, score) in enumerate(ranked_candidates):
            print(f"{i+1}. {resume}: Score = {score:.4f}")
        print("---------------------------------")
        print(f"Screening completed in {end_time - start_time:.2f} seconds.")
