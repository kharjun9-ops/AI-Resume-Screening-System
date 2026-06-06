# A sample list of skills. In a real-world application, this would be much more extensive.
HARD_SKILLS = [
    'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'html', 'css', 'sql', 'nosql', 'mongodb', 'postgresql', 'git', 'docker',
    'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'fastapi', 'spring', '.net',
    'machine learning', 'natural language processing', 'nlp', 'deep learning', 'computer vision',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'scipy',
    'matplotlib', 'seaborn', 'plotly', 'd3.js', 'data analysis', 'data visualization', 'data science',
    'statistics', 'a/b testing', 'aws', 'azure', 'gcp', 'google cloud', 'kubernetes', 'terraform', 'ansible',
    'ci/cd', 'jenkins', 'github actions', 'cybersecurity', 'penetration testing', 'network security'
]

SOFT_SKILLS = [
    'communication', 'teamwork', 'collaboration', 'problem solving', 'project management',
    'agile', 'scrum', 'kanban', 'leadership', 'mentoring', 'adaptability', 'creativity', 'innovation',
    'critical thinking', 'analytical thinking', 'time management', 'work ethic', 'interpersonal skills',
    'emotional intelligence', 'conflict resolution', 'negotiation'
]

from collections import Counter

def extract_skills(text, skill_list=None):
    """Extracts skills from a text based on a provided skill list (case-insensitive)."""
    if skill_list is None:
        skill_list = HARD_SKILLS + SOFT_SKILLS

    text_lower = text.lower()
    found_skills = [skill for skill in skill_list if skill.lower() in text_lower]
    return list(dict.fromkeys(found_skills))

def get_skill_frequencies(text):
    skills = extract_skills(text)
    return Counter(skills)
