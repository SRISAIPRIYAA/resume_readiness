# ----------------------------- IMPORTS -----------------------------
from pathlib import Path
import pdfplumber
import docx
import re
from io import BytesIO

# -------------------------- DOMAIN SKILLS --------------------------
DOMAIN_SKILLS = {
    "web_dev": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Django", "Flask", "Bootstrap", "SQL", "REST APIs"],
    "ai/ml": ["Python", "NumPy", "Pandas", "Scikit-learn", "TensorFlow", "PyTorch", "Matplotlib", "Seaborn", "ML algorithms"],
    "ds": ["Python", "SQL", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Data visualization", "Statistics", "ETL"],
    "cybersecurity": ["Networking", "Penetration testing", "Wireshark", "Python", "Linux", "Firewalls", "Cryptography", "Vulnerability assessment"],
    "devops": ["Docker", "Kubernetes", "CI/CD", "Git", "Jenkins", "AWS", "Terraform", "Linux", "Monitoring tools"]
}

# ------------------------ STUDY RESOURCES --------------------------
RESOURCES = {
    "HTML": "https://www.w3schools.com/html/",
    "CSS": "https://www.w3schools.com/css/",
    "JavaScript": "https://www.javascript.info/",
    "React": "https://reactjs.org/docs/getting-started.html",
    "Node.js": "https://nodejs.dev/learn",
    "Django": "https://docs.djangoproject.com/en/4.2/intro/tutorial01/",
    "Flask": "https://flask.palletsprojects.com/en/2.3.x/tutorial/",
    "Bootstrap": "https://getbootstrap.com/docs/5.0/getting-started/introduction/",
    "SQL": "https://www.w3schools.com/sql/",
    "REST APIs": "https://restfulapi.net/",
    "Python": "https://www.learnpython.org/",
    "NumPy": "https://numpy.org/doc/stable/user/quickstart.html",
    "Pandas": "https://pandas.pydata.org/docs/getting_started/index.html",
    "Scikit-learn": "https://scikit-learn.org/stable/tutorial/index.html",
    "TensorFlow": "https://www.tensorflow.org/tutorials",
    "PyTorch": "https://pytorch.org/tutorials/",
    "Matplotlib": "https://matplotlib.org/stable/tutorials/introductory/pyplot.html",
    "Seaborn": "https://seaborn.pydata.org/tutorial.html",
    "ML algorithms": "https://www.coursera.org/learn/machine-learning",
    "Data visualization": "https://www.datacamp.com/courses/introduction-to-data-visualization-with-python",
    "Statistics": "https://www.khanacademy.org/math/statistics-probability",
    "ETL": "https://www.talend.com/resources/what-is-etl/",
    "Networking": "https://www.cisco.com/c/en/us/solutions/enterprise-networks/what-is-networking.html",
    "Penetration testing": "https://www.eccouncil.org/programs/certified-ethical-hacker-ceh/",
    "Wireshark": "https://www.wireshark.org/docs/wsug_html_chunked/",
    "Linux": "https://linuxjourney.com/",
    "Firewalls": "https://www.cloudflare.com/learning/ddos/what-is-a-firewall/",
    "Cryptography": "https://www.khanacademy.org/computing/computer-science/cryptography",
    "Vulnerability assessment": "https://owasp.org/www-project-top-ten/",
    "Docker": "https://docs.docker.com/get-started/",
    "Kubernetes": "https://kubernetes.io/docs/tutorials/",
    "CI/CD": "https://www.redhat.com/en/topics/devops/what-is-ci-cd",
    "Git": "https://git-scm.com/book/en/v2",
    "Jenkins": "https://www.jenkins.io/doc/",
    "AWS": "https://aws.amazon.com/getting-started/",
    "Terraform": "https://developer.hashicorp.com/terraform/tutorials"
}

# ------------------------ DOMAIN MAPPING --------------------------
UI_TO_INTERNAL_DOMAIN = {
    "Web Development": "web_dev",
    "AI/ML": "ai/ml",
    "Data Science": "ds",
    "Cybersecurity": "cybersecurity",
    "DevOps": "devops"
}

# -------------------------- RESUME READING -------------------------
def read_resume(file) -> str:
    """Accepts either a file path or BytesIO (uploaded file)"""
    text = ""
    
    if isinstance(file, BytesIO):
        # Handle PDF bytes
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    else:
        # Handle Path files (.pdf, .docx, .txt)
        file_path = Path(file)
        if file_path.suffix.lower() == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        elif file_path.suffix.lower() == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + " "
        elif file_path.suffix.lower() == ".txt":
            text = file_path.read_text()
        else:
            raise ValueError("Unsupported file type")
    
    return text

# --------------------------- SKILL EXTRACTION ----------------------
def extract_skills(resume_text: str) -> list:
    resume_text = resume_text.lower()
    skills_found = set()
    for domain_skills in DOMAIN_SKILLS.values():
        for skill in domain_skills:
            if re.search(r"\b" + re.escape(skill.lower()) + r"\b", resume_text):
                skills_found.add(skill)
    return list(skills_found)

# --------------------------- RECOMMENDATION ------------------------
def recommend_skills(candidate_skills: list, chosen_domain: str, max_skills: int = 3) -> dict:
    internal_domain = UI_TO_INTERNAL_DOMAIN.get(chosen_domain)
    if not internal_domain:
        return {}
    missing_skills = [skill for skill in DOMAIN_SKILLS[internal_domain] if skill not in candidate_skills]
    missing_skills = missing_skills[:max_skills]
    recommendations = {skill: RESOURCES.get(skill, "Search online for tutorials") for skill in missing_skills}
    return recommendations

# --------------------------- MAIN FUNCTION -------------------------
def analyze_resume(file, chosen_domain: str):
    text = read_resume(file)
    candidate_skills = extract_skills(text)
    recommendations = recommend_skills(candidate_skills, chosen_domain)
    
    return {
        "candidate_skills": candidate_skills,
        "missing_skills": list(recommendations.keys()),
        "recommendations": recommendations
    }
