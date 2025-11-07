import os
import numpy as np
import pdfplumber
import pandas as pd
from joblib import load
from resume_parser import parse_resume_text

SKILL_SCORES = {
    "web development": {
        "html": 1.0, "html5": 1.0, "css": 1.0, "css3": 1.0, "javascript": 1.0,
        "js": 1.0, "typescript": 0.5, "react": 0.75, "reactjs": 0.75, "angular": 0.5,
        "vue": 0.5, "next.js": 0.5, "nuxt": 0.25, "node": 0.75, "node.js": 0.75,
        "express": 0.75, "django": 0.5, "flask": 0.5, "bootstrap": 0.5, "tailwind": 0.5,
        "jquery": 0.25, "php": 0.5, "laravel": 0.5, "spring": 0.5, "graphql": 0.5,
        "rest api": 0.75, "websocket": 0.5, "sass": 0.25, "scss": 0.25, "less": 0.25,
        "webpack": 0.25, "babel": 0.25, "eda": 0.25, "responsive design": 0.5,
        "web security": 0.5, "testing": 0.5, "unit testing": 0.5, "integration testing": 0.5,
        "seo": 0.5, "performance optimization": 0.5, "cross browser compatibility": 0.25,
        "web animation": 0.25, "ajax": 0.25, "json": 0.25, "api integration": 0.5,
        "progressive web app": 0.5, "web accessibility": 0.25, "cms": 0.25, "wordpress": 0.25,
        "drupal": 0.25, "joomla": 0.25, "web performance": 0.5, "frontend": 0.5, "backend": 0.5
    },
    "devops": {
        "docker": 1.0, "docker-compose": 0.75, "kubernetes": 1.0, "k8s": 1.0,
        "jenkins": 0.75, "ansible": 0.5, "terraform": 0.5, "ci/cd": 1.0, "aws": 1.0,
        "amazon web services": 1.0, "azure": 0.75, "gcp": 0.75, "google cloud": 0.75,
        "linux administration": 0.5, "bash scripting": 0.5, "shell scripting": 0.5,
        "helm": 0.25, "git": 1.0, "gitlab": 0.5, "github": 0.5, "prometheus": 0.5,
        "grafana": 0.5, "nginx": 0.5, "apache": 0.5, "load balancing": 0.5,
        "monitoring": 0.5, "infrastructure as code": 0.75, "cloudformation": 0.5,
        "elastic stack": 0.25, "elk": 0.25, "kibana": 0.25, "eda": 0.25,
        "scripting automation": 0.5, "log analysis": 0.5, "container monitoring": 0.5,
        "system optimization": 0.5, "security compliance": 0.5, "continuous delivery": 0.75,
        "continuous integration": 0.75, "serverless": 0.5, "microservices": 0.5,
        "high availability": 0.5, "disaster recovery": 0.5, "scaling": 0.5,
        "alerting": 0.25, "backup automation": 0.25, "performance tuning": 0.5,
        "configuration management": 0.5, "deployment automation": 0.75, "cloud migration": 0.5
    },
    "ai/ml": {
        "machine learning": 1.0, "ml": 1.0, "deep learning": 0.75, "dl": 0.75,
        "tensorflow": 0.75, "pytorch": 0.75, "scikit-learn": 0.75, "sklearn": 0.75,
        "keras": 0.5, "nlp": 0.5, "natural language processing": 0.5, "computer vision": 0.5,
        "opencv": 0.5, "reinforcement learning": 0.5, "rl": 0.5, "transformers": 0.5,
        "huggingface": 0.5, "llm": 0.5, "time series": 0.5, "feature engineering": 0.75,
        "xgboost": 0.5, "lightgbm": 0.5, "catboost": 0.5, "data preprocessing": 0.75,
        "data cleaning": 0.75, "eda": 0.5, "exploratory data analysis": 0.5,
        "model deployment": 0.5, "mlops": 0.5, "statistical modeling": 0.5,
        "bayesian methods": 0.25, "gradient boosting": 0.5, "random forest": 0.5,
        "svm": 0.25, "decision tree": 0.25, "logistic regression": 0.25,
        "classification": 0.5, "regression": 0.5, "clustering": 0.5,
        "dimensionality reduction": 0.25, "unsupervised learning": 0.5,
        "supervised learning": 0.5, "cnn": 0.25, "rnn": 0.25, "lstm": 0.25,
        "gan": 0.25, "autoencoder": 0.25, "hyperparameter tuning": 0.5
    },
    "data science": {
        "pandas": 1.0, "numpy": 1.0, "matplotlib": 0.75, "seaborn": 0.75, "sql": 1.0,
        "power bi": 0.5, "tableau": 0.5, "excel": 0.5, "data cleaning": 1.0,
        "data wrangling": 0.75, "data visualization": 0.75, "statistics": 0.5,
        "hypothesis testing": 0.5, "r programming": 0.25, "bigquery": 0.25,
        "spark": 0.5, "hadoop": 0.5, "etl": 0.75, "data pipeline": 0.5,
        "feature selection": 0.5, "regression": 0.5, "classification": 0.5,
        "clustering": 0.5, "dash": 0.25, "plotly": 0.25, "time series analysis": 0.5,
        "eda": 0.5, "exploratory data analysis": 0.5, "correlation analysis": 0.25,
        "data preprocessing": 0.75, "data modeling": 0.5, "data aggregation": 0.25,
        "descriptive statistics": 0.25, "inferential statistics": 0.25,
        "data reporting": 0.25, "business intelligence": 0.5, "dashboard creation": 0.5,
        "report automation": 0.25, "trend analysis": 0.25, "predictive analytics": 0.5,
        "sql analysis": 0.5, "kpi reporting": 0.25, "dataset handling": 0.5,
        "data cleaning techniques": 0.5, "data manipulation": 0.5,
        "data summarization": 0.5, "data insights": 0.5, "data storytelling": 0.25,
        "data transformation": 0.5
    },
    "cybersecurity": {
        "penetration testing": 1.0, "ethical hacking": 1.0, "network security": 0.75,
        "cryptography": 0.75, "vulnerability assessment": 0.5, "firewall configuration": 0.5,
        "wireshark": 0.5, "nmap": 0.5, "burp suite": 0.5, "metasploit": 0.5,
        "intrusion detection": 0.5, "incident response": 0.5, "malware analysis": 0.5,
        "security operations": 0.5, "siem": 0.5, "kali linux": 0.5, "threat hunting": 0.5,
        "risk assessment": 0.25, "web application security": 0.5, "forensics": 0.25,
        "cloud security": 0.5, "identity and access management": 0.5, "zero trust": 0.25,
        "public key infrastructure": 0.25, "tls": 0.25, "ssl": 0.25, "eda": 0.25,
        "security compliance": 0.5, "penetration test automation": 0.5,
        "incident reporting": 0.25, "security auditing": 0.5, "security monitoring": 0.5,
        "security assessment": 0.5, "penetration testing lab": 0.25,
        "ethical hacking lab": 0.25, "cyber defense": 0.5, "cyber risk": 0.25,
        "firewall management": 0.25, "endpoint security": 0.5, "network monitoring": 0.5,
        "vulnerability scanning": 0.5, "incident simulation": 0.25, "security policy": 0.25,
        "threat modeling": 0.25, "security tools": 0.25, "penetration tools": 0.25,
        "digital forensics": 0.25, "malware reverse engineering": 0.25
    }
}

PREMIUM_CERTS = [
    'google', 'aws', 'microsoft', 'azure', 'cisco', 'oracle', 'pmp', 'scrum master',
    'cissp', 'ceh', 'oscp', 'vmware', 'red hat', 'ibm data science', 'ibm ai',
    'aws certified solutions architect', 'aws certified developer', 'aws certified sysops',
    'microsoft azure fundamentals', 'microsoft azure administrator', 'google cloud professional',
    'google cloud architect', 'google cloud data engineer', 'salesforce', 'sap',
    'tableau certified', 'power bi certified', 'hadoop', 'cloudera', 'databricks',
    'kaggle grandmaster', 'datacamp expert', 'tensorflow developer', 'pytorch expert',
    'compTIA security+', 'compTIA network+', 'linux foundation', 'docker certified',
    'kubernetes certified', 'jenkins certified', 'redhat certified engineer',
    'redhat certified architect', 'aws machine learning specialty', 'azure AI engineer',
    'cisco certified network professional', 'oracle cloud certified', 'oracle DBA',
    'project management professional', 'lean six sigma', 'itil', 'penetration testing professional',
    'ethical hacking', 'cybersecurity analyst', 'big data professional', 'ai engineer',
    'deep learning specialization', 'nlp specialization', 'data engineering', 'devops professional',
    'solution architect', 'cloud architect', 'blockchain developer', 'full stack developer',
    'software testing professional', 'agile certified practitioner', 'business analyst professional',
    'machine learning engineer', 'data science professional', 'azure solutions architect', 'google professional data engineer',
    'oracle database administrator', 'sap hana consultant', 'information security manager', 'network security expert', 'iitm', 'iit'
]

MAJOR_ACHIEVEMENTS = [
    'sih', 'smart india hackathon', 'google code jam', 'facebook hacker cup',
    'microsoft imagine cup', 'codeforces global round winner', 'icpc world finals',
    'topcoder open finals', 'kaggle competitions grandmaster', 'google kickstart winner',
    'international math olympiad medalist', 'international informatics olympiad medalist',
    'national coding competition winner', 'national science fair winner',
    'first prize ieee project', 'first prize nss project', 'national innovation award',
    'international robotics competition winner', 'international hackathon winner',
    'ieee hackathon champion', 'acm icpc regional winner', 'top 3 google challenge',
    'national techfest winner', 'research paper accepted in reputed journal',
    'patent granted', 'innovation award', 'national startup challenge winner',
    'international startup competition winner', 'global entrepreneurship award',
    'technology excellence award', 'best research project award', 'young scientist award',
    'national coding olympiad winner', 'international coding olympiad winner',
    'global ai challenge winner', 'national hackathon winner', 'cybersecurity challenge winner',
    'robotics innovation award', 'best machine learning project', 'outstanding contribution award',
    'software innovation award'
]

SKILL_MAPPING = {"html/css": "html", "leadership": "leadership", "event": "event", "management": "management", "teamwork": "teamwork"}

FEATURE_NAMES = [
    "CGPA", "Skill_Score", "Cert_Score", "Project_Score", "Internship_Score",
    "Achievement_Score", "Selected",
    "Internship_Domain_AI/ML", "Internship_Domain_Cybersecurity",
    "Internship_Domain_Data Science", "Internship_Domain_DevOps",
    "Internship_Domain_Web Dev"
]

DOMAIN_LIST = ["Internship_Domain_AI/ML", "Internship_Domain_Cybersecurity", "Internship_Domain_Data Science", "Internship_Domain_DevOps", "Internship_Domain_Web Dev"]

# ----------------- Helper functions -----------------
def one_hot_encode_domain(domain):
    domain_lower = domain.lower()
    encoding = []
    for d in DOMAIN_LIST:
        col_lower = d.lower().replace("internship_domain_", "")
        encoding.append(1 if domain_lower == col_lower else 0)
    return encoding

def score_cgpa(resume_data):
    try:
        cgpa = float(resume_data.get("cgpa", 0))
    except ValueError:
        cgpa = 0
    return min((cgpa / 10) * 5, 5)

def score_skills(resume_data, domain):
    resume_skills = [s.lower().strip() for s in resume_data.get("skills", [])]
    domain_skills = SKILL_SCORES.get(domain.lower(), {})  # use lowercase key only
    score = 0
    for skill in resume_skills:
        skill = SKILL_MAPPING.get(skill, skill)
        for ds_skill in domain_skills:
            if ds_skill in skill:
                score += domain_skills[ds_skill]
                break
    return min(score, 5)


def score_certs(resume_data):
    certifications = resume_data.get("certifications", [])
    score = 0
    for cert in certifications:
        cert_lower = cert.lower()
        if any(pc in cert_lower for pc in PREMIUM_CERTS):
            score += 2
        else:
            score += 0.5
    return min(score, 5)

def score_projects(resume_data, domain):
    projects = resume_data.get("projects", [])
    domain_skills = SKILL_SCORES.get(domain.lower(), {})  # domain key lookup
    score = 0
    for project in projects:
        # Combine all field values into a single lowercase string
        combined_text = " ".join(str(value) for value in project.values()).lower()
        # Check if any domain skill appears in the combined text
        if any(skill in combined_text for skill in domain_skills):
            score += 2
        else:
            score += 1
    return min(score, 5)



def score_internships(resume_data, domain):
    internships = resume_data.get("internships", [])
    domain_key = domain.replace(" ", "_").lower()
    domain_skills = SKILL_SCORES.get(domain_key, {})
    score = 1.5
    for intern in internships:
        text = (intern.get("title", "") + " " + intern.get("description", "")).lower()
        if any(skill in text for skill in domain_skills):
            score += 1.5
        else:
            score += 1
    return min(score, 5)

def score_achievements(resume_data):
    achievements = resume_data.get("achievements", [])
    score = 0
    for ach in achievements:
        ach_lower = (ach.get("title", "") + " " + ach.get("description", "")).lower()
        if any(ma in ach_lower for ma in MAJOR_ACHIEVEMENTS):
            score = 5
            break
        else:
            score += 1
    return min(score, 5)

def calculate_resume_score(resume_data, domain):
    return {
        "cgpa": score_cgpa(resume_data),
        "skills": score_skills(resume_data, domain),
        "certifications": score_certs(resume_data),
        "projects": score_projects(resume_data, domain),
        "internships": score_internships(resume_data, domain),
        "achievements": score_achievements(resume_data)
    }

# ----------------- Load model -----------------
def load_model():
    return load("internship_success_model.joblib")

scor=None
def prepare_features(resume_data, domain):
    scores = calculate_resume_score(resume_data, domain)
    global scor
    scor =scores
    domain_vector = one_hot_encode_domain(domain)
    feature_vector = [
        scores["cgpa"], scores["skills"], scores["certifications"],
        scores["projects"], scores["internships"], scores["achievements"]
    ] + domain_vector
    feature_columns = FEATURE_NAMES[:6] + FEATURE_NAMES[7:]
    return pd.DataFrame([feature_vector], columns=feature_columns)

def score_resume(file_path, domain):
    model = load_model()
    resume_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                resume_text += text + "\n"
    parsed_data = parse_resume_text(resume_text)
    features = prepare_features(parsed_data, domain)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1] * 100 if hasattr(model, "predict_proba") else float(prediction[0]) * 100
    
    return {"scores":scor,"prediction": int(prediction[0]), "probability": round(probability, 2), "analyzed_data": parsed_data}

if __name__=='__main__':
    print(score_resume("resume.pdf","data science"))
