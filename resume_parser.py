import re
import json

def normalize_resume_text(resume_text: str) -> str:
    """Fix headers like 'P R O J E C T' -> 'PROJECT'"""
    fixed_lines = []
    for line in resume_text.splitlines():
        if re.fullmatch(r'([A-Z]\s+)+[A-Z]', line.strip()):
            fixed_lines.append(line.replace(" ", ""))
        else:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)


def extract_resume_sections(resume_text: str):
    """Resume parser: cgpa, skills, certs, projects, internships, achievements"""
    resume_text = normalize_resume_text(resume_text)
    lines = resume_text.strip().split("\n")

    result = {
        "cgpa": "",
        "skills": [],
        "certifications": [],
        "projects": [],
        "internships": [],
        "achievements": []
    }

    # -------- Extract CGPA --------
    cgpa_match = re.search(r'CGPA[:\s]*([0-9]\.?[0-9]?)', resume_text, re.IGNORECASE)
    if cgpa_match:
        result["cgpa"] = cgpa_match.group(1)

    # -------- Extract Skills --------
    def extract_skills(lines):
        skills_list = []
        capture = False
        buffer = ""

        for line in lines:
            stripped = line.strip()
        
        # Start capturing after SKILLS header
            if re.match(r'^SKILLS$', stripped, re.IGNORECASE):
                capture = True
                continue
        
        # Stop capturing when a new ALL CAPS section starts
            if capture and re.fullmatch(r'([A-Z][A-Z ]+)', stripped) and stripped.upper() != "SKILLS":
                capture = False
        
            if capture:
            # Remove bullet points
                stripped = stripped.lstrip("•").strip()
            
            # Merge lines starting with comma into previous buffer
                if stripped.startswith(",") or stripped.startswith("|"):
                    buffer += " " + stripped[1:].strip()
                else:
                    if buffer:
                        # Split by comma, pipe, or newline
                        parts = re.split(r',|\||\n', buffer)
                        for part in parts:
                            part = part.strip()
                            if part:
                                skills_list.append(part)
                    buffer = stripped

        # Capture any remaining buffer
        if buffer:
            parts = re.split(r',|\||\n', buffer)
            for part in parts:
                part = part.strip()
                if part:
                    skills_list.append(part)
    
        # Remove duplicates and normalize
        skills_list = list(dict.fromkeys([s for s in skills_list if s]))
        return skills_list


    result["skills"] = extract_skills(lines)

    # -------- Extract Certifications --------
    def extract_certs(lines):
        cert_list = []
        capture = False
        buffer = ""
        for line in lines:
            stripped = line.strip()
            if re.match(r'^CERTIFICATIONS$', stripped, re.IGNORECASE):
                capture = True
                continue
            if capture and re.fullmatch(r'([A-Z][A-Z ]+)', stripped) and stripped.upper() != "CERTIFICATIONS":
                capture = False
            if capture:
                if stripped.startswith(","):
                    buffer += " " + stripped[1:].strip()
                else:
                    if buffer:
                        cert_list.append(buffer.strip())
                    buffer = stripped
        if buffer:
            cert_list.append(buffer.strip())
        return cert_list

    result["certifications"] = extract_certs(lines)

    # -------- Section detection helper --------
    def find_section_start(keywords):
        for i, line in enumerate(lines):
            stripped = line.strip()
            upper = stripped.upper()
            if not stripped:
                continue
            if any(kw in upper for kw in keywords):
                if len(stripped.split()) <= 6 and (stripped == upper or stripped.isupper()):
                    return i
        return None

    sections_idx = {
        "internships": find_section_start(["EXPERIENCE", "INTERNSHIP", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE"]),
        "projects": find_section_start(["PROJECT"]),
        "achievements": find_section_start(["ACHIEVEMENT", "AWARD", "HONOR"])
    }

    def get_section_bounds(section_key):
        start = sections_idx[section_key]
        if start is None:
            return None, None
        end = len(lines)
        stop_keywords = ["EDUCATION", "SKILLS", "CERTIFICATIONS", "LANGUAGE", "ORGANIZATION",
                         "SUMMARY", "ABOUT", "PROFILE", "REFERENCES"]
        for other_key, other_start in sections_idx.items():
            if other_start and other_start > start and other_start < end:
                end = other_start
        for i in range(start + 1, len(lines)):
            upper = lines[i].strip().upper()
            if len(lines[i].strip().split()) <= 6 and any(kw in upper for kw in stop_keywords):
                end = min(end, i)
                break
        return start + 1, end

    def split_entries(start, end):
        if start is None or end is None:
            return []
        entries, current = [], []
        for i in range(start, end):
            line = lines[i]
            if not line.strip():
                if current:
                    entries.append(current)
                    current = []
            else:
                current.append(line)
        if current:
            entries.append(current)
        return entries

    def is_bullet(line):
        return line.strip().startswith(("•", "-", "*", "·", "–", "—"))

    def is_likely_title(line):
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            return False
        if is_bullet(stripped):
            return False
        return True

    def parse_experience_entry(entry_lines):
        entry = {"title": "", "company": "", "date": "", "description": ""}
        desc = []
        title_found = False
        for line in entry_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if not title_found and is_likely_title(stripped):
                entry["title"] = stripped
                title_found = True
            elif title_found and not entry["company"] and not is_bullet(stripped):
                entry["company"] = stripped
            else:
                desc.append(stripped)
        entry["description"] = " ".join(desc).strip()
        return entry if entry["title"] else None

    def parse_achievements(entry_lines):
        achievements = []
        current = []
        for line in entry_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r'^(Runner-Up|Winner|First Prize|Second Prize|Member|Finalist|Selected)\b', stripped):
                if current:
                    achievements.append({
                        "title": " ".join(current).strip(),
                        "organization": "",
                        "description": ""
                    })
                    current = []
                current.append(stripped)
            else:
                current.append(stripped)
        if current:
            achievements.append({
                "title": " ".join(current).strip(),
                "organization": "",
                "description": ""
            })
        return achievements

    # -------- Parse Projects --------
    start, end = get_section_bounds("projects")
    if start and end:
        for e in split_entries(start, end):
            parsed = parse_experience_entry(e)
            if parsed:
                result["projects"].append(parsed)

    # -------- Parse Internships --------
    start, end = get_section_bounds("internships")
    if start and end:
        entries = split_entries(start, end)
    else:
        entries = [[line] for line in lines if re.search(r'\bIntern\b', line, re.IGNORECASE)]
    for e in entries:
        parsed = parse_experience_entry(e)
        if parsed:
            result["internships"].append(parsed)

    # -------- Parse Achievements --------
    start, end = get_section_bounds("achievements")
    if start and end:
        for e in split_entries(start, end):
            parsed = parse_achievements(e)
            if parsed:
                result["achievements"].extend(parsed)

    return result


# ------------------ TEST ------------------
def parse_resume_text(resume_text):
    
    return extract_resume_sections(resume_text)