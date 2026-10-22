
import io
import os
import re
import shutil

import streamlit as st
import pytesseract

from PIL import Image
from pypdf import PdfReader
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# OCR CONFIGURATION
# =========================================================

# Try to find Tesseract automatically
TESSERACT_PATH = shutil.which("tesseract")

# Standard Windows installation path
if not TESSERACT_PATH:

    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]

    for path in possible_paths:

        if os.path.exists(path):
            TESSERACT_PATH = path
            break


# Configure Tesseract
if TESSERACT_PATH:

    pytesseract.pytesseract.tesseract_cmd = (
        TESSERACT_PATH
    )


# =========================================================
# POPPLER CONFIGURATION
# =========================================================

# Your Windows Poppler installation
WINDOWS_POPPLER_PATH = (
    r"C:\Users\rohit\Downloads"
    r"\Release-26.09.0-0"
    r"\poppler-26.09.0"
    r"\Library\bin"
)

if os.path.exists(WINDOWS_POPPLER_PATH):

    POPPLER_PATH = WINDOWS_POPPLER_PATH

else:

    # Streamlit Cloud / Linux
    POPPLER_PATH = None


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📄 AI Resume Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze your resume, identify skills, match job '
    'requirements and receive personalized improvement '
    'suggestions.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SKILLS DATABASE
# =========================================================

SKILLS = [

    # Programming
    "python",
    "java",
    "c",
    "c++",
    "javascript",
    "typescript",
    "sql",

    # Web
    "html",
    "css",
    "react",
    "node.js",
    "django",
    "flask",
    "spring",
    "rest api",

    # AI / ML
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analysis",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "nlp",
    "natural language processing",
    "computer vision",

    # Tools
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "linux",

    # Database / Analytics
    "excel",
    "power bi",
    "tableau",
    "mongodb",
    "mysql",
    "postgresql",
    "firebase",

    # Cybersecurity
    "cybersecurity",
    "network security",
    "computer networks",
    "ethical hacking",
    "penetration testing",
    "vulnerability assessment",
    "incident response",
    "cryptography",
    "firewall",
    "vpn",
    "siem",
    "wireshark",
    "nmap",
    "metasploit",
    "kali linux",
    "security monitoring",
    "threat detection",
    "risk assessment",
    "cloud security",

    # Soft Skills
    "communication",
    "leadership",
    "problem solving",
    "teamwork",
    "project management"
]


# =========================================================
# JOB ROLE SKILLS
# =========================================================

ROLE_SKILLS = {

    "AI/ML Engineer": [
        "python",
        "machine learning",
        "deep learning",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "sql"
    ],

    "Data Analyst": [
        "python",
        "sql",
        "excel",
        "pandas",
        "numpy",
        "power bi",
        "tableau",
        "data analysis"
    ],

    "Web Developer": [
        "html",
        "css",
        "javascript",
        "react",
        "node.js",
        "sql",
        "git",
        "rest api"
    ],

    "Software Developer": [
        "python",
        "java",
        "c++",
        "sql",
        "git",
        "github",
        "rest api"
    ],

    "Data Scientist": [
        "python",
        "machine learning",
        "pandas",
        "numpy",
        "scikit-learn",
        "sql",
        "data science"
    ],

    "Cybersecurity Analyst": [
        "cybersecurity",
        "network security",
        "linux",
        "ethical hacking",
        "penetration testing",
        "wireshark",
        "nmap",
        "siem"
    ]
}


# =========================================================
# RESUME SECTIONS
# =========================================================

SECTION_DEFINITIONS = {

    "Contact Information": [
        "email",
        "phone",
        "mobile",
        "linkedin",
        "github"
    ],

    "Professional Summary": [
        "summary",
        "objective",
        "profile",
        "professional summary"
    ],

    "Education": [
        "education",
        "academic",
        "qualification"
    ],

    "Experience": [
        "experience",
        "work experience",
        "employment",
        "internship"
    ],

    "Projects": [
        "project",
        "projects"
    ],

    "Skills": [
        "skills",
        "technical skills",
        "skill set"
    ],

    "Certifications": [
        "certification",
        "certifications",
        "certificate"
    ]
}


# =========================================================
# CHECK TESSERACT
# =========================================================

def check_tesseract():

    if not TESSERACT_PATH:
        return False, None

    try:

        version = pytesseract.get_tesseract_version()

        return True, str(version)

    except Exception:

        return False, None


# =========================================================
# CHECK POPPLER
# =========================================================

def check_poppler():

    # Windows
    if POPPLER_PATH:

        pdfinfo = os.path.join(
            POPPLER_PATH,
            "pdfinfo.exe"
        )

        pdftoppm = os.path.join(
            POPPLER_PATH,
            "pdftoppm.exe"
        )

        if (
            os.path.exists(pdfinfo)
            and
            os.path.exists(pdftoppm)
        ):

            return True

    # Linux / Streamlit Cloud
    if (
        shutil.which("pdfinfo")
        and
        shutil.which("pdftoppm")
    ):

        return True

    return False


# =========================================================
# OCR IMAGE
# =========================================================

def ocr_image(image):

    tesseract_ok, _ = check_tesseract()

    if not tesseract_ok:

        raise RuntimeError(
            "Tesseract OCR is not installed "
            "or could not be found."
        )

    try:

        return pytesseract.image_to_string(
            image
        )

    except Exception as error:

        raise RuntimeError(
            "Tesseract OCR failed while "
            "reading the image."
        ) from error


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf(file):

    # -----------------------------------------------------
    # FIRST TRY NORMAL PDF TEXT EXTRACTION
    # -----------------------------------------------------

    file.seek(0)

    reader = PdfReader(file)

    text_parts = []

    for page in reader.pages:

        try:

            page_text = page.extract_text()

            if page_text:

                text_parts.append(
                    page_text
                )

        except Exception:

            pass

    text = "\n".join(
        text_parts
    ).strip()

    # If enough text was extracted,
    # this is a normal text-based PDF.
    if len(text) >= 80:

        return text

    # -----------------------------------------------------
    # SCANNED PDF
    # -----------------------------------------------------

    st.info(
        "🔍 Scanned PDF detected. Running OCR..."
    )

    tesseract_ok, _ = check_tesseract()

    if not tesseract_ok:

        raise RuntimeError(
            "Tesseract OCR is not installed "
            "or could not be found."
        )

    if not check_poppler():

        raise RuntimeError(
            "Poppler was not found. "
            "Check your Poppler installation."
        )

    try:

        from pdf2image import convert_from_bytes

        file.seek(0)

        pdf_bytes = file.read()

        # Windows
        if POPPLER_PATH:

            images = convert_from_bytes(
                pdf_bytes,
                dpi=200,
                poppler_path=POPPLER_PATH
            )

        # Linux / Streamlit Cloud
        else:

            images = convert_from_bytes(
                pdf_bytes,
                dpi=200
            )

    except Exception as error:

        raise RuntimeError(
            "Could not convert the scanned PDF "
            "into images."
        ) from error

    results = []

    progress = st.progress(0)

    total_pages = len(images)

    for index, image in enumerate(images):

        extracted_text = ocr_image(
            image
        )

        results.append(
            extracted_text
        )

        progress.progress(
            (index + 1) / total_pages
        )

    progress.empty()

    return "\n".join(results)


# =========================================================
# DOCX TEXT EXTRACTION
# =========================================================

def extract_docx(file):

    document = Document(file)

    parts = []

    # Normal paragraphs
    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            parts.append(
                paragraph.text
            )

    # Tables
    for table in document.tables:

        for row in table.rows:

            row_text = []

            for cell in row.cells:

                if cell.text.strip():

                    row_text.append(
                        cell.text
                    )

            if row_text:

                parts.append(
                    " ".join(row_text)
                )

    # Images inside DOCX
    for relationship in document.part.rels.values():

        if "image" in relationship.reltype:

            try:

                image_data = (
                    relationship.target_part.blob
                )

                image = Image.open(
                    io.BytesIO(image_data)
                )

                extracted = ocr_image(
                    image
                )

                if extracted.strip():

                    parts.append(
                        extracted
                    )

            except Exception:

                pass

    return "\n".join(parts)


# =========================================================
# IMAGE TEXT EXTRACTION
# =========================================================

def extract_image(file):

    image = Image.open(file)

    if image.mode != "RGB":

        image = image.convert(
            "RGB"
        )

    return ocr_image(
        image
    )


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# FIND SKILLS
# =========================================================

def find_skills(text):

    text = text.lower()

    found = []

    for skill in SKILLS:

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(skill.lower())
            + r"(?![a-z0-9])"
        )

        if re.search(
            pattern,
            text
        ):

            found.append(
                skill
            )

    return found


# =========================================================
# SECTION DETECTION
# =========================================================

def section_present(
    text,
    names
):

    text = text.lower()

    for name in names:

        pattern = (
            r"\b"
            + re.escape(name)
            + r"\b"
        )

        if re.search(
            pattern,
            text
        ):

            return True

    return False


# =========================================================
# ANALYZE SECTIONS
# =========================================================

def analyze_sections(text):

    results = {}

    for section, keywords in SECTION_DEFINITIONS.items():

        results[section] = section_present(
            text,
            keywords
        )

    return results


# =========================================================
# RESUME ANALYSIS
# =========================================================

def analyze_resume(
    text,
    selected_role
):

    skills = find_skills(
        text
    )

    word_count = len(
        text.split()
    )

    sections = analyze_sections(
        text
    )

    strengths = []

    weaknesses = []

    recommendations = []

    # -----------------------------------------------------
    # WORD COUNT
    # -----------------------------------------------------

    if 250 <= word_count <= 900:

        strengths.append(
            "Resume has a reasonable amount of content."
        )

    elif word_count < 150:

        weaknesses.append(
            "Resume appears too short."
        )

        recommendations.append(
            "Add more details about your education, "
            "projects, skills and experience."
        )

    else:

        weaknesses.append(
            "Resume may contain too much text."
        )

        recommendations.append(
            "Make descriptions concise and focus "
            "on relevant achievements."
        )

    # -----------------------------------------------------
    # CONTACT
    # -----------------------------------------------------

    if sections[
        "Contact Information"
    ]:

        strengths.append(
            "Contact information detected."
        )

    else:

        weaknesses.append(
            "Contact information is incomplete "
            "or was not detected."
        )

        recommendations.append(
            "Add your email, phone number and "
            "professional LinkedIn/GitHub links."
        )

    # -----------------------------------------------------
    # EDUCATION
    # -----------------------------------------------------

    if sections[
        "Education"
    ]:

        strengths.append(
            "Education section detected."
        )

    else:

        weaknesses.append(
            "Education section was not detected."
        )

        recommendations.append(
            "Add a clearly labelled Education section."
        )

    # -----------------------------------------------------
    # PROJECTS
    # -----------------------------------------------------

    if sections[
        "Projects"
    ]:

        strengths.append(
            "Projects section detected."
        )

    else:

        weaknesses.append(
            "Projects section was not detected."
        )

        recommendations.append(
            "Add 2–3 relevant projects and "
            "mention technologies used."
        )

    # -----------------------------------------------------
    # EXPERIENCE
    # -----------------------------------------------------

    if sections[
        "Experience"
    ]:

        strengths.append(
            "Experience or internship section detected."
        )

    else:

        recommendations.append(
            "Add internships, practical experience "
            "or relevant work experience if available."
        )

    # -----------------------------------------------------
    # SKILLS
    # -----------------------------------------------------

    if sections[
        "Skills"
    ]:

        strengths.append(
            "Technical Skills section detected."
        )

    else:

        weaknesses.append(
            "Technical Skills section was not detected."
        )

        recommendations.append(
            "Create a dedicated Technical Skills section."
        )

    # -----------------------------------------------------
    # CERTIFICATIONS
    # -----------------------------------------------------

    if sections[
        "Certifications"
    ]:

        strengths.append(
            "Certifications section detected."
        )

    else:

        recommendations.append(
            "Add relevant certifications if you have them."
        )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    if sections[
        "Professional Summary"
    ]:

        strengths.append(
            "Professional summary/profile detected."
        )

    else:

        recommendations.append(
            "Add a short professional summary "
            "tailored to the target role."
        )

    # -----------------------------------------------------
    # SKILL COUNT
    # -----------------------------------------------------

    if len(skills) >= 8:

        strengths.append(
            f"Good technical skill coverage "
            f"({len(skills)} skills detected)."
        )

    elif len(skills) >= 4:

        strengths.append(
            f"{len(skills)} technical skills detected."
        )

    else:

        weaknesses.append(
            "Few technical skills were detected."
        )

        recommendations.append(
            "Add relevant technical skills that "
            "you genuinely know."
        )

    # -----------------------------------------------------
    # ROLE MATCH
    # -----------------------------------------------------

    target_skills = ROLE_SKILLS[
        selected_role
    ]

    matching_skills = []

    missing_skills = []

    for skill in target_skills:

        if skill in skills:

            matching_skills.append(
                skill
            )

        else:

            missing_skills.append(
                skill
            )

    role_percentage = round(
        (
            len(matching_skills)
            / len(target_skills)
        ) * 100,
        1
    )

    if role_percentage >= 75:

        strengths.append(
            f"Strong skill alignment with "
            f"{selected_role}."
        )

    elif role_percentage >= 50:

        strengths.append(
            f"Moderate skill alignment with "
            f"{selected_role}."
        )

        recommendations.append(
            f"Improve your {selected_role} skill "
            "coverage where applicable."
        )

    else:

        weaknesses.append(
            f"Low skill alignment with "
            f"{selected_role}."
        )

        recommendations.append(
            f"Tailor your resume toward "
            f"{selected_role}."
        )

    return {

        "skills": skills,

        "matching_skills":
            matching_skills,

        "missing_skills":
            missing_skills,

        "role_percentage":
            role_percentage,

        "word_count":
            word_count,

        "sections":
            sections,

        "strengths":
            strengths,

        "weaknesses":
            weaknesses,

        "recommendations":
            recommendations
    }


# =========================================================
# ATS SCORE
# =========================================================

def calculate_ats_score(
    text,
    selected_role
):

    analysis = analyze_resume(
        text,
        selected_role
    )

    score = 0

    word_count = analysis[
        "word_count"
    ]

    sections = analysis[
        "sections"
    ]

    # Length
    if 250 <= word_count <= 900:

        score += 15

    elif word_count >= 150:

        score += 10

    else:

        score += 5

    # Contact
    if sections[
        "Contact Information"
    ]:

        score += 10

    # Education
    if sections[
        "Education"
    ]:

        score += 10

    # Experience
    if sections[
        "Experience"
    ]:

        score += 15

    # Projects
    if sections[
        "Projects"
    ]:

        score += 15

    # Skills
    if sections[
        "Skills"
    ]:

        score += 15

    # Summary
    if sections[
        "Professional Summary"
    ]:

        score += 5

    # Role match
    score += round(
        analysis[
            "role_percentage"
        ] * 0.15
    )

    return min(
        score,
        100
    )


# =========================================================
# JOB DESCRIPTION MATCHING
# =========================================================

def analyze_job_description(
    resume_text,
    job_description
):

    resume_clean = clean_text(
        resume_text
    )

    jd_clean = clean_text(
        job_description
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        [
            resume_clean,
            jd_clean
        ]
    )

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    text_match = round(
        similarity * 100,
        1
    )

    resume_skills = set(
        find_skills(
            resume_clean
        )
    )

    jd_skills = set(
        find_skills(
            jd_clean
        )
    )

    matching_skills = sorted(
        resume_skills.intersection(
            jd_skills
        )
    )

    missing_skills = sorted(
        jd_skills.difference(
            resume_skills
        )
    )

    if jd_skills:

        skill_match = round(
            (
                len(matching_skills)
                /
                len(jd_skills)
            ) * 100,
            1
        )

    else:

        skill_match = 0

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    jd_vector = matrix[
        1
    ].toarray()[0]

    keyword_scores = list(
        zip(
            feature_names,
            jd_vector
        )
    )

    keyword_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    important_keywords = []

    for word, score in keyword_scores:

        if score <= 0:
            continue

        if len(word) <= 2:
            continue

        important_keywords.append(
            word
        )

        if len(
            important_keywords
        ) >= 15:

            break

    if jd_skills:

        final_match = round(
            (
                text_match * 0.60
                +
                skill_match * 0.40
            ),
            1
        )

    else:

        final_match = text_match

    return {

        "text_match":
            text_match,

        "skill_match":
            skill_match,

        "final_match":
            final_match,

        "job_skills":
            sorted(jd_skills),

        "matching_skills":
            matching_skills,

        "missing_skills":
            missing_skills,

        "important_keywords":
            important_keywords
    }


# =========================================================
# AI IMPROVEMENT SUGGESTIONS
# =========================================================

def generate_ai_suggestions(
    resume_text,
    selected_role,
    analysis,
    jd_analysis=None
):

    suggestions = []

    missing_role_skills = analysis[
        "missing_skills"
    ]

    sections = analysis[
        "sections"
    ]

    # Role skills
    for skill in missing_role_skills[:5]:

        suggestions.append(
            f"Consider adding **{skill.title()}** "
            f"to your resume if you genuinely "
            f"have experience with it."
        )

    # Job description
    if jd_analysis:

        missing_jd_skills = jd_analysis[
            "missing_skills"
        ]

        matching_jd_skills = jd_analysis[
            "matching_skills"
        ]

        if missing_jd_skills:

            suggestions.append(
                "The job description contains skills "
                "that were not detected in your resume. "
                "Add them only if they accurately "
                "represent your experience."
            )

            for skill in missing_jd_skills[:3]:

                suggestions.append(
                    f"If you have experience with "
                    f"**{skill.title()}**, mention it "
                    f"explicitly in your Skills or "
                    f"Projects section."
                )

        if matching_jd_skills:

            suggestions.append(
                "Your resume already contains relevant "
                "skills from this job description. "
                "Connect these skills to specific "
                "projects, internships or achievements."
            )

    # Projects
    if not sections["Projects"]:

        suggestions.append(
            "Add a Projects section containing "
            "2–3 relevant projects. Mention the "
            "technologies used and your contribution."
        )

    else:

        suggestions.append(
            "Strengthen project descriptions by "
            "explaining what you built, technologies "
            "used and measurable results."
        )

    # Experience
    if not sections["Experience"]:

        suggestions.append(
            "If you have internships, freelance work "
            "or practical experience, add an "
            "Experience section."
        )

    else:

        suggestions.append(
            "Use action-oriented statements in your "
            "experience section and include measurable "
            "results where possible."
        )

    # Summary
    if not sections["Professional Summary"]:

        suggestions.append(
            f"Add a 2–3 line professional summary "
            f"focused on **{selected_role}**."
        )

    else:

        suggestions.append(
            f"Tailor your professional summary toward "
            f"**{selected_role}** and highlight your "
            f"strongest relevant skills."
        )

    # Skills
    if len(analysis["skills"]) < 5:

        suggestions.append(
            "Your resume has relatively few detected "
            "technical skills. Add relevant technologies "
            "that you genuinely know."
        )

    else:

        suggestions.append(
            "Organize your technical skills into "
            "clear categories such as Programming, "
            "Tools, Databases, AI/ML or Security."
        )

    # Keywords
    if jd_analysis:

        keywords = jd_analysis[
            "important_keywords"
        ]

        if keywords:

            top_keywords = keywords[:7]

            keyword_string = ", ".join(
                keyword.title()
                for keyword in top_keywords
            )

            suggestions.append(
                "Important terms detected in the job "
                f"description include: **{keyword_string}**. "
                "Use matching terminology naturally "
                "where it truthfully describes your "
                "experience."
            )

    # General
    suggestions.append(
        "Use strong action verbs such as "
        "**developed, implemented, analyzed, "
        "designed, automated and optimized** "
        "when they accurately describe your work."
    )

    suggestions.append(
        f"Keep the resume concise and prioritize "
        f"information directly relevant to "
        f"**{selected_role}**."
    )

    # Remove duplicates
    unique_suggestions = []

    for suggestion in suggestions:

        if suggestion not in unique_suggestions:

            unique_suggestions.append(
                suggestion
            )

    return unique_suggestions


# =========================================================
# REPORT GENERATOR
# =========================================================

def create_report(
    filename,
    selected_role,
    ats_score,
    analysis,
    jd_analysis=None,
    ai_suggestions=None
):

    report = []

    report.append(
        "AI RESUME ANALYZER REPORT"
    )

    report.append(
        "=" * 55
    )

    report.append(
        f"Resume: {filename}"
    )

    report.append(
        f"Target Role: {selected_role}"
    )

    report.append("")

    report.append(
        f"ATS Score: {ats_score}/100"
    )

    report.append(
        f"Role Match: "
        f"{analysis['role_percentage']}%"
    )

    report.append(
        f"Word Count: "
        f"{analysis['word_count']}"
    )

    report.append("")

    report.append(
        "DETECTED SKILLS"
    )

    report.append(
        "-" * 30
    )

    for skill in analysis["skills"]:

        report.append(
            f"- {skill.title()}"
        )

    report.append("")

    report.append(
        "ROLE MATCHING SKILLS"
    )

    report.append(
        "-" * 30
    )

    for skill in analysis["matching_skills"]:

        report.append(
            f"- {skill.title()}"
        )

    report.append("")

    report.append(
        "MISSING ROLE SKILLS"
    )

    report.append(
        "-" * 30
    )

    for skill in analysis["missing_skills"]:

        report.append(
            f"- {skill.title()}"
        )

    report.append("")

    report.append(
        "STRENGTHS"
    )

    report.append(
        "-" * 30
    )

    for item in analysis["strengths"]:

        report.append(
            f"- {item}"
        )

    report.append("")

    report.append(
        "AREAS TO IMPROVE"
    )

    report.append(
        "-" * 30
    )

    for item in analysis["weaknesses"]:

        report.append(
            f"- {item}"
        )

    report.append("")

    report.append(
        "RECOMMENDATIONS"
    )

    report.append(
        "-" * 30
    )

    for item in analysis["recommendations"]:

        report.append(
            f"- {item}"
        )

    # Job description
    if jd_analysis:

        report.append("")

        report.append(
            "JOB DESCRIPTION MATCH"
        )

        report.append(
            "-" * 30
        )

        report.append(
            f"Overall Match: "
            f"{jd_analysis['final_match']}%"
        )

        report.append(
            f"Text Match: "
            f"{jd_analysis['text_match']}%"
        )

        report.append(
            f"Skill Match: "
            f"{jd_analysis['skill_match']}%"
        )

        report.append("")

        report.append(
            "Matching JD Skills:"
        )

        for skill in jd_analysis[
            "matching_skills"
        ]:

            report.append(
                f"- {skill.title()}"
            )

        report.append("")

        report.append(
            "Missing JD Skills:"
        )

        for skill in jd_analysis[
            "missing_skills"
        ]:

            report.append(
                f"- {skill.title()}"
            )

        report.append("")

        report.append(
            "Important JD Keywords:"
        )

        for keyword in jd_analysis[
            "important_keywords"
        ]:

            report.append(
                f"- {keyword.title()}"
            )

    # AI suggestions
    if ai_suggestions:

        report.append("")

        report.append(
            "AI RESUME IMPROVEMENT SUGGESTIONS"
        )

        report.append(
            "-" * 30
        )

        for suggestion in ai_suggestions:

            clean_suggestion = suggestion.replace(
                "**",
                ""
            )

            report.append(
                f"- {clean_suggestion}"
            )

    return "\n".join(
        report
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "⚙️ Resume Settings"
    )

    selected_role = st.selectbox(
        "Target Job Role",
        list(
            ROLE_SKILLS.keys()
        )
    )

    st.divider()

    st.write(
        "### 🧠 AI / ML Technologies"
    )

    st.write(
        "• Natural Language Processing"
    )

    st.write(
        "• TF-IDF"
    )

    st.write(
        "• Cosine Similarity"
    )

    st.write(
        "• OCR"
    )

    st.write(
        "• Keyword Extraction"
    )

    st.divider()

    st.info(
        "No training dataset is required."
    )


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload your Resume",
    type=[
        "pdf",
        "docx",
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file:

    st.write(
        f"📁 **File:** {uploaded_file.name}"
    )

    try:

        filename = uploaded_file.name.lower()

        # -------------------------------------------------
        # TEXT EXTRACTION
        # -------------------------------------------------

        if filename.endswith(".pdf"):

            raw_text = extract_pdf(
                uploaded_file
            )

        elif filename.endswith(".docx"):

            raw_text = extract_docx(
                uploaded_file
            )

        elif filename.endswith(
            (".jpg", ".jpeg", ".png")
        ):

            raw_text = extract_image(
                uploaded_file
            )

        else:

            raw_text = ""

        # -------------------------------------------------
        # CLEAN TEXT
        # -------------------------------------------------

        text = clean_text(
            raw_text
        )

        if len(text) < 20:

            st.error(
                "❌ No readable text was found "
                "in this resume."
            )

            st.stop()

        st.success(
            "✅ Resume text extracted successfully!"
        )

        # -------------------------------------------------
        # ANALYSIS
        # -------------------------------------------------

        analysis = analyze_resume(
            text,
            selected_role
        )

        ats_score = calculate_ats_score(
            text,
            selected_role
        )

        # =================================================
        # DASHBOARD
        # =================================================

        st.divider()

        st.subheader(
            "📊 Resume Score Dashboard"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "ATS Score",
            f"{ats_score}/100"
        )

        col2.metric(
            "Role Match",
            f"{analysis['role_percentage']}%"
        )

        col3.metric(
            "Skills",
            len(analysis["skills"])
        )

        col4.metric(
            "Words",
            analysis["word_count"]
        )

        st.progress(
            ats_score / 100
        )

        if ats_score >= 80:

            st.success(
                "🌟 Excellent ATS compatibility!"
            )

        elif ats_score >= 60:

            st.info(
                "👍 Good resume. Some improvements "
                "can increase the score."
            )

        else:

            st.warning(
                "⚠️ Your resume needs improvement."
            )

        # =================================================
        # SECTION ANALYSIS
        # =================================================

        st.divider()

        st.subheader(
            "📋 Resume Section Analysis"
        )

        section_cols = st.columns(4)

        section_items = list(
            analysis["sections"].items()
        )

        for index, (
            section,
            exists
        ) in enumerate(section_items):

            with section_cols[
                index % 4
            ]:

                if exists:

                    st.success(
                        f"✅ {section}"
                    )

                else:

                    st.error(
                        f"❌ {section}"
                    )

        # =================================================
        # SKILLS
        # =================================================

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "🧠 Detected Skills"
            )

            if analysis["skills"]:

                for skill in analysis["skills"]:

                    st.write(
                        f"✅ {skill.title()}"
                    )

            else:

                st.warning(
                    "No recognized skills detected."
                )

        with col2:

            st.subheader(
                f"🎯 {selected_role} Skills"
            )

            for skill in ROLE_SKILLS[
                selected_role
            ]:

                if skill in analysis[
                    "matching_skills"
                ]:

                    st.write(
                        f"🟢 {skill.title()}"
                    )

                else:

                    st.write(
                        f"🔴 {skill.title()}"
                    )

        # =================================================
        # STRENGTHS
        # =================================================

        st.divider()

        st.subheader(
            "💪 Resume Strengths"
        )

        for strength in analysis[
            "strengths"
        ]:

            st.success(
                "✓ " + strength
            )

        # =================================================
        # WEAKNESSES
        # =================================================

        st.subheader(
            "⚠️ Areas to Improve"
        )

        if analysis["weaknesses"]:

            for weakness in analysis[
                "weaknesses"
            ]:

                st.warning(
                    weakness
                )

        else:

            st.success(
                "No major weaknesses detected."
            )

        # =================================================
        # RECOMMENDATIONS
        # =================================================

        st.divider()

        st.subheader(
            "💡 Resume Recommendations"
        )

        recommendations = list(
            dict.fromkeys(
                analysis["recommendations"]
            )
        )

        for recommendation in recommendations:

            st.write(
                "🔹 " + recommendation
            )

        # =================================================
        # JOB DESCRIPTION MATCHING
        # =================================================

        st.divider()

        st.subheader(
            "🔎 Job Description Matching"
        )

        st.write(
            "Paste the job description of the "
            "position you want to apply for."
        )

        job_description = st.text_area(
            "📋 Job Description",
            height=250,
            placeholder=(
                "Example:\n\n"
                "We are looking for a Cybersecurity "
                "Analyst with experience in network "
                "security, Linux, SIEM, Wireshark, "
                "penetration testing and threat detection."
            )
        )

        jd_analysis = None

        if job_description.strip():

            if len(
                job_description.strip()
            ) < 30:

                st.warning(
                    "Please enter a more detailed "
                    "job description."
                )

            else:

                jd_analysis = (
                    analyze_job_description(
                        text,
                        job_description
                    )
                )

                # -----------------------------------------
                # MATCH SCORE
                # -----------------------------------------

                st.write(
                    "### 📊 Job Match Results"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Overall Match",
                    f"{jd_analysis['final_match']}%"
                )

                c2.metric(
                    "Text Match",
                    f"{jd_analysis['text_match']}%"
                )

                c3.metric(
                    "Skill Match",
                    f"{jd_analysis['skill_match']}%"
                )

                st.progress(
                    jd_analysis["final_match"] / 100
                )

                if jd_analysis[
                    "final_match"
                ] >= 75:

                    st.success(
                        "🟢 Excellent job match!"
                    )

                elif jd_analysis[
                    "final_match"
                ] >= 50:

                    st.info(
                        "🟡 Moderate job match."
                    )

                else:

                    st.warning(
                        "🔴 Low job match. "
                        "Consider tailoring your resume."
                    )

                # -----------------------------------------
                # MATCHING / MISSING
                # -----------------------------------------

                c1, c2 = st.columns(2)

                with c1:

                    st.subheader(
                        "✅ Matching Skills"
                    )

                    if jd_analysis[
                        "matching_skills"
                    ]:

                        for skill in jd_analysis[
                            "matching_skills"
                        ]:

                            st.write(
                                f"🟢 {skill.title()}"
                            )

                    else:

                        st.warning(
                            "No matching skills detected."
                        )

                with c2:

                    st.subheader(
                        "❌ Missing Skills"
                    )

                    if jd_analysis[
                        "missing_skills"
                    ]:

                        for skill in jd_analysis[
                            "missing_skills"
                        ]:

                            st.write(
                                f"🔴 {skill.title()}"
                            )

                    else:

                        st.success(
                            "No major missing skills detected."
                        )

                # -----------------------------------------
                # JOB SKILLS
                # -----------------------------------------

                st.divider()

                st.subheader(
                    "🎯 Skills Detected in Job Description"
                )

                if jd_analysis[
                    "job_skills"
                ]:

                    skill_text = " • ".join(
                        skill.title()
                        for skill in jd_analysis[
                            "job_skills"
                        ]
                    )

                    st.info(
                        skill_text
                    )

                else:

                    st.info(
                        "No predefined technical skills "
                        "were detected."
                    )

                # -----------------------------------------
                # KEYWORDS
                # -----------------------------------------

                st.divider()

                st.subheader(
                    "🔑 Important Job Keywords"
                )

                keywords = jd_analysis[
                    "important_keywords"
                ]

                if keywords:

                    keyword_cols = st.columns(3)

                    for index, keyword in enumerate(
                        keywords
                    ):

                        with keyword_cols[
                            index % 3
                        ]:

                            st.write(
                                f"🔹 {keyword.title()}"
                            )

                # -----------------------------------------
                # IMPROVEMENT
                # -----------------------------------------

                st.divider()

                st.subheader(
                    "💡 How to Improve Your Job Match"
                )

                if jd_analysis[
                    "missing_skills"
                ]:

                    st.write(
                        "The job description contains "
                        "skills that were not detected "
                        "in your resume:"
                    )

                    for skill in jd_analysis[
                        "missing_skills"
                    ]:

                        st.write(
                            f"• **{skill.title()}**"
                        )

                    st.info(
                        "Only add skills that you "
                        "actually possess."
                    )

                else:

                    st.success(
                        "Your resume contains the major "
                        "technical skills detected in "
                        "the job description."
                    )

                st.caption(
                    "ℹ️ Match percentage is based on "
                    "text and skill similarity. It does "
                    "not predict hiring probability."
                )

        # =================================================
        # AI IMPROVEMENT
        # =================================================

        st.divider()

        st.subheader(
            "🤖 AI Resume Improvement Generator"
        )

        st.write(
            "Get personalized suggestions based on "
            "your resume, target role and job description."
        )

        ai_suggestions = generate_ai_suggestions(
            text,
            selected_role,
            analysis,
            jd_analysis
        )

        for index, suggestion in enumerate(
            ai_suggestions,
            start=1
        ):

            st.markdown(
                f"**{index}.** {suggestion}"
            )

        st.info(
            "💡 Only include skills and experiences "
            "that are genuinely yours."
        )

        # =================================================
        # DOWNLOAD REPORT
        # =================================================

        st.divider()

        st.subheader(
            "📥 Download Analysis"
        )

        report = create_report(
            uploaded_file.name,
            selected_role,
            ats_score,
            analysis,
            jd_analysis,
            ai_suggestions
        )

        st.download_button(
            label="📄 Download Resume Analysis",
            data=report,
            file_name="resume_analysis.txt",
            mime="text/plain"
        )

        # =================================================
        # EXTRACTED TEXT
        # =================================================

        st.divider()

        with st.expander(
            "📃 View Extracted Resume Text"
        ):

            st.text_area(
                "Extracted Text",
                text,
                height=400
            )

    except Exception as error:

        st.error(
            "❌ Resume analysis failed."
        )

        st.code(
            str(error)
        )


# =========================================================
# HOME SCREEN
# =========================================================

else:

    st.markdown(
        """
        ## 🚀 How It Works

        **1️⃣ Upload Resume**

        Upload a PDF, DOCX, JPG, JPEG or PNG resume.

        **2️⃣ Text Extraction**

        The application extracts text from your resume.
        OCR is automatically used for scanned PDFs
        and images.

        **3️⃣ Resume Analysis**

        The system analyzes:

        - ATS compatibility
        - Resume sections
        - Technical skills
        - Target job-role skills
        - Strengths
        - Weaknesses
        - Recommendations

        **4️⃣ Job Description Matching**

        Paste a job description and the system
        calculates:

        - Overall match
        - Text similarity
        - Skill similarity
        - Matching skills
        - Missing skills
        - Important keywords

        **5️⃣ AI Resume Improvement**

        Receive personalized recommendations
        based on your resume and target job.

        **6️⃣ Download Report**

        Download your complete analysis report.

        ---

        ## 🧠 AI / ML Technologies

        ### Natural Language Processing

        Used to process and analyze resume
        and job-description text.

        ### TF-IDF

        Used to identify important terms
        in the text.

        ### Cosine Similarity

        Used to calculate similarity between
        resume and job-description text.

        ### OCR

        Used to extract text from scanned
        resumes and images.

        ### Keyword Extraction

        Used to identify important terms
        from job descriptions.

        ---

        ## 📌 Dataset

        This project does **not require a
        training dataset**.

        It uses NLP algorithms, OCR,
        predefined skill knowledge and
        text similarity techniques.
        """
    )
