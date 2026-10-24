import io
import os
import re
import streamlit as st
import pytesseract
from PIL import Image
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = None

possible_tesseract_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]

for path in possible_tesseract_paths:
    if os.path.exists(path):
        TESSERACT_PATH = path
        break

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            linear-gradient(
                rgba(4, 8, 20, 0.90),
                rgba(4, 10, 25, 0.96)
            ),
            url("https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=2400&q=85");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;

        color: #f8fafc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #070b18 0%,
                #0a1020 50%,
                #070c17 100%
            );

        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white;
    }

    .app-title {
        color: white;
        font-size: clamp(2.2rem, 4vw, 3.4rem);
        font-weight: 900;
        line-height: 1.15;
        letter-spacing: -1.5px;
        margin: 0;
        padding: 0;
    }

    .app-subtitle {
        color: #9fb0c9;
        font-size: 1rem;
        margin-top: 0.4rem;
        margin-bottom: 1.2rem;
    }

    .hero-box {
        background:
            linear-gradient(
                135deg,
                rgba(49, 46, 129, 0.94),
                rgba(12, 74, 110, 0.91)
            );

        border: 1px solid rgba(129, 140, 248, 0.38);
        border-radius: 20px;
        padding: 1.7rem;
        min-height: 185px;

        box-shadow:
            0 18px 45px rgba(0, 0, 0, 0.28),
            inset 0 1px 0 rgba(255,255,255,0.08);
    }

    .hero-label {
        color: #c7d2fe;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 1.5px;
        margin-bottom: 0.7rem;
    }

    .hero-heading {
        color: white;
        font-size: 1.8rem;
        font-weight: 900;
        margin-bottom: 0.6rem;
    }

    .hero-text {
        color: #dbeafe;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .upload-card {
        background: rgba(9, 17, 33, 0.86);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 18px;
        padding: 1.2rem;
        min-height: 185px;
    }

    .upload-title {
        color: white;
        font-size: 1.3rem;
        font-weight: 850;
    }

    .upload-subtitle {
        color: #94a3b8;
        margin-top: 0.25rem;
        margin-bottom: 0.8rem;
    }

    .section-title {
        color: white;
        font-size: 1.45rem;
        font-weight: 900;
        margin-top: 1.3rem;
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(8, 16, 31, 0.78);
        border-color: rgba(148, 163, 184, 0.15) !important;
        border-radius: 16px !important;
    }

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(11, 20, 38, 0.96),
                rgba(20, 31, 55, 0.90)
            );

        border: 1px solid rgba(129, 140, 248, 0.22);
        border-radius: 15px;
        padding: 0.8rem;
        min-height: 90px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.7rem !important;
    }

    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.45rem !important;
        font-weight: 900 !important;
    }

    .compact-skill-card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(99, 102, 241, 0.28);
        border-radius: 10px;
        padding: 0.55rem 0.7rem;
        margin-bottom: 0.5rem;
        min-height: 58px;
        box-sizing: border-box;
        transition: all 0.2s ease;
    }

    .compact-skill-card:hover {
        border-color: rgba(129, 140, 248, 0.7);
        transform: translateY(-1px);
    }

    .compact-skill-title {
        color: #f8fafc;
        font-size: 1rem;
        line-height: 1.2;
        font-weight: 800;
        word-break: break-word;
    }

    .compact-skill-caption {
        color: #94a3b8;
        font-size: 0.72rem;
        margin-top: 0.25rem;
        line-height: 1.2;
    }

    .compact-skill-match {
        border-color: rgba(34, 197, 94, 0.38);
    }

    .compact-skill-missing {
        border-color: rgba(239, 68, 68, 0.38);
    }

    [data-testid="stTabs"] {
        margin-top: 1.3rem;
    }

    [data-testid="stTabs"] button {
        color: #cbd5e1 !important;
        font-weight: 800;
        font-size: 0.92rem;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: white !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background: #6366f1 !important;
    }

    [data-baseweb="select"] > div {
        background-color: #0e1728 !important;
        border-color: rgba(148, 163, 184, 0.20) !important;
    }

    textarea,
    input {
        background-color: #0e1728 !important;
        color: white !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 11px;

        background:
            linear-gradient(
                135deg,
                #4f46e5,
                #2563eb
            );

        color: white;
        border: 1px solid rgba(129, 140, 248, 0.35);

        font-weight: 800;
    }

    [data-testid="stFileUploader"] {
        background: rgba(8, 16, 31, 0.82);
        border: 1px dashed rgba(129, 140, 248, 0.48);
        border-radius: 15px;
        padding: 0.5rem;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    @media (max-width: 800px) {

        .block-container {
            padding-top: 1rem;
        }

        .app-title {
            font-size: 2.2rem;
        }

        .hero-box {
            min-height: auto;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SKILLS DATABASE
# ============================================================

SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "javascript",
    "typescript",
    "sql",
    "html",
    "css",
    "react",
    "node.js",
    "django",
    "flask",
    "spring",
    "rest api",
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
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "linux",
    "excel",
    "power bi",
    "tableau",
    "mongodb",
    "mysql",
    "postgresql",
    "firebase",
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
    "communication",
    "leadership",
    "problem solving",
    "teamwork",
    "project management"
]

# ============================================================
# ROLE SKILLS
# ============================================================

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

# ============================================================
# RESUME SECTION DEFINITIONS
# ============================================================

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

# ============================================================
# TESSERACT CHECK
# ============================================================

def check_tesseract():

    if not TESSERACT_PATH:
        return False, None

    try:

        version = pytesseract.get_tesseract_version()

        return True, str(version)

    except Exception:

        return False, None


# ============================================================
# OCR IMAGE
# ============================================================

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


# ============================================================
# PDF EXTRACTION
# NO POPPLER REQUIRED
# ============================================================

def extract_pdf(file):

    file.seek(0)

    # --------------------------------------------------------
    # FIRST: TRY NORMAL PDF TEXT EXTRACTION
    # --------------------------------------------------------

    try:

        reader = PdfReader(file)

        text_parts = []

        for page in reader.pages:

            try:

                page_text = page.extract_text()

                if page_text:
                    text_parts.append(page_text)

            except Exception:

                pass

        text = "\n".join(
            text_parts
        ).strip()

    except Exception:

        text = ""

    # If normal text extraction worked,
    # return it without OCR.

    if len(text) >= 80:

        return text

    # --------------------------------------------------------
    # SECOND: OCR SCANNED PDF
    # --------------------------------------------------------

    st.info(
        "🔍 Scanned PDF detected. Running OCR..."
    )

    tesseract_ok, _ = check_tesseract()

    if not tesseract_ok:

        raise RuntimeError(
            "Tesseract OCR is not installed "
            "or could not be found."
        )

    try:

        import fitz

    except ImportError:

        raise RuntimeError(
            "PyMuPDF is not installed. "
            "Run: pip install PyMuPDF"
        )

    try:

        file.seek(0)

        pdf_bytes = file.read()

        pdf_document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        if len(pdf_document) == 0:

            raise RuntimeError(
                "The PDF contains no pages."
            )

        results = []

        progress = st.progress(0)

        total_pages = len(
            pdf_document
        )

        for index, page in enumerate(
            pdf_document
        ):

            # Render PDF page into an image.
            # This completely removes the need
            # for Poppler.

            pix = page.get_pixmap(
                matrix=fitz.Matrix(
                    2,
                    2
                ),
                alpha=False
            )

            image = Image.frombytes(
                "RGB",
                [
                    pix.width,
                    pix.height
                ],
                pix.samples
            )

            extracted_text = ocr_image(
                image
            )

            if extracted_text.strip():

                results.append(
                    extracted_text
                )

            progress.progress(
                (index + 1) / total_pages
            )

        progress.empty()

        pdf_document.close()

        final_text = "\n".join(
            results
        ).strip()

        if not final_text:

            raise RuntimeError(
                "OCR could not extract readable "
                "text from this PDF."
            )

        return final_text

    except Exception as error:

        raise RuntimeError(
            f"Could not process the scanned PDF: {error}"
        ) from error


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(file):

    document = Document(file)

    parts = []

    # Paragraphs
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
                    io.BytesIO(
                        image_data
                    )
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


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def extract_image(file):

    image = Image.open(file)

    if image.mode != "RGB":

        image = image.convert(
            "RGB"
        )

    return ocr_image(
        image
    )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# FIND SKILLS
# ============================================================

def find_skills(text):

    text = text.lower()

    found = []

    for skill in SKILLS:

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(
                skill.lower()
            )
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


# ============================================================
# SECTION DETECTION
# ============================================================

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


def analyze_sections(text):

    results = {}

    for section, keywords in SECTION_DEFINITIONS.items():

        results[section] = section_present(
            text,
            keywords
        )

    return results


# ============================================================
# RESUME ANALYSIS
# ============================================================

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

    # Word count
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

    # Contact
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

    # Education
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

    # Projects
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

    # Experience
    if sections[
        "Experience"
    ]:

        strengths.append(
            "Experience or internship section detected."
        )

    else:

        recommendations.append(
            "Add internships, freelance work "
            "or relevant practical experience if available."
        )

    # Skills
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

    # Certifications
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

    # Summary
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

    # Skill count
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

    # Role matching
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
            /
            len(target_skills)
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
            f"Improve your {selected_role} "
            "skill coverage where applicable."
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
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "role_percentage": role_percentage,
        "word_count": word_count,
        "sections": sections,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations
    }


# ============================================================
# ATS SCORE
# ============================================================

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

    if 250 <= word_count <= 900:

        score += 15

    elif word_count >= 150:

        score += 10

    else:

        score += 5

    if sections[
        "Contact Information"
    ]:
        score += 10

    if sections[
        "Education"
    ]:
        score += 10

    if sections[
        "Experience"
    ]:
        score += 15

    if sections[
        "Projects"
    ]:
        score += 15

    if sections[
        "Skills"
    ]:
        score += 15

    if sections[
        "Professional Summary"
    ]:
        score += 5

    score += round(
        analysis[
            "role_percentage"
        ] * 0.15
    )

    return min(
        score,
        100
    )


# ============================================================
# JOB DESCRIPTION ANALYSIS
# ============================================================

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
        "text_match": text_match,
        "skill_match": skill_match,
        "final_match": final_match,
        "job_skills": sorted(
            jd_skills
        ),
        "matching_skills": matching_skills,
        "missing_skills": missing_skills
    }


# ============================================================
# IMPROVEMENT SUGGESTIONS
# ============================================================

def generate_ai_suggestions(
    resume_text,
    selected_role,
    analysis,
    jd_analysis=None
):

    suggestions = []

    # Missing role skills
    for skill in analysis[
        "missing_skills"
    ][:5]:

        suggestions.append(
            f"Consider adding {skill.title()} "
            f"to your resume if you genuinely "
            f"have experience with it."
        )

    # Job description matching
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
                    f"{skill.title()}, mention it "
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
    if not analysis[
        "sections"
    ]["Projects"]:

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
    if not analysis[
        "sections"
    ]["Experience"]:

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
    if not analysis[
        "sections"
    ]["Professional Summary"]:

        suggestions.append(
            f"Add a 2–3 line professional summary "
            f"focused on {selected_role}."
        )

    else:

        suggestions.append(
            f"Tailor your professional summary toward "
            f"{selected_role} and highlight your "
            f"strongest relevant skills."
        )

    # Skills
    if len(
        analysis["skills"]
    ) < 5:

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

    # Action verbs
    suggestions.append(
        "Use strong action verbs such as "
        "developed, implemented, analyzed, "
        "designed, automated and optimized "
        "when they accurately describe your work."
    )

    # Conciseness
    suggestions.append(
        f"Keep the resume concise and prioritize "
        f"information directly relevant to "
        f"{selected_role}."
    )

    # Remove duplicates
    unique_suggestions = []

    for suggestion in suggestions:

        if suggestion not in unique_suggestions:

            unique_suggestions.append(
                suggestion
            )

    return unique_suggestions


# ============================================================
# CREATE TEXT REPORT
# ============================================================

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

    # Detected skills
    report.append(
        "DETECTED SKILLS"
    )

    report.append(
        "-" * 30
    )

    for skill in analysis[
        "skills"
    ]:

        report.append(
            f"- {skill.title()}"
        )

    report.append("")

    # Role matching skills
    report.append(
        "ROLE MATCHING SKILLS"
    )

    report.append(
        "-" * 30
    )

    for skill in analysis[
        "matching_skills"
    ]:

        report.append(
            f"- {skill.title()}"
        )

    report.append("")

    # Missing role skills
    report.append(
        "MISSING ROLE SKILLS"
    )

    report.append(
        "-" * 30
    )

    for skill in analysis[
        "missing_skills"
    ]:

        report.append(
            f"- {skill.title()}"
        )

    report.append("")

    # Strengths
    report.append(
        "STRENGTHS"
    )

    report.append(
        "-" * 30
    )

    for item in analysis[
        "strengths"
    ]:

        report.append(
            f"- {item}"
        )

    report.append("")

    # Areas to improve
    report.append(
        "AREAS TO IMPROVE"
    )

    report.append(
        "-" * 30
    )

    for item in analysis[
        "weaknesses"
    ]:

        report.append(
            f"- {item}"
        )

    if ai_suggestions:

        for suggestion in ai_suggestions:

            clean_suggestion = suggestion.replace(
                "**",
                ""
            )

            if clean_suggestion not in analysis[
                "weaknesses"
            ]:

                report.append(
                    f"- {clean_suggestion}"
                )

    report.append("")

    # Recommendations
    report.append(
        "RECOMMENDATIONS"
    )

    report.append(
        "-" * 30
    )

    for item in analysis[
        "recommendations"
    ]:

        report.append(
            f"- {item}"
        )

    # Job description section
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

    return "\n".join(
        report
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Resume Settings"
    )

    st.caption(
        "Personalize your analysis"
    )

    selected_role = st.selectbox(
        "TARGET JOB ROLE",
        list(
            ROLE_SKILLS.keys()
        )
    )

    st.markdown(
        "### 🎯 Analysis Profile"
    )

    st.info(
        f"Your resume will be evaluated "
        f"for **{selected_role}**."
    )

    st.markdown(
        "### 📁 Supported Files"
    )

    st.write(
        "📄 PDF"
    )

    st.write(
        "📝 DOCX"
    )

    st.write(
        "🖼️ JPG / JPEG / PNG"
    )

    st.divider()

    st.caption(
        "💡 For best results, use a text-based "
        "PDF or DOCX."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="app-title">'
    '📄 AI Resume Analyzer'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Analyze your resume, identify skills, match '
    'job requirements and discover areas to improve.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

with st.container(
    border=True
):

    st.markdown(
        "### 📤 Upload Your Resume"
    )

    st.caption(
        "PDF, DOCX, JPG, JPEG or PNG"
    )

    uploaded_file = st.file_uploader(
        "Choose your resume",
        type=[
            "pdf",
            "docx",
            "jpg",
            "jpeg",
            "png"
        ],
        help=(
            "Supported formats: PDF, DOCX, "
            "JPG, JPEG and PNG"
        )
    )


# ============================================================
# HERO INFORMATION
# ============================================================

with st.container(
    border=True
):

    st.markdown(
        "### ✨ Analyze. Match. Improve."
    )

    st.markdown(
        """
        Turn your resume into a stronger application
        with instant ATS insights, role-based skill
        analysis, job-description matching and
        practical improvement suggestions.
        """
    )

    st.caption(
        "AI-powered resume intelligence"
    )


# ============================================================
# LANDING PAGE
# ============================================================

if not uploaded_file:

    st.markdown(
        '<div class="section-title">'
        '🚀 Get Started in 3 Simple Steps'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Upload your resume and let the analyzer '
        'handle the rest.'
        '</div>',
        unsafe_allow_html=True
    )

    step1, step2, step3 = st.columns(
        3,
        gap="medium"
    )

    with step1:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 01"
            )

            st.markdown(
                "#### 📤 Upload Resume"
            )

            st.caption(
                "Upload your PDF, DOCX, JPG, "
                "JPEG or PNG resume."
            )

    with step2:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 02"
            )

            st.markdown(
                "#### 🔍 Analyze"
            )

            st.caption(
                "Extract text, detect skills, "
                "evaluate ATS compatibility "
                "and analyze your target role."
            )

    with step3:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 03"
            )

            st.markdown(
                "#### 📊 Review"
            )

            st.caption(
                "Check your resume analysis, "
                "job matching and downloadable report."
            )

    st.info(
        "💡 Select a target role from the sidebar, "
        "then upload your resume to begin."
    )

    st.stop()


# ============================================================
# PROCESS RESUME
# ============================================================

try:

    st.success(
        f"📁 {uploaded_file.name} uploaded successfully."
    )

    filename = uploaded_file.name.lower()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if filename.endswith(".pdf"):

        raw_text = extract_pdf(
            uploaded_file
        )

    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    elif filename.endswith(".docx"):

        raw_text = extract_docx(
            uploaded_file
        )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    elif filename.endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    ):

        raw_text = extract_image(
            uploaded_file
        )

    else:

        raw_text = ""

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    analysis = analyze_resume(
        text,
        selected_role
    )

    ats_score = calculate_ats_score(
        text,
        selected_role
    )

    # ========================================================
    # SCORE DASHBOARD
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Resume Score Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(
        4,
        gap="small"
    )

    with c1:

        st.metric(
            "ATS Score",
            f"{ats_score}/100"
        )

    with c2:

        st.metric(
            "Role Match",
            f"{analysis['role_percentage']}%"
        )

    with c3:

        st.metric(
            "Skills Detected",
            len(
                analysis["skills"]
            )
        )

    with c4:

        st.metric(
            "Word Count",
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


    # ========================================================
    # TABS
    # ========================================================

    overview_tab, job_tab, report_tab = st.tabs(
        [
            "📋 Resume Analysis",
            "🎯 Job Description Matching",
            "📥 Report"
        ]
    )


    # ========================================================
    # RESUME ANALYSIS TAB
    # ========================================================

    with overview_tab:

        # ----------------------------------------------------
        # SECTION ANALYSIS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '📋 Resume Section Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Check whether important resume sections '
            'were detected.'
            '</div>',
            unsafe_allow_html=True
        )

        section_items = list(
            analysis[
                "sections"
            ].items()
        )

        section_cols = st.columns(
            4,
            gap="small"
        )

        for index, (
            section,
            exists
        ) in enumerate(
            section_items
        ):

            with section_cols[
                index % 4
            ]:

                with st.container(
                    border=True
                ):

                    if exists:

                        st.success(
                            f"✅ {section}"
                        )

                    else:

                        st.error(
                            f"❌ {section}"
                        )


        # ----------------------------------------------------
        # SKILLS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🧠 Skills Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Every detected skill is displayed as an individual card.'
            '</div>',
            unsafe_allow_html=True
        )

        skill_col1, skill_col2 = st.columns(
            2,
            gap="medium"
        )

        # ----------------------------------------------------
        # DETECTED SKILLS
        # ----------------------------------------------------

        with skill_col1:

            st.subheader(
                "🔹 Detected Skills"
            )

            if analysis[
                "skills"
            ]:

                detected_skill_cols = st.columns(
                    2,
                    gap="small"
                )

                for index, skill in enumerate(
                    analysis["skills"]
                ):

                    with detected_skill_cols[
                        index % 2
                    ]:

                        st.markdown(
                            f"""
                            <div class="compact-skill-card">
                                <div class="compact-skill-title">
                                    💠 {skill.title()}
                                </div>
                                <div class="compact-skill-caption">
                                    Detected in resume
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            else:

                st.warning(
                    "No recognized skills detected."
                )


        # ----------------------------------------------------
        # TARGET ROLE SKILLS
        # ----------------------------------------------------

        with skill_col2:

            st.subheader(
                f"🎯 {selected_role} Skills"
            )

            role_skill_cols = st.columns(
                2,
                gap="small"
            )

            for index, skill in enumerate(
                ROLE_SKILLS[
                    selected_role
                ]
            ):

                with role_skill_cols[
                    index % 2
                ]:

                    if skill in analysis[
                        "matching_skills"
                    ]:

                        st.markdown(
                            f"""
                            <div class="compact-skill-card compact-skill-match">
                                <div class="compact-skill-title">
                                    🟢 {skill.title()}
                                </div>
                                <div class="compact-skill-caption">
                                    Skill matched
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            f"""
                            <div class="compact-skill-card compact-skill-missing">
                                <div class="compact-skill-title">
                                    🔴 {skill.title()}
                                </div>
                                <div class="compact-skill-caption">
                                    Skill not detected
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


        # ----------------------------------------------------
        # FEEDBACK
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '💎 Resume Feedback'
            '</div>',
            unsafe_allow_html=True
        )

        strengths_col, improve_col = st.columns(
            2,
            gap="large"
        )

        # ----------------------------------------------------
        # STRENGTHS
        # ----------------------------------------------------

        with strengths_col:

            with st.container(
                border=True
            ):

                st.subheader(
                    "💪 Resume Strengths"
                )

                if analysis[
                    "strengths"
                ]:

                    for strength in analysis[
                        "strengths"
                    ]:

                        st.success(
                            "✓ " + strength
                        )

                else:

                    st.info(
                        "No major strengths detected."
                    )


        # ----------------------------------------------------
        # AREAS TO IMPROVE
        # ----------------------------------------------------

        with improve_col:

            with st.container(
                border=True
            ):

                st.subheader(
                    "⚡ Areas to Improve"
                )

                ai_suggestions = generate_ai_suggestions(
                    text,
                    selected_role,
                    analysis
                )

                areas_to_improve = list(
                    dict.fromkeys(
                        analysis[
                            "weaknesses"
                        ]
                        +
                        ai_suggestions
                    )
                )

                if areas_to_improve:

                    for index, improvement in enumerate(
                        areas_to_improve,
                        start=1
                    ):

                        clean_improvement = (
                            improvement.replace(
                                "**",
                                ""
                            )
                        )

                        st.warning(
                            f"**{index}.** "
                            f"{clean_improvement}"
                        )

                else:

                    st.success(
                        "🎉 No major areas to improve detected."
                    )


        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '💡 Resume Recommendations'
            '</div>',
            unsafe_allow_html=True
        )

        recommendations = list(
            dict.fromkeys(
                analysis[
                    "recommendations"
                ]
            )
        )

        if recommendations:

            rec_cols = st.columns(
                2,
                gap="small"
            )

            for index, recommendation in enumerate(
                recommendations
            ):

                with rec_cols[
                    index % 2
                ]:

                    with st.container(
                        border=True
                    ):

                        st.info(
                            "🔹 " + recommendation
                        )

        else:

            st.success(
                "No additional recommendations."
            )


    # ========================================================
    # JOB DESCRIPTION TAB
    # ========================================================

    with job_tab:

        st.markdown(
            '<div class="section-title">'
            '🎯 Job Description Matching'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Paste the job description to compare '
            'your resume with the actual requirements.'
            '</div>',
            unsafe_allow_html=True
        )

        job_description = st.text_area(
            "📋 Job Description",
            height=180,
            placeholder=(
                "Paste the complete job description here...\n\n"
                "Example:\n"
                "We are looking for a Cybersecurity Analyst "
                "with experience in Python, Linux, "
                "network security, SIEM and Wireshark."
            )
        )

        if job_description.strip():

            if len(
                job_description.strip()
            ) < 30:

                st.warning(
                    "Please enter a more detailed "
                    "job description."
                )

            else:

                jd_analysis = analyze_job_description(
                    text,
                    job_description
                )

                # ------------------------------------------------
                # JOB MATCH SCORE
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">'
                    '📊 Job Match Results'
                    '</div>',
                    unsafe_allow_html=True
                )

                j1, j2, j3 = st.columns(
                    3,
                    gap="small"
                )

                with j1:

                    st.metric(
                        "Overall Match",
                        f"{jd_analysis['final_match']}%"
                    )

                with j2:

                    st.metric(
                        "Text Match",
                        f"{jd_analysis['text_match']}%"
                    )

                with j3:

                    st.metric(
                        "Skill Match",
                        f"{jd_analysis['skill_match']}%"
                    )

                st.progress(
                    jd_analysis[
                        "final_match"
                    ] / 100
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


                # ------------------------------------------------
                # MATCHING / MISSING
                # ------------------------------------------------

                match_col, missing_col = st.columns(
                    2,
                    gap="large"
                )

                with match_col:

                    with st.container(
                        border=True
                    ):

                        st.subheader(
                            "✅ Matching Skills"
                        )

                        if jd_analysis[
                            "matching_skills"
                        ]:

                            for skill in jd_analysis[
                                "matching_skills"
                            ]:

                                st.success(
                                    f"🟢 {skill.title()}"
                                )

                        else:

                            st.warning(
                                "No matching skills detected."
                            )


                with missing_col:

                    with st.container(
                        border=True
                    ):

                        st.subheader(
                            "❌ Missing Skills"
                        )

                        if jd_analysis[
                            "missing_skills"
                        ]:

                            for skill in jd_analysis[
                                "missing_skills"
                            ]:

                                st.error(
                                    f"🔴 {skill.title()}"
                                )

                        else:

                            st.success(
                                "No major missing skills detected."
                            )


                # ------------------------------------------------
                # JOB SKILLS
                # ------------------------------------------------

                st.subheader(
                    "🎯 Skills Detected in Job Description"
                )

                if jd_analysis[
                    "job_skills"
                ]:

                    job_skill_cols = st.columns(
                        3,
                        gap="small"
                    )

                    for index, skill in enumerate(
                        jd_analysis[
                            "job_skills"
                        ]
                    ):

                        with job_skill_cols[
                            index % 3
                        ]:

                            with st.container(
                                border=True
                            ):

                                st.markdown(
                                    f"### 💠 {skill.title()}"
                                )

                else:

                    st.info(
                        "No predefined technical skills "
                        "were detected."
                    )


                # ------------------------------------------------
                # IMPORTANT JOB KEYWORDS BLOCK REMOVED
                # ------------------------------------------------
                #
                # There is intentionally NO:
                #
                # 🔑 Important Job Keywords
                #
                # block here.
                #
                # ------------------------------------------------


                # ------------------------------------------------
                # HOW TO IMPROVE JOB MATCH
                # ------------------------------------------------

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

        else:

            st.info(
                "👆 Paste a job description above "
                "to see the matching analysis."
            )


    # ========================================================
    # REPORT TAB
    # ========================================================

    with report_tab:

        st.markdown(
            '<div class="section-title">'
            '📥 Resume Analysis Report'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Download the complete analysis report.'
            '</div>',
            unsafe_allow_html=True
        )

        ai_suggestions = generate_ai_suggestions(
            text,
            selected_role,
            analysis
        )

        report = create_report(
            uploaded_file.name,
            selected_role,
            ats_score,
            analysis,
            None,
            ai_suggestions
        )

        st.download_button(
            label="📥 Download Resume Analysis Report",
            data=report,
            file_name="resume_analysis.txt",
            mime="text/plain",
            use_container_width=True
        )

        # ----------------------------------------------------
        # REPORT PREVIEW
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '📄 Report Preview'
            '</div>',
            unsafe_allow_html=True
        )

        with st.container(
            border=True
        ):

            st.text(
                report
            )

        # ----------------------------------------------------
        # EXTRACTED TEXT
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '📃 Extracted Resume Text'
            '</div>',
            unsafe_allow_html=True
        )

        with st.expander(
            "View Extracted Resume Text"
        ):

            st.text_area(
                "Extracted Text",
                text,
                height=350
            )


# ============================================================
# ERROR HANDLING
# ============================================================

except Exception as error:

    st.error(
        "❌ Resume analysis failed."
    )

    st.exception(
        error
    )