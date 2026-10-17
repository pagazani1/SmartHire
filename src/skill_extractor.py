"""
src/skill_extractor.py – SmartHire rule-based skill extractor
--------------------------------------------------------------
Detects technical skills in free-form resume and job-description text using
case-insensitive keyword matching with word-boundary guards.

IMPORTANT: This module is intentionally separate from the ML classifier.
The skills it finds are NOT guaranteed to be the features the Logistic
Regression model used internally. They are a human-readable summary of
what appears in each text, useful for candidate feedback.

No new dependencies are required. Only Python stdlib (re) is used.
"""

import re
from typing import Set


# ---------------------------------------------------------------------------
# Skill vocabulary
# Each entry is the canonical display name for a skill.  The matching regex
# for each name is built automatically (see _build_pattern), but entries that
# contain special regex characters (e.g. C++, .NET) get custom patterns.
# ---------------------------------------------------------------------------

SKILL_VOCABULARY: list[str] = [
    # ── Programming Languages ───────────────────────────────────────────────
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#",
    "Go", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R",
    "MATLAB", "Perl", "Haskell", "Lua", "Dart", "Bash", "Shell",
    "PowerShell", "SQL", "PL/SQL", "T-SQL",

    # ── AI / ML / Data Science ──────────────────────────────────────────────
    "Machine Learning", "Deep Learning", "Natural Language Processing", "NLP",
    "Computer Vision", "Reinforcement Learning",
    "scikit-learn", "TensorFlow", "Keras", "PyTorch", "XGBoost", "LightGBM",
    "CatBoost", "Hugging Face", "spaCy", "NLTK", "OpenCV",
    "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly",
    "Jupyter", "ONNX", "MLflow", "Weights & Biases", "Optuna",
    "Feature Engineering", "Model Deployment", "Transfer Learning",
    "Transformers", "BERT", "GPT", "LLM",

    # ── Data Engineering & Databases ────────────────────────────────────────
    "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Cassandra",
    "Elasticsearch", "DynamoDB", "BigQuery", "Snowflake", "Redshift",
    "Apache Spark", "PySpark", "Hadoop", "Hive", "Kafka", "Airflow",
    "dbt", "Flink", "ETL", "Data Pipeline", "Data Warehousing",

    # ── Cloud Platforms ─────────────────────────────────────────────────────
    "AWS", "Azure", "GCP", "Google Cloud", "Heroku", "Vercel", "Netlify",
    "DigitalOcean", "Cloudflare",

    # ── DevOps & Infrastructure ─────────────────────────────────────────────
    "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitHub Actions",
    "GitLab CI", "CircleCI", "CI/CD", "Linux", "Unix", "Nginx", "Apache",
    "Prometheus", "Grafana", "Datadog",

    # ── Web Frameworks & APIs ───────────────────────────────────────────────
    "FastAPI", "Flask", "Django", "Node.js", "Express", "React", "Vue.js",
    "Angular", "Next.js", "Svelte", "Spring Boot", "ASP.NET", ".NET",
    "REST API", "GraphQL", "gRPC", "WebSocket",

    # ── Version Control & Collaboration ─────────────────────────────────────
    "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence",

    # ── Mobile ──────────────────────────────────────────────────────────────
    "Android", "iOS", "React Native", "Flutter",

    # ── General Engineering Practices ───────────────────────────────────────
    "Agile", "Scrum", "TDD", "Unit Testing", "API Development",
    "Microservices", "System Design", "OOP", "Functional Programming",
]


# ---------------------------------------------------------------------------
# Pattern builder
# ---------------------------------------------------------------------------

# Skills that contain special regex characters need a manually crafted pattern
# instead of the auto-generated word-boundary one.
#
# Key: canonical display name (must match an entry in SKILL_VOCABULARY exactly)
# Value: raw regex string (will be used inside re.compile with re.IGNORECASE)
_CUSTOM_PATTERNS: dict[str, str] = {
    "C++":          r"c\+\+",
    "C#":           r"c#",
    ".NET":         r"\.net",
    "ASP.NET":      r"asp\.net",
    "Node.js":      r"node\.js",
    "Vue.js":       r"vue\.js",
    "Next.js":      r"next\.js",
    "scikit-learn": r"scikit[\-\s]learn",
    "PL/SQL":       r"pl/sql",
    "T-SQL":        r"t[\-\s]sql",
    "CI/CD":        r"ci/cd",
    "REST API":     r"rest\s+api",
    "NLP":          r"\bNLP\b",          # keep as uppercase-only acronym guard
    "LLM":          r"\bLLMs?\b",
    "GPT":          r"\bGPT[\-\d]*\b",
    "BERT":         r"\bBERT\b",
    "AWS":          r"\bAWS\b",
    "GCP":          r"\bGCP\b",
    "ETL":          r"\bETL\b",
    "TDD":          r"\bTDD\b",
    "OOP":          r"\bOOP\b",
    "dbt":          r"\bdbt\b",
    "R":            r"\bR\b",            # single-letter: only match the lone letter
    "C":            r"\bC\b",
    "Go":           r"\bGo\b",
    "Weights & Biases": r"weights\s*&\s*biases|wandb",
}


def _build_pattern(skill_name: str) -> re.Pattern:
    """
    Build a compiled regex pattern for a skill name.

    - Skills with a custom pattern use that directly.
    - All other skills use \\b word-boundary anchors and escape special chars.
    """
    if skill_name in _CUSTOM_PATTERNS:
        raw = _CUSTOM_PATTERNS[skill_name]
    else:
        # Escape any special regex chars in the name, then add word boundaries
        escaped = re.escape(skill_name)
        raw = rf"\b{escaped}\b"
    return re.compile(raw, re.IGNORECASE)


# Pre-compile all patterns once at import time for efficiency
_COMPILED_PATTERNS: dict[str, re.Pattern] = {
    skill: _build_pattern(skill) for skill in SKILL_VOCABULARY
}


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> Set[str]:
    """
    Scan *text* and return the set of canonical skill names found.

    Parameters
    ----------
    text : str
        Raw text (resume or job description). May be empty.

    Returns
    -------
    set[str]
        Canonical skill names detected in the text (display form, e.g. "Python").
    """
    if not isinstance(text, str) or not text.strip():
        return set()

    found: Set[str] = set()
    for skill_name, pattern in _COMPILED_PATTERNS.items():
        if pattern.search(text):
            found.add(skill_name)
    return found


# ---------------------------------------------------------------------------
# High-level analyser (used by predict.py and tests)
# ---------------------------------------------------------------------------

def analyse_skills(resume_text: str, jd_text: str) -> dict:
    """
    Extract and compare skills between a resume and a job description.

    Returns
    -------
    dict with three keys:
        resume_skills  : set[str] – skills found in the resume
        matched_skills : set[str] – skills present in BOTH texts
        missing_skills : set[str] – skills in the JD but NOT in the resume
    """
    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(jd_text)

    matched_skills = resume_skills & jd_skills          # intersection
    missing_skills = jd_skills - resume_skills          # in JD but not resume

    return {
        "resume_skills":  resume_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper (used by predict.py display layer)
# ---------------------------------------------------------------------------

def display_skills(skill_result: dict) -> None:
    """
    Print a human-readable skill gap report from the dict returned by
    analyse_skills().

    Uses plain ASCII markers so output works on all terminals (including
    Windows cp1252) without a UnicodeEncodeError.
      [+]  skill found in both resume and JD (matched)
      [ ]  skill found in resume only
      [-]  skill required by JD but missing from resume
    """
    resume_skills  = sorted(skill_result["resume_skills"])
    matched_skills = sorted(skill_result["matched_skills"])
    missing_skills = sorted(skill_result["missing_skills"])

    print("\n" + "=" * 50)
    print("          SKILL ANALYSIS REPORT")
    print("=" * 50)

    print(f"\n  Skills found in resume  ({len(resume_skills)}):")
    if resume_skills:
        for s in resume_skills:
            marker = "[+]" if s in skill_result["matched_skills"] else "[ ]"
            print(f"    {marker} {s}")
    else:
        print("    (none detected)")

    print(f"\n  Matched skills (resume & JD)  ({len(matched_skills)}):")
    if matched_skills:
        for s in matched_skills:
            print(f"    [+] {s}")
    else:
        print("    (none)")

    print(f"\n  Skills required by JD but missing from resume  ({len(missing_skills)}):")
    if missing_skills:
        for s in missing_skills:
            print(f"    [-] {s}")
    else:
        print("    (none -- great coverage!)")

    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Self-test  (python src/skill_extractor.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Test 1: Normal case with several skills ──────────────────────────────
    resume1 = """
    Senior Python developer with 6 years of experience.
    Proficient in scikit-learn, TensorFlow, Pandas, NumPy, and Matplotlib.
    Worked with PostgreSQL, Redis, and Docker.
    Deployed ML models on AWS using FastAPI.
    Version control via Git and GitHub. Familiar with CI/CD pipelines.
    """

    jd1 = """
    We are hiring a Python ML Engineer.
    Required: scikit-learn, PyTorch, Pandas, NumPy.
    Experience with Docker, Kubernetes, and AWS is a plus.
    Familiarity with FastAPI or Flask preferred.
    Must know Git and have worked with SQL databases.
    Knowledge of NLP and Hugging Face Transformers is a bonus.
    """

    print("=" * 60)
    print("TEST 1 – Normal resume vs JD")
    print("=" * 60)
    result1 = analyse_skills(resume1, jd1)
    display_skills(result1)

    # ── Test 2: Special-character skill names ────────────────────────────────
    resume2 = "Experienced in C++, C#, .NET, Node.js and Vue.js development."
    jd2     = "Looking for C++ or C# developer. Node.js, .NET, and ASP.NET experience needed."

    print("=" * 60)
    print("TEST 2 – Special-character skill names (C++, C#, .NET, Node.js)")
    print("=" * 60)
    result2 = analyse_skills(resume2, jd2)
    display_skills(result2)

    # ── Test 3: Empty resume input ───────────────────────────────────────────
    print("=" * 60)
    print("TEST 3 – Empty resume text")
    print("=" * 60)
    result3 = analyse_skills("", jd1)
    display_skills(result3)

    # ── Test 4: No skill overlap ─────────────────────────────────────────────
    resume4 = "Expert in Ruby on Rails, PHP, and MySQL web development."
    jd4     = "Need a data scientist with PyTorch, Spark, and Kafka experience."

    print("=" * 60)
    print("TEST 4 – Zero overlap between resume and JD")
    print("=" * 60)
    result4 = analyse_skills(resume4, jd4)
    display_skills(result4)

    # ── Test 5: Case-insensitivity ───────────────────────────────────────────
    resume5 = "PYTHON, PANDAS, DOCKER expert."
    jd5     = "python, pandas, and docker knowledge required."

    print("=" * 60)
    print("TEST 5 – Case-insensitivity (uppercase resume vs lowercase JD)")
    print("=" * 60)
    result5 = analyse_skills(resume5, jd5)
    assert "Python" in result5["matched_skills"], "FAIL: Python should match"
    assert "Pandas" in result5["matched_skills"], "FAIL: Pandas should match"
    assert "Docker" in result5["matched_skills"], "FAIL: Docker should match"
    print("  [PASS] All case-insensitivity assertions passed.\n")

    print("All tests completed successfully.")
