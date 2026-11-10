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
from typing import Any, Set


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
    "Transformers", "BERT", "GPT", "LLM", "Text Classification",

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
    "REST API":     r"rest\s+apis?",
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

_ALIAS_PATTERNS: dict[str, re.Pattern] = {
    "JavaScript": re.compile(
        r"\bjavascript\b|(?<![\w.])js(?![\w])", re.IGNORECASE
    ),
    "scikit-learn": re.compile(
        r"\b(?:scikit[\-\s]?learn|sklearn)\b", re.IGNORECASE
    ),
    "PostgreSQL": re.compile(r"\bpostgres(?:ql)?\b", re.IGNORECASE),
}

_NEGATED_BEFORE = re.compile(
    r"(?:\bno(?:\s+experience)?|\bnot|\bnever|\bwithout|\black(?:s|ing)?"
    r"|\bdoes(?:n't| not) have|\bdo not have|\bdon't know|\bdo not know"
    r"|\bno knowledge of|\bnot familiar with"
    r"|\bnot proficient in|\bnot experienced in)"
    r"(?:\s+[\w'-]+){0,3}\s*$",
    re.IGNORECASE,
)
_NEGATED_AFTER = re.compile(
    r"^\s*(?:(?:is|are|was|were|has|have|had)\s+)?"
    r"(?:not|never|isn't|aren't|wasn't|weren't|hasn't|haven't)\s+"
    r"(?:required|used|worked with|experienced|proficient|"
    r"familiar|available|included|supported|in|on|among)\b",
    re.IGNORECASE,
)
_ASPIRATIONAL_BEFORE = re.compile(
    r"(?:\b(?:interested in|eager to|keen to|hope to|hoping to|want to|would like to|"
    r"looking to|planning to|aspiring to|willing to)\s+"
    r"(?:learn(?:ing)?|study|pick up|explore|develop(?:\s+my)?\s+skills in)"
    r"|\b(?:(?:am|is|are|have been)\s+)?(?:currently\s+|actively\s+|still\s+)?"
    r"learning(?:\s+to use)?|\blearning to use)\s*$",
    re.IGNORECASE,
)
_REQUIRED_SECTION = re.compile(
    r"^\s*(?:required(?:\s+(?:skills?|qualifications?))?|requirements?|"
    r"minimum qualifications?|basic qualifications?|must[- ]have(?:\s+skills?)?)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)
_PREFERRED_SECTION = re.compile(
    r"^\s*(?:preferred(?:\s+(?:skills?|qualifications?))?|"
    r"nice to have|bonus(?:\s+skills?)?|desirable(?:\s+skills?)?|"
    r"advantages?)\s*:?\s*$",
    re.IGNORECASE,
)
_OTHER_SECTION = re.compile(
    r"^\s*(?:responsibilities|what you(?:'|’)ll do|about the role|"
    r"what we offer|benefits|company overview)\s*:?\s*$",
    re.IGNORECASE,
)
_REQUIRED_CUE = re.compile(
    r"\b(?:required|must[- ]have|should[- ]have|essential|mandatory|minimum|"
    r"need(?:ed)?|proficiency in|proficient in|good understanding of|"
    r"understanding of|basic knowledge of|knowledge of|familiarity with|"
    r"experience with|experience building|experience using)"
    r"\b[^.!?;]{0,100}$",
    re.IGNORECASE,
)
_REQUIRED_CUE_AFTER = re.compile(
    r"^[^.!?;]{0,70}\b(?:is\s+)?(?:required|mandatory|essential|a requirement)\b",
    re.IGNORECASE,
)
_PREFERRED_CUE = re.compile(
    r"\b(?:preferred|preferably|nice to have|a plus|bonus|desirable|"
    r"advantage|advantageous|ideally|would be a plus)\b[^.!?;]{0,100}$",
    re.IGNORECASE,
)
_PREFERRED_CUE_AFTER = re.compile(
    r"^[^.!?;]{0,70}\b(?:is\s+)?(?:preferred|a plus|a bonus|desirable|"
    r"an advantage|advantageous)\b",
    re.IGNORECASE,
)
_HEADING_WITH_CONTENT = re.compile(
    r"^\s*(required(?:\s+(?:skills?|qualifications?))?|requirements?|"
    r"minimum qualifications?|basic qualifications?|must[- ]have(?:\s+skills?)?|"
    r"preferred(?:\s+(?:skills?|qualifications?))?|nice to have|"
    r"bonus(?:\s+skills?)?|desirable(?:\s+skills?)?|advantages?)\s*:\s*(.+)$",
    re.IGNORECASE,
)
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*")
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_OR_CONNECTOR = re.compile(r"^(?:\s|,)*\b(?:or|either)\b(?:\s|,)*$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def _find_skill_occurrences(text: str) -> list[tuple[str, int, int]]:
    """Return canonical skill names and source spans for all vocabulary matches."""
    occurrences: set[tuple[str, int, int]] = set()
    patterns = list(_COMPILED_PATTERNS.items())
    patterns.extend(_ALIAS_PATTERNS.items())
    for skill_name, pattern in patterns:
        occurrences.update(
            (skill_name, match.start(), match.end())
            for match in pattern.finditer(text)
        )
    return sorted(occurrences, key=lambda match: (match[1], match[2], match[0]))


def _has_negative_or_aspirational_context(
    text: str,
    start: int,
    end: int,
) -> bool:
    """Ignore clear negations and statements of intent to learn a skill."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    before = text[max(line_start, start - 100):start]
    after = text[end:min(line_end, end + 70)]
    return bool(
        _NEGATED_BEFORE.search(before)
        or _NEGATED_AFTER.search(after)
        or _ASPIRATIONAL_BEFORE.search(before)
    )


def _extract_positive_skills(text: object) -> Set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    return {
        skill_name
        for skill_name, start, end in _find_skill_occurrences(text)
        if not _has_negative_or_aspirational_context(text, start, end)
    }


def extract_skills(text: object) -> Set[str]:
    """
    Scan *text* and return canonical skills in positive, non-aspirational context.

    Parameters
    ----------
    text : str
        Raw text (resume or job description). May be empty or non-string.

    Returns
    -------
    set[str]
        Canonical skill names detected in the text (e.g. "Python"). Aliases
        such as "JS" and "JavaScript" produce one canonical skill.
    """
    return _extract_positive_skills(text)


def extract_resume_skills(resume_text: object) -> Set[str]:
    """Extract canonical, positively-contextualized skills from resume text."""
    return extract_skills(resume_text)


def _heading_classification(line: str) -> tuple[bool, str | None]:
    """Return whether a line is a recognized section heading and its class."""
    normalized_line = _MARKDOWN_HEADING.sub("", line)
    normalized_line = _BULLET_PREFIX.sub("", normalized_line).strip()
    content_heading = _HEADING_WITH_CONTENT.match(normalized_line)
    heading_text = content_heading.group(1) if content_heading else normalized_line
    if _REQUIRED_SECTION.match(heading_text):
        return True, "required"
    if _PREFERRED_SECTION.match(heading_text):
        return True, "preferred"
    if _OTHER_SECTION.match(heading_text):
        return True, None
    return False, None


def _classify_jd_occurrence(
    line: str,
    start: int,
    end: int,
    section_class: str | None,
) -> str | None:
    """Classify a skill occurrence only when local or section evidence is clear."""
    before = line[max(0, start - 80):start]
    after = line[end:min(len(line), end + 60)]
    required = bool(
        _REQUIRED_CUE.search(before)
        or _REQUIRED_CUE_AFTER.search(after)
    )
    preferred = bool(
        _PREFERRED_CUE.search(before)
        or _PREFERRED_CUE_AFTER.search(after)
    )
    if preferred:
        return "preferred"
    if required:
        return "required"
    return section_class


def _find_alternative_groups(
    line: str,
    occurrences: list[tuple[str, int, int]],
    occurrence_classes: dict[tuple[str, int, int], str | None],
) -> list[tuple[Set[str], str]]:
    """Find same-line ``or`` groups with one shared classification."""
    groups: list[tuple[Set[str], str]] = []
    current_skills: Set[str] = set()
    current_class: str | None = None
    previous_end: int | None = None

    def finish_group() -> None:
        if len(current_skills) > 1 and current_class is not None:
            groups.append((set(current_skills), current_class))

    for skill_name, start, end in occurrences:
        classification = occurrence_classes[(skill_name, start, end)]
        if previous_end is not None:
            between = line[previous_end:start]
            connects = bool(_OR_CONNECTOR.fullmatch(between))
            if (
                connects
                and classification == current_class
                and classification is not None
            ):
                current_skills.add(skill_name)
                previous_end = end
                continue
            finish_group()

        current_skills = {skill_name}
        current_class = classification
        previous_end = end

    finish_group()
    return groups


def _category_groups(
    skills: Set[str],
    alternatives: list[Set[str]],
) -> list[Set[str]]:
    """Represent alternative sets as one requirement and other skills singly."""
    groups = [group & skills for group in alternatives if group & skills]
    grouped_skills = set().union(*groups) if groups else set()
    groups.extend({skill} for skill in sorted(skills - grouped_skills))
    return sorted(groups, key=lambda group: tuple(sorted(group)))


def extract_jd_skills(jd_text: object) -> dict[str, Any]:
    """Extract JD skills and conservatively classify requirement strength.

    Skills not tied to an unambiguous requirement phrase or recognized section
    remain unclassified. A mention in a negated or aspirational context is not
    counted as a detected JD skill.
    """
    result: dict[str, Set[str]] = {
        "jd_skills": set(),
        "required_skills": set(),
        "preferred_skills": set(),
        "unclassified_jd_skills": set(),
        "required_skill_groups": [],
        "preferred_skill_groups": [],
    }
    if not isinstance(jd_text, str) or not jd_text.strip():
        return result

    classifications: dict[str, set[str]] = {}
    alternative_groups: list[tuple[Set[str], str]] = []
    section_class: str | None = None
    offset = 0
    for line in jd_text.splitlines(keepends=True):
        line_text = line.rstrip("\r\n")
        is_heading, new_section_class = _heading_classification(line_text)
        if is_heading:
            section_class = new_section_class

        occurrences = _find_skill_occurrences(line_text)
        occurrence_classes: dict[tuple[str, int, int], str | None] = {}
        for skill_name, start, end in occurrences:
            absolute_start = offset + start
            absolute_end = offset + end
            if _has_negative_or_aspirational_context(
                jd_text, absolute_start, absolute_end
            ):
                occurrence_classes[(skill_name, start, end)] = None
                continue
            classification = _classify_jd_occurrence(
                line_text, start, end, section_class
            )
            occurrence_classes[(skill_name, start, end)] = classification
            classifications.setdefault(skill_name, set())
            if classification is not None:
                classifications[skill_name].add(classification)
        alternative_groups.extend(
            _find_alternative_groups(line_text, occurrences, occurrence_classes)
        )
        offset += len(line)

    required_alternatives: list[Set[str]] = []
    preferred_alternatives: list[Set[str]] = []
    for skill_name, evidence in classifications.items():
        result["jd_skills"].add(skill_name)
        if evidence == {"required"}:
            result["required_skills"].add(skill_name)
        elif evidence == {"preferred"}:
            result["preferred_skills"].add(skill_name)
        else:
            result["unclassified_jd_skills"].add(skill_name)
    for skill_group, classification in alternative_groups:
        category_skills = result[f"{classification}_skills"]
        if skill_group <= category_skills:
            destination = (
                required_alternatives
                if classification == "required"
                else preferred_alternatives
            )
            if skill_group not in destination:
                destination.append(skill_group)
    result["required_skill_groups"] = _category_groups(
        result["required_skills"], required_alternatives
    )
    result["preferred_skill_groups"] = _category_groups(
        result["preferred_skills"], preferred_alternatives
    )
    return result


# ---------------------------------------------------------------------------
# High-level analyser (used by predict.py and tests)
# ---------------------------------------------------------------------------

def analyse_skills(resume_text: object, jd_text: object) -> dict:
    """
    Extract and compare skills between a resume and a job description.

    Returns
    -------
    The legacy set-valued keys remain available. Additional set-valued keys
    describe classified and unclassified JD skills. ``coverage_metrics``
    contains ratios and counts; a ratio is ``None`` when its denominator is
    zero, because coverage cannot be calculated without detected skills.
    """
    resume_skills = extract_resume_skills(resume_text)
    jd_analysis = extract_jd_skills(jd_text)
    jd_skills = jd_analysis["jd_skills"]

    matched_skills = resume_skills & jd_skills
    missing_skills = jd_skills - resume_skills
    matched_required = resume_skills & jd_analysis["required_skills"]
    missing_required = jd_analysis["required_skills"] - resume_skills
    matched_preferred = resume_skills & jd_analysis["preferred_skills"]
    missing_preferred = jd_analysis["preferred_skills"] - resume_skills

    def coverage(matched_count: int, total_count: int) -> float | None:
        return matched_count / total_count if total_count else None

    required_groups = jd_analysis["required_skill_groups"]
    preferred_groups = jd_analysis["preferred_skill_groups"]
    matched_required_groups = sum(bool(group & resume_skills) for group in required_groups)
    matched_preferred_groups = sum(bool(group & resume_skills) for group in preferred_groups)
    missing_required = {
        skill
        for group in required_groups
        if not group & resume_skills
        for skill in group
    }
    missing_preferred = {
        skill
        for group in preferred_groups
        if not group & resume_skills
        for skill in group
    }
    coverage_metrics = {
        "required_skill_coverage": coverage(
            matched_required_groups, len(required_groups)
        ),
        "preferred_skill_coverage": coverage(
            matched_preferred_groups, len(preferred_groups)
        ),
        "overall_skill_coverage": coverage(len(matched_skills), len(jd_skills)),
        "required_skill_count": len(required_groups),
        "preferred_skill_count": len(preferred_groups),
        "matched_required_group_count": matched_required_groups,
        "matched_preferred_group_count": matched_preferred_groups,
        "unclassified_skill_count": len(jd_analysis["unclassified_jd_skills"]),
        "matched_skill_count": len(matched_skills),
        "missing_skill_count": len(missing_skills),
    }

    return {
        "resume_skills": resume_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "jd_skills": jd_skills,
        "required_skills": jd_analysis["required_skills"],
        "preferred_skills": jd_analysis["preferred_skills"],
        "required_skill_groups": required_groups,
        "preferred_skill_groups": preferred_groups,
        "matched_required_skills": matched_required,
        "missing_required_skills": missing_required,
        "matched_preferred_skills": matched_preferred,
        "missing_preferred_skills": missing_preferred,
        "unclassified_jd_skills": jd_analysis["unclassified_jd_skills"],
        "total_jd_skills": len(jd_skills),
        "coverage_metrics": coverage_metrics,
        "extraction_limitations": [
            "Keyword detection does not establish actual candidate competence.",
            "Only skills in the curated vocabulary and recognized aliases are detected.",
            "Required/preferred classification uses conservative section and phrase rules; ambiguous mentions remain unclassified.",
            "Negation and aspirational-context detection is heuristic and may miss complex wording.",
        ],
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
      [-]  JD skill not detected in resume
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

    print(f"\n  JD skills not detected in resume  ({len(missing_skills)}):")
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
