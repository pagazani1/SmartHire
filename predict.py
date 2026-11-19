"""
predict.py – SmartHire single-sample inference
-----------------------------------------------
Usage:
    python predict.py

The script will prompt you to paste a resume and a job description,
then print the predicted fit category and confidence scores for all
three classes: Good Fit, No Fit, Potential Fit.

No new dependencies required – only joblib and scikit-learn, which
are already listed in requirements.txt.
"""

import os
import sys
import threading
from pathlib import Path

import joblib

# Re-use the same clean_text() that was used during training.
# This keeps the preprocessing identical so predictions are reliable.
from src.preprocessing import clean_text

# Rule-based skill extraction (runs separately from the ML classifier).
# Skill matches are NOT an explanation of the model's internal reasoning;
# they are a human-readable keyword summary of both texts.
from src.skill_extractor import analyse_skills, display_skills

# ---------------------------------------------------------------------------
# Paths to saved model artifacts (mirrors config.py but kept explicit here
# so predict.py can be run standalone without importing the full pipeline).
# ---------------------------------------------------------------------------
# Absolute paths anchored to this file, so inference works from any working
# directory (the API server, tests and the CLI all resolve the same artifacts).
_PROJECT_ROOT = Path(__file__).resolve().parent
VECTORIZER_PATH = str(_PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib")
MODEL_PATH      = str(_PROJECT_ROOT / "models" / "logistic_regression.joblib")


def load_artifacts(verbose: bool = False):
    """Load the TF-IDF vectorizer and the trained Logistic Regression model.

    Reads both files from disk on every call. Server code should use
    ``load_artifacts_cached`` instead. Progress messages are printed only when
    ``verbose`` is true (the interactive CLI), never inside the API process.
    """
    if verbose:
        print("[predict] Loading TF-IDF vectorizer...")
    vectorizer = joblib.load(VECTORIZER_PATH)

    if verbose:
        print("[predict] Loading Logistic Regression model...")
    model = joblib.load(MODEL_PATH)

    if verbose:
        print("[predict] Artifacts loaded successfully.\n")
    return vectorizer, model


_ARTIFACT_CACHE: dict[tuple, tuple] = {}
_ARTIFACT_CACHE_LOCK = threading.Lock()


def _artifact_cache_key() -> tuple:
    """Identify the artifact files by path, size and modification time."""
    parts = []
    for path in (VECTORIZER_PATH, MODEL_PATH):
        stat = os.stat(path)  # FileNotFoundError propagates like joblib's would
        parts.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(parts)


def load_artifacts_cached():
    """Return the artifacts, loading them from disk only once per file version.

    The cache key includes size and mtime, so replacing a model file on disk
    is picked up on the next call without restarting the process. Failed loads
    are never cached. Loaded scikit-learn objects are used read-only by
    ``predict``, so sharing them between request threads is safe.
    """
    key = _artifact_cache_key()
    with _ARTIFACT_CACHE_LOCK:
        cached = _ARTIFACT_CACHE.get(key)
        if cached is not None:
            return cached
        artifacts = load_artifacts()
        _ARTIFACT_CACHE.clear()
        _ARTIFACT_CACHE[key] = artifacts
        return artifacts


def get_multiline_input(prompt: str) -> str:
    """
    Prompt the user for multi-line text.
    They press Enter twice (blank line) to finish.
    """
    print(prompt)
    print("(Press Enter twice when done)\n")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            # Two consecutive blank lines → end of input
            break
        lines.append(line)
    # Strip the trailing blank line and join
    text = "\n".join(lines).strip()
    return text


def validate_text_input(text: object, name: str) -> str:
    """Validate an inference text value and return it unchanged."""
    if not isinstance(text, str):
        raise ValueError(f"{name} must be a string.")
    if not text.strip():
        raise ValueError(f"{name} cannot be empty. Please provide some text.")
    return text


def validate_input(text: str, name: str) -> None:
    """Preserve the CLI's printed error and exit behavior for invalid input."""
    try:
        validate_text_input(text, name)
    except ValueError as error:
        print(f"\n[ERROR] {error}")
        sys.exit(1)


def predict(resume_text: str, jd_text: str, vectorizer, model) -> dict:
    """
    Run one prediction for a (resume, job description) pair.

    Steps:
      1. Clean both texts using the same clean_text() used in training.
      2. Combine them the same way the pipeline does: resume_clean + " " + jd_clean
      3. Transform with the fitted TF-IDF vectorizer.
      4. Predict label and per-class probabilities with Logistic Regression.

    Returns a dict with:
        - 'label'  : predicted class string
        - 'probs'  : dict mapping each class name → probability (0-1)
    """
    validate_text_input(resume_text, "Resume")
    validate_text_input(jd_text, "Job description")

    # Step 1 – clean
    resume_clean = clean_text(resume_text)
    jd_clean     = clean_text(jd_text)

    # Step 2 – combine (identical to training pipeline)
    combined_text = resume_clean + " " + jd_clean

    # Step 3 – vectorize (transform only – do NOT fit)
    X = vectorizer.transform([combined_text])

    # Step 4 – predict
    label      = model.predict(X)[0]
    proba_vals = model.predict_proba(X)[0]          # one probability per class
    classes    = model.classes_                      # e.g. ['Good Fit', 'No Fit', 'Potential Fit']

    probs = {cls: float(p) for cls, p in zip(classes, proba_vals)}

    return {"label": label, "probs": probs}


def display_result(result: dict) -> None:
    """Print the prediction result in a readable format.

    Uses plain ASCII so output works on Windows cp1252 terminals.
      [>>]  marks the predicted class
    """
    label = result["label"]
    probs = result["probs"]

    # Plain-text label indicators (no emoji – avoids cp1252 UnicodeEncodeError)
    icons = {"Good Fit": "[GOOD FIT]", "Potential Fit": "[POTENTIAL FIT]", "No Fit": "[NO FIT]"}
    icon  = icons.get(label, "[?]")

    print("\n" + "=" * 50)
    print("         SMARTHIRE PREDICTION RESULT")
    print("=" * 50)
    print(f"  Predicted Category : {icon}")
    print("-" * 50)
    print("  Confidence Scores:")
    # Sort by probability descending for easy reading
    for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        bar    = "#" * int(prob * 20)          # simple ASCII bar (max 20 chars)
        marker = " << predicted" if cls == label else ""
        print(f"    {cls:<15}  {prob:.1%}  {bar}{marker}")
    print("=" * 50 + "\n")


def main():
    print("=" * 50)
    print("    SmartHire – Resume-Job Fit Predictor")
    print("=" * 50 + "\n")

    # 1. Load model artifacts
    vectorizer, model = load_artifacts(verbose=True)

    # 2. Collect resume text
    resume_text = get_multiline_input("Paste the RESUME text below:")
    validate_input(resume_text, "Resume")

    # 3. Collect job description text
    jd_text = get_multiline_input("Paste the JOB DESCRIPTION text below:")
    validate_input(jd_text, "Job description")

    # 4. Run prediction
    print("\n[predict] Running prediction...")
    result = predict(resume_text, jd_text, vectorizer, model)

    # 5. Display ML prediction result (label + confidence scores)
    display_result(result)

    # 6. Skill analysis (rule-based, separate from the ML model)
    # This identifies keywords from a fixed vocabulary in each text.
    # NOTE: Skills listed as 'not detected in resume' mean the keyword
    # was not found in the resume text -- not that the candidate lacks
    # the skill.  The model may have weighted other signals.
    print("[predict] Running skill analysis...")
    skill_result = analyse_skills(resume_text, jd_text)
    display_skills(skill_result)


if __name__ == "__main__":
    main()