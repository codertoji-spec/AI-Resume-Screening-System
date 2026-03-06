"""
skill_extractor.py
------------------
Detects technical skills mentioned in resume text and job descriptions.
Uses a curated skill vocabulary with alias support so that, for example,
"ML" maps to "machine learning".

This is intentionally rule-based (no heavy NLP model required) so the
project stays beginner-friendly and fast to run.
"""

import re
from typing import List, Set

# ---------------------------------------------------------------------------
# Master skill list
# Each entry is a tuple: (canonical_name, [aliases_or_variants])
# Matching is case-insensitive and whole-word.
# ---------------------------------------------------------------------------
SKILL_VOCAB: List[tuple] = [
    # Programming languages
    ("python",            ["python3", "python 3"]),
    ("java",              ["java 8", "java 11", "java 17"]),
    ("javascript",        ["js", "node.js", "nodejs", "node js"]),
    ("typescript",        ["ts"]),
    ("c++",               ["cpp", "c plus plus"]),
    ("c#",                ["csharp", "c sharp", ".net"]),
    ("r",                 ["r language", "r programming"]),
    ("scala",             []),
    ("go",                ["golang"]),
    ("rust",              []),
    ("kotlin",            []),
    ("swift",             []),
    ("php",               []),
    ("ruby",              ["ruby on rails", "rails"]),

    # Data & ML
    ("machine learning",  ["ml", "machine-learning"]),
    ("deep learning",     ["dl", "deep-learning"]),
    ("nlp",               ["natural language processing", "natural-language processing"]),
    ("computer vision",   ["cv", "image recognition", "object detection"]),
    ("data analysis",     ["data analytics", "data analyst", "eda"]),
    ("data science",      ["data scientist"]),
    ("statistics",        ["statistical analysis", "statistical modeling"]),
    ("feature engineering", []),
    ("model deployment",  ["mlops", "ml ops", "model serving"]),

    # ML Frameworks
    ("tensorflow",        ["tf", "tensorflow 2"]),
    ("pytorch",           ["torch"]),
    ("keras",             []),
    ("scikit-learn",      ["sklearn", "scikit learn"]),
    ("xgboost",           ["xgb"]),
    ("lightgbm",          ["lgbm"]),
    ("hugging face",      ["huggingface", "transformers"]),

    # Databases
    ("sql",               ["mysql", "postgresql", "postgres", "sqlite", "t-sql", "pl/sql"]),
    ("nosql",             ["mongodb", "cassandra", "dynamodb", "couchdb"]),
    ("redis",             []),
    ("elasticsearch",     ["elastic search"]),

    # Cloud & DevOps
    ("aws",               ["amazon web services", "amazon aws", "ec2", "s3", "lambda"]),
    ("azure",             ["microsoft azure"]),
    ("gcp",               ["google cloud", "google cloud platform"]),
    ("docker",            ["containerization", "containers"]),
    ("kubernetes",        ["k8s"]),
    ("ci/cd",             ["ci cd", "continuous integration", "continuous deployment", "jenkins", "github actions"]),
    ("git",               ["github", "gitlab", "version control"]),

    # Data Engineering
    ("apache spark",      ["spark", "pyspark"]),
    ("apache kafka",      ["kafka"]),
    ("airflow",           ["apache airflow"]),
    ("dbt",               ["data build tool"]),
    ("hadoop",            ["hdfs", "mapreduce"]),

    # Visualization
    ("tableau",           []),
    ("power bi",          ["powerbi"]),
    ("matplotlib",        []),
    ("seaborn",           []),
    ("plotly",            []),

    # Web / APIs
    ("rest api",          ["restful", "rest", "api development"]),
    ("graphql",           []),
    ("flask",             []),
    ("django",            []),
    ("fastapi",           ["fast api"]),
    ("react",             ["reactjs", "react.js"]),
    ("angular",           ["angularjs"]),
    ("vue",               ["vuejs", "vue.js"]),

    # Soft / domain skills
    ("agile",             ["scrum", "kanban", "sprint"]),
    ("communication",     []),
    ("leadership",        ["team lead", "tech lead"]),
    ("problem solving",   ["problem-solving"]),
    ("research",          []),
]

# Build a flat lookup: pattern → canonical name
_PATTERNS: List[tuple] = []   # (compiled_regex, canonical_name)

def _build_patterns():
    """Pre-compile regex patterns once at import time."""
    for canonical, aliases in SKILL_VOCAB:
        terms = [canonical] + aliases
        for term in terms:
            # Escape special chars, then wrap in word-boundary assertion
            escaped = re.escape(term)
            # For very short tokens (≤2 chars like "r", "go") use strict boundaries
            pattern = re.compile(
                r"(?<![a-zA-Z0-9_])" + escaped + r"(?![a-zA-Z0-9_])",
                re.IGNORECASE,
            )
            _PATTERNS.append((pattern, canonical))

_build_patterns()


def extract_skills(text: str) -> List[str]:
    """
    Return a deduplicated, sorted list of canonical skill names found in *text*.

    Parameters
    ----------
    text : str

    Returns
    -------
    List[str]  e.g. ["machine learning", "python", "sql"]
    """
    found: Set[str] = set()
    for pattern, canonical in _PATTERNS:
        if pattern.search(text):
            found.add(canonical)
    return sorted(found)


def skill_match_score(resume_skills: List[str], required_skills: List[str]) -> float:
    """
    Compute what fraction of *required_skills* appear in *resume_skills*.

    Parameters
    ----------
    resume_skills   : skills extracted from the resume
    required_skills : skills extracted from the job description

    Returns
    -------
    float in [0, 1].  Returns 0.0 if required_skills is empty.
    """
    if not required_skills:
        return 0.0
    matched = set(resume_skills) & set(required_skills)
    return len(matched) / len(required_skills)


def get_matched_skills(resume_skills: List[str], required_skills: List[str]) -> List[str]:
    """Return skills that appear in both lists (for display purposes)."""
    return sorted(set(resume_skills) & set(required_skills))


def get_missing_skills(resume_skills: List[str], required_skills: List[str]) -> List[str]:
    """Return required skills missing from the resume."""
    return sorted(set(required_skills) - set(resume_skills))
