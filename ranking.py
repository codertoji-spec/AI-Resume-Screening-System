"""
ranking.py
----------
Generates sentence embeddings with SentenceTransformer, computes
cosine similarity, then blends semantic similarity with a rule-based
skill-match score to produce a final ranking.

Final Score = 0.7 * semantic_similarity + 0.3 * skill_match_score
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Lazy-import so the app can still show UI errors gracefully
_model = None
MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    """Load (and cache) the SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised embedding vectors.

    Parameters
    ----------
    texts : List[str]

    Returns
    -------
    np.ndarray  shape (n, embedding_dim)
    """
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # L2-normalise so cosine similarity == dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)          # avoid division by zero
    return embeddings / norms


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D (or 2-D row) vectors.

    Returns a float in [0, 1] (embeddings are already non-negative after
    the MiniLM model, though not guaranteed).
    """
    a = vec_a.reshape(1, -1)
    b = vec_b.reshape(1, -1)
    score = cosine_similarity(a, b)[0][0]
    # Clamp to [0, 1] for display safety
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Core ranking function
# ---------------------------------------------------------------------------

SEMANTIC_WEIGHT = 0.7
SKILL_WEIGHT    = 0.3


def rank_candidates(
    job_description: str,
    resumes: List[Dict[str, Any]],
    job_skills: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Rank a list of resumes against a job description.

    Parameters
    ----------
    job_description : str
        Full text of the job posting.

    resumes : List[dict]
        Each dict must have:
          - "name"         : str   candidate display name
          - "text"         : str   full resume text
          - "file"         : str   original filename
          - "skills"       : List[str]  skills extracted from resume
          - "skill_score"  : float      skill_match_score (0–1)
          - "matched_skills": List[str] skills in both resume & JD
          - "missing_skills": List[str] required skills not found

    job_skills : List[str], optional
        Skills extracted from the JD (for display in the output table).

    Returns
    -------
    pd.DataFrame  sorted descending by final_score, with columns:
        rank, name, file, semantic_similarity, skill_match_score,
        final_score, matched_skills, missing_skills, skills_in_resume
    """
    if not resumes:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1. Embed job description
    # ------------------------------------------------------------------
    jd_embedding = embed_texts([job_description])[0]

    # ------------------------------------------------------------------
    # 2. Embed all resumes in one batch (faster than one-by-one)
    # ------------------------------------------------------------------
    resume_texts = [r["text"] for r in resumes]
    resume_embeddings = embed_texts(resume_texts)

    # ------------------------------------------------------------------
    # 3. Score each candidate
    # ------------------------------------------------------------------
    rows = []
    for i, resume in enumerate(resumes):
        sem_sim    = compute_cosine_similarity(resume_embeddings[i], jd_embedding)
        skill_sc   = resume.get("skill_score", 0.0)
        final_sc   = SEMANTIC_WEIGHT * sem_sim + SKILL_WEIGHT * skill_sc

        rows.append(
            {
                "name":               resume["name"],
                "file":               resume["file"],
                "semantic_similarity": round(sem_sim,  4),
                "skill_match_score":  round(skill_sc,  4),
                "final_score":        round(final_sc,  4),
                "matched_skills":     ", ".join(resume.get("matched_skills", [])),
                "missing_skills":     ", ".join(resume.get("missing_skills", [])),
                "skills_in_resume":   ", ".join(resume.get("skills",         [])),
            }
        )

    # ------------------------------------------------------------------
    # 4. Sort & assign rank
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# Convenience: percentage helpers for UI display
# ---------------------------------------------------------------------------

def score_to_percent(score: float) -> str:
    return f"{score * 100:.1f}%"


def score_to_stars(score: float, max_stars: int = 5) -> str:
    filled = round(score * max_stars)
    return "★" * filled + "☆" * (max_stars - filled)
