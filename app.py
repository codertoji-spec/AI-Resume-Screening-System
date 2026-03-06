"""
app.py
------
AI Resume Screening System — Streamlit front-end.

Run with:
    streamlit run app.py
"""

import io
import os
import time
import streamlit as st
import pandas as pd

from resume_parser   import parse_resume
from skill_extractor import extract_skills, skill_match_score, get_matched_skills, get_missing_skills
from ranking         import rank_candidates, score_to_percent, score_to_stars, MODEL_NAME

# ─── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Font import ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Main background ── */
    .stApp { background: #0d1117; color: #e6edf3; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    /* ── Cards ── */
    .candidate-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        transition: border-color 0.2s;
    }
    .candidate-card:hover { border-color: #58a6ff; }

    /* Rank badges */
    .rank-1  { background: linear-gradient(135deg, #ffd700, #ff8c00); color: #000; }
    .rank-2  { background: linear-gradient(135deg, #c0c0c0, #8a8a8a); color: #000; }
    .rank-3  { background: linear-gradient(135deg, #cd7f32, #8b4513); color: #fff; }
    .rank-other { background: #21262d; color: #8b949e; }

    .rank-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Score bar */
    .score-bar-bg {
        background: #21262d;
        border-radius: 9999px;
        height: 8px;
        width: 100%;
        margin-top: 4px;
    }
    .score-bar-fill {
        height: 8px;
        border-radius: 9999px;
        background: linear-gradient(90deg, #238636, #2ea043);
    }
    .score-bar-fill-blue {
        height: 8px;
        border-radius: 9999px;
        background: linear-gradient(90deg, #1f6feb, #388bfd);
    }

    /* Skill chips */
    .skill-chip {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.73rem;
        font-weight: 500;
        margin: 2px 3px;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip-match   { background: #0d4429; color: #3fb950; border: 1px solid #238636; }
    .chip-missing { background: #2d0e0e; color: #f85149; border: 1px solid #da3633; }
    .chip-extra   { background: #1c2128; color: #8b949e; border: 1px solid #30363d; }

    /* Metric box */
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
    .metric-label { font-size: 0.8rem; color: #8b949e; margin-top: 2px; }

    /* Header gradient */
    .page-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #58a6ff, #a5d6ff, #79c0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Table overrides */
    .dataframe thead th { background: #161b22 !important; color: #8b949e !important; }
    .dataframe tbody tr:hover td { background: #1c2128 !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        font-size: 1rem;
        font-weight: 600;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Info / warning */
    .stAlert { border-radius: 8px; }

    /* Remove default Streamlit padding */
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**🤖 Embedding Model**")
    st.code(MODEL_NAME, language=None)
    st.caption("Generates dense vector representations of text.")

    st.markdown("---")
    st.markdown("**📊 Scoring Weights**")
    semantic_weight = st.slider("Semantic Similarity", 0.0, 1.0, 0.7, 0.05,
                                help="Weight for embedding-based cosine similarity.")
    skill_weight = round(1.0 - semantic_weight, 2)
    st.info(f"Skill Match Weight: **{skill_weight}**  *(auto-calculated)*")
    st.caption("Both weights always sum to 1.0")

    st.markdown("---")
    st.markdown("**ℹ️ How It Works**")
    st.markdown(
        """
        1. 📄 Upload resumes (PDF / TXT)
        2. 📝 Enter a job description
        3. 🔍 Skills are extracted from both
        4. 🧠 Embeddings generated via **SentenceTransformers**
        5. 📐 Cosine similarity computed
        6. 🏆 Candidates ranked by weighted score
        """
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>Built with ❤️ · SentenceTransformers + Streamlit</small>",
        unsafe_allow_html=True,
    )


# ─── Page title ───────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">🤖 AI Resume Screening System</p>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#8b949e; margin-top:-0.5rem;'>Rank candidates using NLP embeddings + skill matching — no manual review required.</p>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)


# ─── Two-column layout ────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT: Job Description ────────────────────────────────────────────────────
with col_left:
    st.markdown("### 📋 Job Description")
    default_jd = (
        "We are looking for a Senior Machine Learning Engineer with strong Python skills.\n\n"
        "Requirements:\n"
        "• 4+ years experience in machine learning and deep learning\n"
        "• Strong proficiency in Python, TensorFlow or PyTorch\n"
        "• Experience with NLP techniques and transformer models\n"
        "• Familiarity with SQL databases and data analysis\n"
        "• Knowledge of AWS cloud services and Docker\n"
        "• Experience deploying models at scale (MLOps)\n"
        "• Familiarity with Apache Spark or Kafka is a plus\n"
        "• Excellent communication and teamwork skills"
    )
    job_description = st.text_area(
        label="Paste the job description here",
        value=default_jd,
        height=280,
        placeholder="Paste the full job description...",
        help="The more detailed the description, the better the semantic match.",
        label_visibility="collapsed",
    )

    if job_description.strip():
        jd_skills = extract_skills(job_description)
        if jd_skills:
            st.markdown(
                f"<small style='color:#8b949e'>📌 **{len(jd_skills)} skill(s) detected in JD:**</small>",
                unsafe_allow_html=True,
            )
            chips_html = "".join(
                f'<span class="skill-chip chip-match">✓ {s}</span>' for s in jd_skills
            )
            st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)
        else:
            st.caption("No skills detected in JD yet.")
    st.markdown("<br>", unsafe_allow_html=True)


# ── RIGHT: Resume Upload ──────────────────────────────────────────────────────
with col_right:
    st.markdown("### 📎 Upload Resumes")
    uploaded_files = st.file_uploader(
        label="Upload candidate resumes (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more resumes. PDF and plain-text files are both supported.",
        label_visibility="collapsed",
    )

    # Demo mode toggle
    use_demo = st.checkbox(
        "🎯 Use built-in demo resumes (dataset/ folder)",
        value=(len(uploaded_files) == 0),
        help="Load the 3 sample resumes from the dataset/ directory for a quick demo.",
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        for f in uploaded_files:
            st.markdown(
                f"<small style='color:#8b949e'>📄 {f.name} ({f.size / 1024:.1f} KB)</small>",
                unsafe_allow_html=True,
            )
    elif use_demo:
        demo_dir = os.path.join(os.path.dirname(__file__), "dataset")
        demo_files = [f for f in os.listdir(demo_dir) if f.endswith((".txt", ".pdf"))]
        st.info(f"📂 Found {len(demo_files)} demo file(s) in dataset/: {', '.join(demo_files)}")

    st.markdown("<br>", unsafe_allow_html=True)


# ─── Rank button ──────────────────────────────────────────────────────────────
st.markdown("---")
btn_col, _ = st.columns([1, 3])
with btn_col:
    rank_clicked = st.button("🚀 Rank Candidates")

# ─── Processing ───────────────────────────────────────────────────────────────
if rank_clicked:

    if not job_description.strip():
        st.error("❌ Please enter a job description.")
        st.stop()

    # ── Collect resume files ──────────────────────────────────────────────
    resume_file_list = []  # list of (file_obj_or_path, filename)

    if uploaded_files:
        for uf in uploaded_files:
            resume_file_list.append((uf, uf.name))

    if use_demo and not uploaded_files:
        demo_dir = os.path.join(os.path.dirname(__file__), "dataset")
        for fname in sorted(os.listdir(demo_dir)):
            if fname.endswith((".txt", ".pdf")):
                fpath = os.path.join(demo_dir, fname)
                resume_file_list.append((fpath, fname))

    if not resume_file_list:
        st.error("❌ No resumes found. Please upload files or enable the demo mode.")
        st.stop()

    # ── Parse resumes ─────────────────────────────────────────────────────
    with st.spinner("📄 Parsing resumes…"):
        parsed_resumes = []
        parse_errors   = []
        for file_obj, filename in resume_file_list:
            try:
                result = parse_resume(file_obj, filename)
                if not result["text"]:
                    parse_errors.append(f"{filename}: extracted text is empty.")
                else:
                    parsed_resumes.append(result)
            except Exception as e:
                parse_errors.append(f"{filename}: {e}")

        if parse_errors:
            for err in parse_errors:
                st.warning(f"⚠️ {err}")

        if not parsed_resumes:
            st.error("❌ Could not parse any resumes. Check file formats.")
            st.stop()

    # ── Extract skills ────────────────────────────────────────────────────
    with st.spinner("🔍 Extracting skills…"):
        jd_skills = extract_skills(job_description)
        for r in parsed_resumes:
            r_skills         = extract_skills(r["text"])
            r["skills"]        = r_skills
            r["skill_score"]   = skill_match_score(r_skills, jd_skills)
            r["matched_skills"] = get_matched_skills(r_skills, jd_skills)
            r["missing_skills"] = get_missing_skills(r_skills, jd_skills)

    # ── Generate embeddings + rank ────────────────────────────────────────
    with st.spinner(f"🧠 Generating embeddings with **{MODEL_NAME}**… (first run downloads ~90 MB)"):
        t0 = time.time()
        # Override weights from sidebar sliders
        import ranking as _ranking_module
        _ranking_module.SEMANTIC_WEIGHT = semantic_weight
        _ranking_module.SKILL_WEIGHT    = skill_weight

        results_df = rank_candidates(
            job_description=job_description,
            resumes=parsed_resumes,
            job_skills=jd_skills,
        )
        elapsed = time.time() - t0

    # ─────────────────────────────────────────────────────────────────────
    # RESULTS SECTION
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 🏆 Ranking Results")
    st.caption(f"Processed {len(parsed_resumes)} resume(s) in {elapsed:.2f}s")

    # ── Summary metrics ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-box"><div class="metric-value">{len(parsed_resumes)}</div>'
            f'<div class="metric-label">Candidates Evaluated</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        top_score = results_df["final_score"].iloc[0]
        st.markdown(
            f'<div class="metric-box"><div class="metric-value">{score_to_percent(top_score)}</div>'
            f'<div class="metric-label">Top Candidate Score</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        avg_score = results_df["final_score"].mean()
        st.markdown(
            f'<div class="metric-box"><div class="metric-value">{score_to_percent(avg_score)}</div>'
            f'<div class="metric-label">Average Score</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-box"><div class="metric-value">{len(jd_skills)}</div>'
            f'<div class="metric-label">Skills in Job Description</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Candidate cards ───────────────────────────────────────────────────
    st.markdown("### 📊 Detailed Candidate Profiles")

    for _, row in results_df.iterrows():
        rank        = int(row["rank"])
        name        = row["name"]
        final_score = float(row["final_score"])
        sem_sim     = float(row["semantic_similarity"])
        skill_sc    = float(row["skill_match_score"])
        matched     = [s.strip() for s in row["matched_skills"].split(",") if s.strip()]
        missing     = [s.strip() for s in row["missing_skills"].split(",") if s.strip()]
        all_skills  = [s.strip() for s in row["skills_in_resume"].split(",") if s.strip()]
        extra_skills = [s for s in all_skills if s not in matched and s not in missing]

        # Rank badge class
        badge_cls = {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-other")
        badge_icon = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")

        with st.container():
            st.markdown(
                f"""
                <div class="candidate-card">
                  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                    <div>
                      <span class="rank-badge {badge_cls}">{badge_icon} Rank {rank}</span>
                      &nbsp;
                      <span style="font-size:1.15rem; font-weight:600; color:#e6edf3;">{name}</span>
                      <span style="color:#8b949e; font-size:0.8rem; margin-left:0.5rem;">· {row['file']}</span>
                    </div>
                    <div style="text-align:right;">
                      <span style="font-size:1.5rem; font-weight:700; color:#58a6ff;">{score_to_percent(final_score)}</span>
                      <span style="color:#8b949e; font-size:0.8rem;"> final score</span>
                    </div>
                  </div>
                """,
                unsafe_allow_html=True,
            )

            # Score bars
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(
                    f"""
                    <div style="margin-top:0.75rem;">
                      <div style="display:flex; justify-content:space-between;">
                        <small style="color:#8b949e;">🧠 Semantic Similarity</small>
                        <small style="color:#3fb950; font-family:'JetBrains Mono',monospace;">{score_to_percent(sem_sim)}</small>
                      </div>
                      <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{sem_sim*100:.1f}%;"></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sc2:
                st.markdown(
                    f"""
                    <div style="margin-top:0.75rem;">
                      <div style="display:flex; justify-content:space-between;">
                        <small style="color:#8b949e;">🎯 Skill Match</small>
                        <small style="color:#388bfd; font-family:'JetBrains Mono',monospace;">{score_to_percent(skill_sc)}</small>
                      </div>
                      <div class="score-bar-bg">
                        <div class="score-bar-fill-blue" style="width:{skill_sc*100:.1f}%;"></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Skill chips
            chips = ""
            if matched:
                chips += "".join(f'<span class="skill-chip chip-match">✓ {s}</span>' for s in matched)
            if missing:
                chips += "".join(f'<span class="skill-chip chip-missing">✗ {s}</span>' for s in missing)
            if extra_skills:
                chips += "".join(f'<span class="skill-chip chip-extra">{s}</span>' for s in extra_skills[:6])
                if len(extra_skills) > 6:
                    chips += f'<span class="skill-chip chip-extra">+{len(extra_skills)-6} more</span>'

            if chips:
                st.markdown(
                    f"""
                    <div style="margin-top:0.75rem;">
                      <small style="color:#8b949e; display:block; margin-bottom:4px;">
                        Skills &nbsp;·&nbsp;
                        <span style="color:#3fb950;">✓ matched</span> &nbsp;
                        <span style="color:#f85149;">✗ missing</span> &nbsp;
                        <span style="color:#8b949e;">extra</span>
                      </small>
                      {chips}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

    # ── Sortable data table ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 View Raw Results Table", expanded=False):
        display_df = results_df[
            ["rank", "name", "file", "semantic_similarity", "skill_match_score", "final_score",
             "matched_skills", "missing_skills"]
        ].copy()
        display_df.columns = [
            "Rank", "Candidate", "File", "Semantic Sim", "Skill Match", "Final Score",
            "Matched Skills", "Missing Skills"
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name="resume_ranking_results.csv",
            mime="text/csv",
        )

    # ── Score formula explainer ───────────────────────────────────────────
    with st.expander("🔢 Scoring Formula Explained", expanded=False):
        st.markdown(
            f"""
            **Final Score** is a weighted blend of two signals:

            ```
            Final Score = {semantic_weight} × Semantic Similarity
                        + {skill_weight} × Skill Match Score
            ```

            | Signal | Description | Weight |
            |---|---|---|
            | **Semantic Similarity** | Cosine similarity between resume & JD embeddings (via `{MODEL_NAME}`) | `{semantic_weight}` |
            | **Skill Match Score** | `matched skills ÷ required skills` | `{skill_weight}` |

            > Adjust weights in the **sidebar** to change the balance between deep semantic understanding and keyword-based skill matching.
            """
        )

# ─── Empty state ─────────────────────────────────────────────────────────────
elif not rank_clicked:
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 0; color: #8b949e;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎯</div>
            <h3 style="color: #e6edf3;">Ready to screen candidates</h3>
            <p>Fill in the job description, upload resumes (or use demo data),<br>then click <strong>Rank Candidates</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
