# 🤖 AI Resume Screening System

A beginner-friendly Machine Learning project that **automatically ranks resumes** against a job description using **NLP embeddings** and **skill matching** — no manual review required.

---

## 🌟 Features

| Feature | Description |
|---|---|
| 📄 **Resume Parsing** | Extracts text from `.pdf` and `.txt` files via **PyPDF2** |
| 🔍 **Skill Extraction** | Rule-based detection of 60+ tech & soft skills |
| 🧠 **Semantic Embeddings** | Dense vector representations via **SentenceTransformers** (`all-MiniLM-L6-v2`) |
| 📐 **Cosine Similarity** | Measures how semantically close each resume is to the job description |
| 🏆 **Weighted Ranking** | Blends semantic similarity (70%) + skill match score (30%) |
| 🖥️ **Streamlit UI** | Interactive web interface with candidate cards & score bars |

---

## 🗂️ Project Structure

```
AI_Resume_Screener/
│
├── app.py              ← Streamlit UI (main entry point)
├── resume_parser.py    ← PDF & TXT text extraction
├── skill_extractor.py  ← Rule-based skill detection
├── ranking.py          ← Embeddings + cosine similarity + ranking logic
├── requirements.txt    ← Python dependencies
│
├── dataset/
│   ├── resume1.txt     ← Demo: Senior ML Engineer (Alex Johnson)
│   ├── resume2.txt     ← Demo: Data Analyst (Priya Sharma)
│   └── resume3.txt     ← Demo: Backend Java Engineer (Marcus Williams)
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / Download the project

```bash
# If using git
git clone <your-repo-url>
cd AI_Resume_Screener
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` model (~90 MB). This only happens once and is cached automatically.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**

---

## 🎯 Example Usage

### Quick demo (no uploads needed)

1. Launch the app
2. Leave the default job description in place
3. Check **"Use built-in demo resumes"**
4. Click **🚀 Rank Candidates**

You should see the 3 demo resumes ranked, with Alex Johnson (Senior ML Engineer) scoring highest against the ML job description.

### Testing with your own job description

Paste this into the job description box:

```
We are hiring a Senior Machine Learning Engineer.

Requirements:
- 5+ years of Python and ML experience
- Deep expertise in TensorFlow or PyTorch
- Strong NLP background (transformers, BERT, LLMs)
- Experience with AWS, Docker, and Kubernetes
- SQL proficiency and data analysis skills
- MLOps and model deployment experience
- Apache Spark or Kafka knowledge is a plus
- Strong communication and leadership skills
```

### Testing with your own resumes

- Click **"Browse files"** and upload `.pdf` or `.txt` resume files
- Uncheck "Use built-in demo resumes" if you don't want them included
- Click **🚀 Rank Candidates**

---

## 🔢 Scoring Formula

```
Final Score = 0.7 × Semantic Similarity
            + 0.3 × Skill Match Score

Skill Match Score = matched_skills / required_skills_in_JD
```

You can adjust the weights using the **sidebar slider** in the app.

| Signal | What it measures |
|---|---|
| **Semantic Similarity** | How closely the resume *meaning* matches the JD, using cosine distance between sentence embeddings |
| **Skill Match Score** | What fraction of required skills explicitly appear in the resume |

---

## 🧠 ML Concepts Demonstrated

| Concept | Where Used |
|---|---|
| **NLP Text Preprocessing** | `resume_parser.py` — raw text extraction |
| **Vocabulary / Pattern Matching** | `skill_extractor.py` — regex-based skill detection |
| **Dense Sentence Embeddings** | `ranking.py` — SentenceTransformer encoding |
| **Vector Similarity** | `ranking.py` — cosine similarity via scikit-learn |
| **Weighted Score Fusion** | `ranking.py` — combining two signals |
| **Information Retrieval** | Core concept: ranking documents by relevance to a query |

---

## 📦 Dependencies

```
PyPDF2==3.0.1               # PDF text extraction
sentence-transformers==2.7.0 # Pre-trained NLP embeddings
scikit-learn==1.4.2          # Cosine similarity
pandas==2.2.2                # Data handling
streamlit==1.35.0            # Web UI
torch==2.3.0                 # Deep learning backend
transformers==4.41.2         # Hugging Face model loading
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Slow first run | Downloading the model (~90 MB) — wait a moment |
| PDF text is empty | Some PDFs are image-based; use `.txt` files instead or use OCR |
| Port already in use | `streamlit run app.py --server.port 8502` |

---

## 🔮 Possible Extensions

- [ ] Add OCR support for scanned PDFs (pytesseract)
- [ ] Support DOCX files (python-docx)
- [ ] Store results in a database (SQLite / PostgreSQL)
- [ ] Add a recruiter login + candidate shortlisting feature
- [ ] Fine-tune the embedding model on domain-specific resumes
- [ ] Integrate with an ATS (Applicant Tracking System) via REST API

---

## 📄 License

MIT License — free to use, modify, and distribute.
