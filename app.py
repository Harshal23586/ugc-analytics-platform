# app.py
"""
Option C - SIH 2025 Final: All-in-one Streamlit app for AI-based institutional approval analysis.

Features:
- Generates dummy 10-year data for 20 institutions using Appendix-1 framework (Input, Process, Outcome, Impact)
- Produces mandatory/supporting document checklist per institution-year; computes sufficiency %
- Composite scoring, risk level, approval recommendations
- Streamlit UI: Dashboard, Document Sufficiency heatmaps, RAG-style analyzer (optional embeddings), Institution portal, Reports & downloads
- Graceful handling of optional heavy libs: sentence-transformers, PyPDF2, python-docx, reportlab
- Safe writable storage using tempfile (suitable for Streamlit Cloud)
- Suppresses noisy warnings from torch where possible
- Modular functions; caching for heavy ops
"""

# Standard libs
import os
import io
import sys
import json
import re
import math
import time
import random
import string
import hashlib
import tempfile
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Data + plotting
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Persistence and docs
import sqlite3

# Streamlit
import streamlit as st

# Try to import optional heavy dependencies and handle gracefully
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SENTE = True
except Exception:
    SentenceTransformer = None
    cosine_similarity = None
    HAS_SENTE = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    PyPDF2 = None
    HAS_PYPDF2 = False

try:
    import docx
    HAS_DOCX = True
except Exception:
    docx = None
    HAS_DOCX = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# Silence some noisy warnings (Torch, future warnings)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# CONFIG - Use writable temp directory (safe for Streamlit Cloud)
# ---------------------------

APP_LABEL = "AI Institutional Approval - SIH 2025"
TMP_BASE = tempfile.gettempdir()  # safe writable path
APP_SUBDIR = "sih_ugc_app_data"
DATA_DIR = os.path.join(TMP_BASE, APP_SUBDIR)
os.makedirs(DATA_DIR, exist_ok=True)

# File paths
CSV_FULL = os.path.join(DATA_DIR, "institutions_10yrs_20inst.csv")
CSV_DOCS = os.path.join(DATA_DIR, "institution_documents_10yrs_20inst.csv")
CSV_SUM = os.path.join(DATA_DIR, "institutions_summary.csv")
DB_PATH = os.path.join(DATA_DIR, "institutions.db")
LOG_PATH = os.path.join(DATA_DIR, "app.log")

# Embedding model (optional)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(page_title=APP_LABEL, layout="wide", page_icon="🏛️", initial_sidebar_state="expanded")

# ---------------------------
# Utilities
# ---------------------------

def log(msg: str):
    ts = datetime.utcnow().isoformat()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{ts} {msg}\n")
    except Exception:
        pass

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            log(f"safe_read_csv error reading {path}: {e}")
            return None
    return None

def save_df_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    return path

def humanize(n: float, precision: int = 2) -> str:
    return f"{n:.{precision}f}"

# ---------------------------
# Data Generation - Appendix-1 guided (10 years × 20 institutes)
# ---------------------------

def generate_dummy_dataset(force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates or loads synthetic dataset:
    - df: time-series (20 inst × 10 years) with many metrics and composite score
    - df_docs: document checklist (per inst-year, mandatory/supporting)
    - summary: aggregated averages per institution
    """

    # Load if exists and not forcing
    if not force and os.path.exists(CSV_FULL) and os.path.exists(CSV_DOCS) and os.path.exists(CSV_SUM):
        df = pd.read_csv(CSV_FULL)
        df_docs = pd.read_csv(CSV_DOCS)
        summary = pd.read_csv(CSV_SUM)
        return df, df_docs, summary

    random.seed(42)
    np.random.seed(42)

    # Create 20 institutions mixing categories
    inst_names = []
    for i in range(5):
        inst_names.append(f"IIT Example {i+1}")
    for i in range(5):
        inst_names.append(f"State Univ {i+1}")
    for i in range(5):
        inst_names.append(f"Private Univ {i+1}")
    for i in range(5):
        inst_names.append(f"Specialised Inst {i+1}")

    categories = [
        "Multi-disciplinary Education and Research-Intensive",
        "Research-Intensive",
        "Teaching-Intensive",
        "Specialised Streams",
        "Vocational and Skill-Intensive",
        "Community Engagement & Service",
        "Rural & Remote location"
    ]
    heritage_types = ["Old and Established", "New and Upcoming"]

    institution_templates = []
    for idx, name in enumerate(inst_names):
        inst_id = f"HEI_{idx+1:02d}"
        inst_type = random.choice(categories)
        heritage = random.choice(heritage_types)
        institution_templates.append((inst_id, name, inst_type, heritage))

    # 10 years: 2016-2025 inclusive (10 years)
    years = list(range(2016, 2016 + 10))

    mandatory_docs = [
        "Institution Profile (SSR)", "Faculty Roster", "Program Curriculum Document",
        "Accreditation/Approval Certificates", "Audited Financial Statements (last 3 years)",
        "Student Enrollment Data", "Examination Results Summary", "Library & Lab Inventory"
    ]
    supporting_docs = [
        "Research Publications List", "Patents & IPR filings", "Alumni Placement Reports",
        "External Collaboration MOUs", "Community Engagement Reports", "Student Feedback Surveys",
        "Teacher Development Records", "Annual Reports"
    ]

    def generate_metrics(inst_type: str, heritage: str, year: int) -> Dict[str, Any]:
        base = {}
        # curriculum
        base["curriculum_relevance"] = float(np.clip(np.random.normal(7.0, 1.0), 3.0, 9.5))
        base["curriculum_updates_per_year"] = int(np.clip(np.random.poisson(1 if "Teaching" in inst_type else 2), 0, 6))
        # faculty
        if "Research" in inst_type:
            base["faculty_fte_per_100_students"] = float(np.clip(np.random.normal(18, 4), 6, 40))
            base["faculty_phd_pct"] = float(np.clip(np.random.normal(45, 15), 5, 90))
        else:
            base["faculty_fte_per_100_students"] = float(np.clip(np.random.normal(12, 3), 4, 30))
            base["faculty_phd_pct"] = float(np.clip(np.random.normal(25, 10), 2, 70))
        base["faculty_training_score"] = float(np.clip(np.random.normal(6.5, 1.5), 2.0, 9.5))
        # research
        if "Research" in inst_type or "Multi-disciplinary" in inst_type:
            base["research_publications_count"] = int(np.clip(np.random.poisson(20), 0, 500))
            base["research_citations_per_pub"] = float(np.clip(np.random.normal(4.0, 2.0), 0.0, 50.0))
            base["patents_filed"] = int(np.clip(np.random.poisson(2), 0, 50))
        else:
            base["research_publications_count"] = int(np.clip(np.random.poisson(3), 0, 50))
            base["research_citations_per_pub"] = float(np.clip(np.random.normal(1.0, 0.5), 0.0, 10.0))
            base["patents_filed"] = int(np.clip(np.random.poisson(0.2), 0, 10))
        # infra + governance + financial + outcomes
        base["lab_capacity_index"] = float(np.clip(np.random.normal(6.5, 1.5), 1.0, 10.0))
        base["library_resources_index"] = float(np.clip(np.random.normal(6.0, 1.5), 1.0, 10.0))
        base["digital_resources_index"] = float(np.clip(np.random.normal(6.0, 2.0), 1.0, 10.0))
        base["gov_transparency_score"] = float(np.clip(np.random.normal(6.5, 1.5), 2.0, 9.5))
        base["grievance_resolution_time_days"] = int(np.clip(np.random.normal(30, 20), 1, 365))
        base["autonomy_index"] = float(np.clip(np.random.normal(5.5, 2.0), 1.0, 10.0))
        base["financial_stability_score"] = float(np.clip(np.random.normal(6.5, 1.8), 1.0, 10.0))
        base["research_funding_per_fte"] = float(np.clip(np.random.normal(50000 if "Research" in inst_type else 5000, 20000), 0, 1e6))
        base["infrastructure_spend_pct"] = float(np.clip(np.random.normal(8 if heritage == "Old and Established" else 12, 4), 1, 50))
        base["graduation_rate_pct"] = float(np.clip(np.random.normal(78, 10), 30, 99))
        base["placements_pct"] = float(np.clip(np.random.normal(65 if "Research" in inst_type else 50, 20), 0, 100))
        base["student_satisfaction_score"] = float(np.clip(np.random.normal(7.0, 1.5), 2.0, 9.5))
        # small time trend
        trend = (year - years[0]) * np.random.normal(0.05, 0.1)
        base["curriculum_relevance"] = float(np.clip(base["curriculum_relevance"] + trend, 1, 10))
        base["student_satisfaction_score"] = float(np.clip(base["student_satisfaction_score"] + trend / 2, 1, 10))
        base["research_publications_count"] = int(max(0, base["research_publications_count"] * (1 + trend * 0.1)))
        return base

    rows = []
    doc_rows = []
    for inst_id, inst_name, inst_type, heritage in institution_templates:
        for year in years:
            m = generate_metrics(inst_type, heritage, year)
            # composite weights
            weights = {
                "curriculum_relevance": 0.12, "faculty_fte_per_100_students": 0.08, "faculty_phd_pct": 0.08,
                "research_publications_count": 0.15, "research_citations_per_pub": 0.07, "patents_filed": 0.03,
                "lab_capacity_index": 0.06, "digital_resources_index": 0.05, "library_resources_index": 0.04,
                "gov_transparency_score": 0.05, "financial_stability_score": 0.10, "graduation_rate_pct": 0.10,
                "placements_pct": 0.07
            }
            norm_map = {
                "curriculum_relevance": (1, 10),
                "faculty_fte_per_100_students": (4, 40),
                "faculty_phd_pct": (0, 90),
                "research_publications_count": (0, 200),
                "research_citations_per_pub": (0, 10),
                "patents_filed": (0, 20),
                "lab_capacity_index": (1, 10),
                "digital_resources_index": (1, 10),
                "library_resources_index": (1, 10),
                "gov_transparency_score": (1, 10),
                "financial_stability_score": (1, 10),
                "graduation_rate_pct": (30, 100),
                "placements_pct": (0, 100)
            }
            def norm(v, a, b):
                if b <= a:
                    return 0.0
                return (v - a) / (b - a)
            comp = 0.0
            for k, w in weights.items():
                v = m.get(k, 0)
                a, b = norm_map.get(k, (0, 1))
                comp += norm(v, a, b) * w
            composite_score = round(comp * 100, 2)

            # map to risk & recommendation
            if composite_score >= 75:
                risk = "Low"
                recommendation = "Full Approval - 5 Years"
            elif composite_score >= 60:
                risk = "Medium"
                recommendation = "Provisional Approval - 3 Years"
            elif composite_score >= 45:
                risk = "High"
                recommendation = "Conditional Approval - 1 Year"
            else:
                risk = "Critical"
                recommendation = "Rejection / Requires Major Improvement"

            # Document submission probabilities based on composite (higher -> more likely)
            doc_prob_base = composite_score / 120.0  # ~0 - 0.9
            mand_present = 0
            for d in mandatory_docs:
                submitted = np.random.rand() < min(0.95, max(0.2, doc_prob_base + np.random.normal(0, 0.1)))
                if submitted:
                    mand_present += 1
                doc_rows.append({
                    "institution_id": inst_id, "year": year, "document_name": d, "category": "mandatory", "submitted": bool(submitted)
                })
            supp_present = 0
            for d in supporting_docs:
                submitted = np.random.rand() < min(0.9, max(0.05, doc_prob_base - 0.1 + np.random.normal(0, 0.18)))
                if submitted:
                    supp_present += 1
                doc_rows.append({
                    "institution_id": inst_id, "year": year, "document_name": d, "category": "supporting", "submitted": bool(submitted)
                })
            total_mand = len(mandatory_docs)
            total_supp = len(supporting_docs)
            mand_pct = round((mand_present / total_mand) * 100, 2)
            overall_pct = round(((mand_present + supp_present) / (total_mand + total_supp)) * 100, 2)

            row = {
                "institution_id": inst_id,
                "institution_name": inst_name,
                "year": year,
                "institution_type": inst_type,
                "heritage_category": heritage,
                **m,
                "composite_score": composite_score,
                "risk_level": risk,
                "approval_recommendation": recommendation,
                "mandatory_documents_present": mand_present,
                "total_mandatory_documents": total_mand,
                "supporting_documents_present": supp_present,
                "total_supporting_documents": total_supp,
                "mandatory_sufficiency_pct": mand_pct,
                "overall_document_sufficiency_pct": overall_pct
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df_docs = pd.DataFrame(doc_rows)
    summary = df.groupby(["institution_id", "institution_name", "institution_type", "heritage_category"]).agg({
        "composite_score": "mean",
        "mandatory_sufficiency_pct": "mean",
        "overall_document_sufficiency_pct": "mean",
        "research_publications_count": "mean",
        "placements_pct": "mean",
        "graduation_rate_pct": "mean"
    }).reset_index()
    summary = summary.round(2)

    # Save CSVs
    save_df_csv(df, CSV_FULL)
    save_df_csv(df_docs, CSV_DOCS)
    save_df_csv(summary, CSV_SUM)

    log("Generated dataset and saved to temp folder.")
    return df, df_docs, summary

# ---------------------------
# Simple Vector Store / RAG (optional, robust)
# ---------------------------

class SimpleRAG:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.texts: List[str] = []
        self.embeddings = None
        if HAS_SENTE:
            try:
                # load lazily when needed
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                log(f"SentenceTransformer load failed: {e}")
                self.model = None

    def extract_text_from_file(self, uploaded_file) -> str:
        name = uploaded_file.name.lower()
        try:
            if name.endswith(".pdf") and HAS_PYPDF2:
                reader = PyPDF2.PdfReader(uploaded_file)
                pages = [p.extract_text() or "" for p in reader.pages]
                return "\n".join(pages)
            elif (name.endswith(".docx") or name.endswith(".doc")) and HAS_DOCX:
                doc = docx.Document(uploaded_file)
                return "\n".join([p.text for p in doc.paragraphs])
            else:
                raw = uploaded_file.getvalue()
                try:
                    return raw.decode("utf-8")
                except Exception:
                    return raw.decode("latin-1", errors="ignore")
        except Exception as e:
            log(f"extract_text_from_file error for {name}: {e}")
            return ""

    def prepare(self, files: List[Any], chunk_size: int = 800):
        self.texts = []
        for f in files:
            t = self.extract_text_from_file(f)
            if not t:
                continue
            # lightweight chunking by sentences
            chunks = re.split(r'(?<=[\.\?\!])\s+', t)
            cur = ""
            for s in chunks:
                if len(cur) + len(s) + 1 <= chunk_size:
                    cur = (cur + " " + s).strip() if cur else s.strip()
                else:
                    if cur:
                        self.texts.append(cur.strip())
                    cur = s.strip()
            if cur:
                self.texts.append(cur.strip())
        # build embeddings (if available)
        if self.model and len(self.texts) > 0:
            try:
                self.embeddings = self.model.encode(self.texts, show_progress_bar=False)
            except Exception as e:
                log(f"Embedding error: {e}")
                self.embeddings = None

    def query(self, q: str, topk: int = 5) -> List[Dict[str, Any]]:
        if self.model is None or self.embeddings is None or len(self.texts) == 0:
            return []
        try:
            qemb = self.model.encode([q])
            sims = cosine_similarity(qemb, self.embeddings)[0]
            idxs = np.argsort(sims)[-topk:][::-1]
            return [{"text": self.texts[i], "score": float(sims[i])} for i in idxs if sims[i] > 0]
        except Exception as e:
            log(f"RAG query error: {e}")
            return []

# ---------------------------
# Report generation helpers
# ---------------------------

def generate_html_report(inst_df: pd.DataFrame, inst_docs: pd.DataFrame, inst_meta: Dict[str, Any]) -> str:
    title = f"Report - {inst_meta.get('institution_name')} ({inst_meta.get('institution_id')})"
    html = [
        "<!doctype html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;padding:20px;}table{border-collapse:collapse;}td,th{border:1px solid #ddd;padding:8px;}th{background:#f0f0f0;}</style>",
        "</head><body>"
    ]
    html.append(f"<h1>{title}</h1>")
    html.append(f"<p><b>Type:</b> {inst_meta.get('institution_type')} &nbsp; | &nbsp; <b>Heritage:</b> {inst_meta.get('heritage_category')}</p>")
    html.append("<h2>Recent Performance (last 3 years)</h2>")
    last = inst_df.sort_values("year", ascending=False).head(3)
    html.append("<table><tr><th>Year</th><th>Composite Score</th><th>Risk Level</th><th>Approval Recommendation</th><th>Mand. Suff.%</th><th>Overall Suff.%</th></tr>")
    for _, r in last.iterrows():
        html.append(f"<tr><td>{int(r['year'])}</td><td>{r['composite_score']}</td><td>{r['risk_level']}</td><td>{r['approval_recommendation']}</td><td>{r['mandatory_sufficiency_pct']}</td><td>{r['overall_document_sufficiency_pct']}</td></tr>")
    html.append("</table>")
    html.append("<h2>Average Summary</h2><ul>")
    avg = inst_df.mean(numeric_only=True).to_dict()
    for k in ["research_publications_count", "placements_pct", "graduation_rate_pct", "financial_stability_score", "student_satisfaction_score"]:
        html.append(f"<li><b>{k}</b>: {avg.get(k, 0):.2f}</li>")
    html.append("</ul>")
    html.append("<h2>Document Checklist (latest year)</h2>")
    if not inst_docs.empty:
        latest_year = int(inst_docs['year'].max())
        html.append(f"<p><b>Year:</b> {latest_year}</p>")
        ddf = inst_docs[inst_docs['year'] == latest_year]
        html.append("<table><tr><th>Document</th><th>Category</th><th>Submitted</th></tr>")
        for _, d in ddf.iterrows():
            html.append(f"<tr><td>{d['document_name']}</td><td>{d['category']}</td><td>{'Yes' if d['submitted'] else 'No'}</td></tr>")
        html.append("</table>")
    else:
        html.append("<p>No document records found.</p>")
    html.append("</body></html>")
    return "\n".join(html)

def generate_pdf_snapshot(inst_df: pd.DataFrame, inst_meta: Dict[str,Any]) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 10)
    text.textLine(f"Report: {inst_meta.get('institution_name')} ({inst_meta.get('institution_id')})")
    text.textLine(f"Generated: {datetime.utcnow().isoformat()} UTC")
    text.textLine("")
    text.textLine("Recent Performance (last 3 years):")
    last = inst_df.sort_values("year", ascending=False).head(3)
    for _, r in last.iterrows():
        text.textLine(f"Year {int(r['year'])} | Composite: {r['composite_score']} | Risk: {r['risk_level']} | Approval: {r['approval_recommendation']}")
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

# ---------------------------
# Lightweight DB utilities (store only metadata)
# ---------------------------

@st.cache_resource
def get_db_conn(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(conn: sqlite3.Connection):
    try:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS institution_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                institution_id TEXT,
                year INTEGER,
                document_name TEXT,
                document_type TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'Uploaded'
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS rag_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                institution_id TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                extracted_data TEXT,
                ai_insights TEXT,
                confidence REAL
            )
        ''')
        conn.commit()
    except Exception as e:
        log(f"init_db error: {e}")

# ---------------------------
# Streamlit UI components
# ---------------------------

def sidebar_user_login(summary: pd.DataFrame):
    st.sidebar.header("User / Institution")
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    if st.session_state['user'] is None:
        # Simple login: institution_id as username (for demo)
        mode = st.sidebar.selectbox("Mode", ["Guest", "Institution User"])
        if mode == "Institution User":
            username = st.sidebar.text_input("Institution ID (e.g., HEI_01)")
            pwd = st.sidebar.text_input("Password (demo)", type="password")
            if st.sidebar.button("Login"):
                if username and username in summary['institution_id'].values:
                    # In real app do proper auth; for demo accept any matching inst id
                    sel = summary[summary['institution_id'] == username].iloc[0].to_dict()
                    st.session_state['user'] = {"institution_id": username, "institution_name": sel['institution_name']}
                    st.sidebar.success(f"Logged in: {sel['institution_name']}")
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Institution ID not found. Use Guest or choose a valid HEI_XX.")
        else:
            st.sidebar.info("You are browsing as Guest.")
    else:
        st.sidebar.success(f"Signed in: {st.session_state['user']['institution_name']}")
        if st.sidebar.button("Logout"):
            st.session_state['user'] = None
            st.experimental_rerun()

def main_dashboard(df: pd.DataFrame, summary: pd.DataFrame):
    st.header("📊 System Dashboard — Overall View")
    cur_year = st.selectbox("Select year for KPI snapshot", sorted(df['year'].unique(), reverse=True), index=0)
    df_year = df[df['year'] == cur_year]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Institutions", df_year['institution_id'].nunique())
    with c2:
        st.metric("Average Composite Score", f"{df_year['composite_score'].mean():.2f}/100")
    with c3:
        st.metric("Avg Mandatory Suff %", f"{df_year['mandatory_sufficiency_pct'].mean():.1f}%")
    with c4:
        st.metric("Avg Overall Doc Suff %", f"{df_year['overall_document_sufficiency_pct'].mean():.1f}%")

    st.markdown("---")
    st.subheader("Score Distribution")
    fig = px.histogram(df_year, x="composite_score", nbins=20, title=f"Composite Score Distribution ({cur_year})", labels={"composite_score":"Composite Score"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Publications vs Placements (bubble size = composite score)")
    fig2 = px.scatter(df_year, x="research_publications_count", y="placements_pct", size="composite_score",
                      color="institution_type", hover_data=["institution_id","institution_name"], title="Publications vs Placements")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Top & Bottom Institutions (selected year)")
    left, right = st.columns(2)
    with left:
        st.write("Top 5 by composite score")
        st.dataframe(df_year.nlargest(5, "composite_score")[["institution_id","institution_name","composite_score","approval_recommendation"]].reset_index(drop=True))
    with right:
        st.write("Bottom 5 by composite score")
        st.dataframe(df_year.nsmallest(5, "composite_score")[["institution_id","institution_name","composite_score","approval_recommendation"]].reset_index(drop=True))

def document_sufficiency_ui(df: pd.DataFrame, df_docs: pd.DataFrame, summary: pd.DataFrame):
    st.header("📂 Document Sufficiency & Checklist")
    view_mode = st.radio("View mode", ["Average (all years)", "Specific Year"])
    if view_mode == "Average (all years)":
        heat = summary.set_index("institution_id")[["mandatory_sufficiency_pct","overall_document_sufficiency_pct","composite_score"]]
        # Create a heatmap for mandatory sufficiency %
        fig = px.imshow(heat[["mandatory_sufficiency_pct"]].T, labels=dict(x="Institution", y="Metric"), x=heat.index, y=["Mand. Suff %"], text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        sel_year = st.selectbox("Year", sorted(df['year'].unique(), reverse=True))
        tmp = df[df['year'] == sel_year].set_index("institution_id")[["mandatory_sufficiency_pct","overall_document_sufficiency_pct","composite_score"]]
        fig = px.imshow(tmp[["mandatory_sufficiency_pct"]].T, labels=dict(x="Institution", y="Metric"), x=tmp.index, y=["Mand. Suff %"], text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Per-Institution Document Checklist")
    sel_inst = st.selectbox("Select Institution", df['institution_id'].unique(), key="doc_inst_select")
    sel_year = st.selectbox("Select Year for checklist", sorted(df['year'].unique(), reverse=True), key="doc_year_select")
    subset = df_docs[(df_docs['institution_id'] == sel_inst) & (df_docs['year'] == sel_year)]
    if subset.empty:
        st.info("No document records found for this selection.")
    else:
        st.dataframe(subset[['document_name','category','submitted']].sort_values(['category','document_name']), width=900)

def rag_analyzer_ui():
    st.header("🤖 Document Analyzer (RAG-lite)")
    st.markdown("Upload documents (PDF/DOCX/TXT/XLSX). If embeddings are available, you can run similarity search.")
    rag = SimpleRAG()  # will try to load sentence-transformers if available
    uploaded = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf','docx','txt','xlsx'])
    if uploaded:
        if st.button("Extract & (optionally) build embeddings"):
            with st.spinner("Preparing / extracting text and embeddings (if available)..."):
                rag.prepare(uploaded)
                st.success(f"Prepared {len(rag.texts)} text chunks.")
                if rag.embeddings is not None:
                    st.success("Embeddings built.")
                else:
                    st.info("Embeddings not available (model missing or build failed). You can still search by regex.")
        st.markdown("#### Previews")
        for i, f in enumerate(uploaded):
            txt = rag.extract_text_from_file(f)
            if not txt:
                st.write(f"**{f.name}** - (no extracted text)")
            else:
                preview = txt[:2000] + ("..." if len(txt) > 2000 else "")
                st.write(f"**{f.name}** preview:")
                st.text_area(f"preview_{i}", value=preview, height=150)
        st.markdown("#### Query uploaded documents")
        q = st.text_input("Enter a natural language query")
        topk = st.slider("Top K", min_value=1, max_value=10, value=5)
        if st.button("Query"):
            if rag.embeddings is not None and HAS_SENTE:
                results = rag.query(q, topk)
                if not results:
                    st.info("No matches found.")
                for r in results:
                    st.write(f"Score: {r['score']:.4f}")
                    st.write(r['text'][:1000])
                    st.markdown("---")
            else:
                st.info("Embeddings not available. Performing regex substring search across uploaded files.")
                all_text = " ".join([rag.extract_text_from_file(f) for f in uploaded])
                if not q.strip():
                    st.warning("Enter a query to search.")
                else:
                    matches = re.findall(r'.{0,120}' + re.escape(q) + r'.{0,120}', all_text, flags=re.IGNORECASE)
                    if not matches:
                        st.info("No matches found.")
                    else:
                        for m in matches[:topk]:
                            st.write(m)

def institution_portal_ui(df: pd.DataFrame, df_docs: pd.DataFrame):
    st.header("🏫 Institution Portal")
    user = st.session_state.get('user')
    if user is None:
        st.info("Login as an Institution (sidebar) to use this portal. Use 'Institution User' in the sidebar.")
        return
    inst_id = user['institution_id']
    st.subheader(f"Welcome {user.get('institution_name')} ({inst_id})")
    st.markdown("Upload documents — metadata will be stored in a lightweight DB (metadata only). Actual file upload is not persisted by this demo for security.")
    uploaded = st.file_uploader("Upload documents (metadata saved)", accept_multiple_files=True, type=['pdf','docx','txt','xlsx'])
    year = st.number_input("Document Year", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=int(df['year'].max()))
    dtype = st.text_input("Document type / tag", value="other")
    if uploaded and st.button("Save metadata to DB"):
        conn = get_db_conn()
        init_db(conn)
        cur = conn.cursor()
        for f in uploaded:
            try:
                cur.execute("INSERT INTO institution_documents (institution_id, year, document_name, document_type) VALUES (?, ?, ?, ?)",
                            (inst_id, int(year), f.name, dtype))
            except Exception as e:
                log(f"DB insert error: {e}")
        conn.commit()
        st.success("Document metadata saved (DB).")
    # show saved metadata
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM institution_documents WHERE institution_id = ? ORDER BY uploaded_at DESC", (inst_id,))
    rows = cur.fetchall()
    if rows:
        df_saved = pd.DataFrame([dict(r) for r in rows])
        st.dataframe(df_saved[['year','document_name','document_type','uploaded_at']])
    else:
        st.info("No saved document metadata for your institution.")

def reports_ui(df: pd.DataFrame, df_docs: pd.DataFrame, summary: pd.DataFrame):
    st.header("📥 Reports & Downloads")
    st.markdown("Download full CSVs or per-institution reports. HTML reports include performance summary and document checklist.")
    c1, c2, c3 = st.columns(3)
    with c1:
        if os.path.exists(CSV_FULL):
            with open(CSV_FULL, "rb") as f:
                st.download_button("Download full time-series CSV", data=f.read(), file_name="institutions_10yrs_20inst.csv", mime="text/csv")
        else:
            st.info("Full CSV not found (generate dataset).")
    with c2:
        if os.path.exists(CSV_DOCS):
            with open(CSV_DOCS, "rb") as f:
                st.download_button("Download documents CSV", data=f.read(), file_name="institution_documents_10yrs_20inst.csv", mime="text/csv")
        else:
            st.info("Documents CSV not found.")
    with c3:
        if os.path.exists(CSV_SUM):
            with open(CSV_SUM, "rb") as f:
                st.download_button("Download summary CSV", data=f.read(), file_name="institutions_summary.csv", mime="text/csv")
        else:
            st.info("Summary CSV not found.")

    st.markdown("---")
    st.subheader("Per-institution HTML/PDF report")
    sel_inst = st.selectbox("Select institution for report", df['institution_id'].unique())
    inst_df = df[df['institution_id'] == sel_inst].sort_values("year", ascending=False)
    inst_docs = df_docs[df_docs['institution_id'] == sel_inst].sort_values(['year','category'])
    inst_meta = {
        "institution_id": sel_inst,
        "institution_name": inst_df.iloc[0]['institution_name'],
        "institution_type": inst_df.iloc[0]['institution_type'],
        "heritage_category": inst_df.iloc[0]['heritage_category']
    }
    if st.button("Generate HTML report"):
        html = generate_html_report(inst_df, inst_docs, inst_meta)
        b = html.encode("utf-8")
        st.download_button("Download HTML report", data=b, file_name=f"{sel_inst}_report.html", mime="text/html")
        st.success("HTML report generated.")
    if HAS_REPORTLAB:
        if st.button("Generate PDF snapshot"):
            pdf_bytes = generate_pdf_snapshot(inst_df, inst_meta)
            if pdf_bytes:
                st.download_button("Download PDF snapshot", data=pdf_bytes, file_name=f"{sel_inst}_snapshot.pdf", mime="application/pdf")
            else:
                st.error("PDF generation failed.")
    else:
        st.info("PDF snapshot requires reportlab; not available in this environment.")

def system_ui(summary: pd.DataFrame):
    st.header("⚙️ System & Diagnostics")
    st.markdown("Environment & availability of optional components.")
    st.write("SentenceTransformers available:", HAS_SENTE)
    st.write("PyPDF2 available:", HAS_PYPDF2)
    st.write("python-docx available:", HAS_DOCX)
    st.write("reportlab available:", HAS_REPORTLAB)
    st.write("Temporary data directory:", DATA_DIR)
    st.write("Log file:", LOG_PATH)
    if st.button("Open log (last 200 lines)"):
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()[-200:]
            st.text("".join(lines))
        else:
            st.info("No log file yet.")
    if st.button("Force regenerate dataset (overwrite)"):
        df, df_docs, summary = generate_dummy_dataset(force=True)
        st.success("Dataset regenerated (force). You can re-open other tabs to view data.")

# ---------------------------
# Main app
# ---------------------------

def main():
    st.title("🏛️ AI Institutional Approval — SIH 2025 Final")
    st.markdown("""
        This is the SIH 2025 final demo app (Option C).
        It generates synthetic institution data (10 years × 20 institutes),
        computes composite performance scores, document sufficiency,
        and provides a lightweight RAG-style document analyzer and reporting.
    """)

    # Data generation / load area (collapsible)
    with st.expander("Dataset generation & load (Appendix-1 framework)"):
        st.write("Generate or load the synthetic dataset. Files are stored in a writable temp folder (suitable for Streamlit Cloud).")
        col1, col2, col3 = st.columns([1,1,2])
        if col1.button("Generate dataset (force)"):
            df, df_docs, summary = generate_dummy_dataset(force=True)
            st.success("Dataset generated (force).")
        if col2.button("Load dataset (if exists)"):
            df, df_docs, summary = generate_dummy_dataset(force=False)
            st.success("Dataset loaded (if present).")
        col3.markdown(f"Files are stored in: `{DATA_DIR}`. CSVs: `institutions_10yrs_20inst.csv`, `institution_documents_10yrs_20inst.csv`, `institutions_summary.csv`.")

    # ensure dataset loaded for the app
    df, df_docs, summary = generate_dummy_dataset(force=False)

    # initialize DB
    conn = get_db_conn()
    init_db(conn)

    # sidebar user
    sidebar_user_login(summary)

    # top-level tabs
    tabs = st.tabs(["Dashboard","Document Sufficiency","RAG Analyzer","Institution Portal","Reports & Downloads","System"])
    with tabs[0]:
        main_dashboard(df, summary)
    with tabs[1]:
        document_sufficiency_ui(df, df_docs, summary)
    with tabs[2]:
        rag_analyzer_ui()
    with tabs[3]:
        institution_portal_ui(df, df_docs)
    with tabs[4]:
        reports_ui(df, df_docs, summary)
    with tabs[5]:
        system_ui(summary)

    # bottom: credits & notes
    st.markdown("---")
    st.markdown("**Notes for Smart India Hackathon 2025 Demo**")
    st.markdown("""
    - This app is a self-contained demonstration with simulated data aligned to Appendix-1 (Input, Process, Outcome, Impact).
    - For production: integrate secure authentication, persistent storage (S3/DB), FAISS/Chroma for vectors, and a scalable model hosting solution.
    - Optional improvements: integrate LLM for narrative insights, automatic document verification pipelines, and role-based access control.
    """)

if __name__ == "__main__":
    main()
