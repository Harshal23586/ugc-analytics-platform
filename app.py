# app.py
"""
Single-file Streamlit application for:
- Generating dummy 10-year dataset for 20 institutions (Appendix-1 framework)
- Document checklist (mandatory/supporting) per institution-year
- Composite scoring, approval recommendation, risk levels
- RAG-style doc extraction (optional embeddings)
- Dashboards, heatmaps, per-institution downloadable reports (HTML/CSV)
"""

# Standard libraries
import os
import io
import json
import re
import hashlib
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

# Data + plotting
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Persistence and docs
import sqlite3

# Streamlit
import streamlit as st

# Optional heavy deps (graceful)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    SentenceTransformer = None
    cosine_similarity = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

# Try reportlab for PDF export (optional)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:
    reportlab = None

# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = "/mnt/data/hackathon_generated"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_FULL = os.path.join(DATA_DIR, "institutions_10yrs_20inst.csv")
CSV_DOCS = os.path.join(DATA_DIR, "institution_documents_10yrs_20inst.csv")
CSV_SUM = os.path.join(DATA_DIR, "institutions_summary.csv")
DB_PATH = os.path.join(DATA_DIR, "institutions.db")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(page_title="AI Institutional Approval - Hackathon 2025", layout="wide", page_icon="🏛️")

# ---------------------------
# Utilities
# ---------------------------
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    return path

def read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ---------------------------
# Data generation (Appendix-1 guided)
# ---------------------------
def generate_dummy_dataset(force=False):
    # If CSV exists and not forcing, load it
    if not force and os.path.exists(CSV_FULL) and os.path.exists(CSV_DOCS) and os.path.exists(CSV_SUM):
        df = pd.read_csv(CSV_FULL)
        df_docs = pd.read_csv(CSV_DOCS)
        summary = pd.read_csv(CSV_SUM)
        return df, df_docs, summary

    random.seed(42)
    np.random.seed(42)

    # Create 20 institutions (mix)
    inst_names = []
    for i in range(5):
        inst_names.append(f"IIT Example {i+1}")
    for i in range(5):
        inst_names.append(f"State Univ {i+1}")
    for i in range(5):
        inst_names.append(f"Private Univ {i+1}")
    for i in range(5):
        inst_names.append(f"Specialised Inst {i+1}")

    institution_templates = []
    categories = [
        "Multi-disciplinary Education and Research-Intensive",
        "Research-Intensive",
        "Teaching-Intensive",
        "Specialised Streams",
        "Vocational and Skill-Intensive",
        "Community Engagement & Service",
        "Rural & Remote location"
    ]
    heritage = ["Old and Established", "New and Upcoming"]

    for idx, name in enumerate(inst_names):
        i_id = f"HEI_{idx+1:02d}"
        inst_type = random.choice(categories)
        heritage_cat = random.choice(heritage)
        institution_templates.append((i_id, name, inst_type, heritage_cat))

    years = list(range(2016, 2016 + 10))

    # Document lists
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

    # metric groups and generator
    def generate_metrics(inst_type: str, heritage_type: str, year: int):
        base = {}
        # curriculum
        base["curriculum_relevance"] = np.clip(np.random.normal(7.0, 1.0), 3.0, 9.5)
        base["curriculum_updates_per_year"] = int(np.clip(np.random.poisson(1 if "Teaching" in inst_type else 2), 0, 6))
        # faculty
        if "Research" in inst_type:
            base["faculty_fte_per_100_students"] = np.clip(np.random.normal(18, 4), 6, 40)
            base["faculty_phd_pct"] = np.clip(np.random.normal(45, 15), 5, 90)
        else:
            base["faculty_fte_per_100_students"] = np.clip(np.random.normal(12, 3), 4, 30)
            base["faculty_phd_pct"] = np.clip(np.random.normal(25, 10), 2, 70)
        base["faculty_training_score"] = np.clip(np.random.normal(6.5, 1.5), 2.0, 9.5)
        # research
        if "Research" in inst_type or "Multi-disciplinary" in inst_type:
            base["research_publications_count"] = int(np.clip(np.random.poisson(20), 0, 500))
            base["research_citations_per_pub"] = np.clip(np.random.normal(4.0, 2.0), 0.0, 50.0)
            base["patents_filed"] = int(np.clip(np.random.poisson(2), 0, 50))
        else:
            base["research_publications_count"] = int(np.clip(np.random.poisson(3), 0, 50))
            base["research_citations_per_pub"] = np.clip(np.random.normal(1.0, 0.5), 0.0, 10.0)
            base["patents_filed"] = int(np.clip(np.random.poisson(0.2), 0, 10))
        base["lab_capacity_index"] = np.clip(np.random.normal(6.5, 1.5), 1.0, 10.0)
        base["library_resources_index"] = np.clip(np.random.normal(6.0, 1.5), 1.0, 10.0)
        base["digital_resources_index"] = np.clip(np.random.normal(6.0, 2.0), 1.0, 10.0)
        base["gov_transparency_score"] = np.clip(np.random.normal(6.5, 1.5), 2.0, 9.5)
        base["grievance_resolution_time_days"] = int(np.clip(np.random.normal(30, 20), 1, 365))
        base["autonomy_index"] = np.clip(np.random.normal(5.5, 2.0), 1.0, 10.0)
        base["financial_stability_score"] = np.clip(np.random.normal(6.5, 1.8), 1.0, 10.0)
        base["research_funding_per_fte"] = np.clip(np.random.normal(50000 if "Research" in inst_type else 5000, 20000), 0, 1e6)
        base["infrastructure_spend_pct"] = np.clip(np.random.normal(8 if heritage_type == "Old and Established" else 12, 4), 1, 50)
        base["graduation_rate_pct"] = np.clip(np.random.normal(78, 10), 30, 99)
        base["placements_pct"] = np.clip(np.random.normal(65 if "Research" in inst_type else 50, 20), 0, 100)
        base["student_satisfaction_score"] = np.clip(np.random.normal(7.0, 1.5), 2.0, 9.5)
        trend = (year - years[0]) * np.random.normal(0.05, 0.1)
        base["curriculum_relevance"] = np.clip(base["curriculum_relevance"] + trend, 1, 10)
        base["student_satisfaction_score"] = np.clip(base["student_satisfaction_score"] + trend / 2, 1, 10)
        base["research_publications_count"] = max(0, int(base["research_publications_count"] * (1 + trend * 0.1)))
        return base

    rows = []
    doc_rows = []
    for inst_id, inst_name, inst_type, heritage_cat in institution_templates:
        for year in years:
            m = generate_metrics(inst_type, heritage_cat, year)
            # weights for composite
            weights = {
                "curriculum_relevance": 0.12, "faculty_fte_per_100_students": 0.08, "faculty_phd_pct": 0.08,
                "research_publications_count": 0.15, "research_citations_per_pub": 0.07, "patents_filed": 0.03,
                "lab_capacity_index": 0.06, "digital_resources_index": 0.05, "library_resources_index": 0.04,
                "gov_transparency_score": 0.05, "financial_stability_score": 0.10, "graduation_rate_pct": 0.10,
                "placements_pct": 0.07
            }
            # normalization map
            nm = {
                "curriculum_relevance": (1,10),
                "faculty_fte_per_100_students": (4,40),
                "faculty_phd_pct": (0,90),
                "research_publications_count": (0,200),
                "research_citations_per_pub": (0,10),
                "patents_filed": (0,20),
                "lab_capacity_index": (1,10),
                "digital_resources_index": (1,10),
                "library_resources_index": (1,10),
                "gov_transparency_score": (1,10),
                "financial_stability_score": (1,10),
                "graduation_rate_pct": (30,100),
                "placements_pct": (0,100)
            }
            def norm(v, a, b): 
                if b<=a: return 0.0
                return float((v - a) / (b - a))
            composite = 0.0
            for k,w in weights.items():
                v = m.get(k, 0)
                a,b = nm.get(k,(0,1))
                composite += norm(v,a,b) * w
            composite_score = round(composite * 100, 2)

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

            # documents probabilistic
            doc_prob_base = composite_score / 120.0
            mand_present = 0
            for d in mandatory_docs:
                submitted = np.random.rand() < min(0.95, max(0.2, doc_prob_base + np.random.normal(0,0.1)))
                mand_present += 1 if submitted else 0
                doc_rows.append({
                    "institution_id": inst_id, "year": year, "document_name": d, "category": "mandatory", "submitted": submitted
                })
            supp_present = 0
            for d in supporting_docs:
                submitted = np.random.rand() < min(0.9, max(0.05, doc_prob_base - 0.1 + np.random.normal(0,0.18)))
                supp_present += 1 if submitted else 0
                doc_rows.append({
                    "institution_id": inst_id, "year": year, "document_name": d, "category": "supporting", "submitted": submitted
                })
            total_mand = len(mandatory_docs)
            total_supp = len(supporting_docs)
            mand_pct = round((mand_present / total_mand) * 100, 2)
            overall_pct = round(((mand_present + supp_present) / (total_mand + total_supp) * 100), 2)
            row = {
                "institution_id": inst_id,
                "institution_name": inst_name,
                "year": year,
                "institution_type": inst_type,
                "heritage_category": heritage_cat,
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
    # summary average per institution
    summary = df.groupby(["institution_id","institution_name","institution_type","heritage_category"]).agg({
        "composite_score":"mean",
        "mandatory_sufficiency_pct":"mean",
        "overall_document_sufficiency_pct":"mean",
        "research_publications_count":"mean",
        "placements_pct":"mean",
        "graduation_rate_pct":"mean"
    }).reset_index()
    summary = summary.round(2)
    save_csv(df, CSV_FULL)
    save_csv(df_docs, CSV_DOCS)
    save_csv(summary, CSV_SUM)
    return df, df_docs, summary

# ---------------------------
# Simple RAG extractor (optional embeddings)
# ---------------------------
class SimpleRAG:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model = None
        self.texts = []
        self.embeddings = None
        self.model_name = model_name
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                st.warning(f"Embedding model load failed: {e}")
                self.model = None

    def extract_text_from_file(self, uploaded_file) -> str:
        name = uploaded_file.name.lower()
        try:
            if name.endswith(".pdf") and PyPDF2:
                reader = PyPDF2.PdfReader(uploaded_file)
                pages = []
                for p in reader.pages:
                    txt = p.extract_text() or ""
                    pages.append(txt)
                return "\n".join(pages)
            elif (name.endswith(".docx") or name.endswith(".doc")) and docx:
                d = docx.Document(uploaded_file)
                return "\n".join([p.text for p in d.paragraphs])
            else:
                content = uploaded_file.getvalue()
                try:
                    return content.decode("utf-8")
                except Exception:
                    return content.decode("latin-1", errors="ignore")
        except Exception as e:
            st.warning(f"File text extraction failed ({uploaded_file.name}): {e}")
            return ""

    def prepare(self, files: List):
        self.texts = []
        for f in files:
            t = self.extract_text_from_file(f)
            if t:
                # chunk lightly
                chunks = re.split(r'(?<=[\.\?\!])\s+', t)
                self.texts.extend([c.strip() for c in chunks if c.strip()])
        # build embeddings
        if self.model and self.texts:
            try:
                self.embeddings = self.model.encode(self.texts, show_progress_bar=False)
            except Exception as e:
                st.warning(f"Embedding failure: {e}")
                self.embeddings = None

    def query(self, q: str, topk: int = 5):
        if self.model is None or self.embeddings is None or len(self.texts)==0:
            return []
        qemb = self.model.encode([q])
        sims = cosine_similarity(qemb, self.embeddings)[0]
        idxs = np.argsort(sims)[-topk:][::-1]
        results = [{"text": self.texts[i], "score": float(sims[i])} for i in idxs if sims[i] > 0]
        return results

# ---------------------------
# PDF / HTML report generation helpers
# ---------------------------
def generate_html_report(inst_df: pd.DataFrame, inst_docs: pd.DataFrame, inst_info: Dict[str,Any]) -> str:
    # produce simple HTML string
    html = []
    html.append(f"<html><head><meta charset='utf-8'><title>Report - {inst_info['institution_name']}</title></head><body>")
    html.append(f"<h1>{inst_info['institution_name']} ({inst_info['institution_id']})</h1>")
    html.append(f"<p><b>Type:</b> {inst_info['institution_type']} | <b>Heritage:</b> {inst_info['heritage_category']}</p>")
    html.append("<h2>Recent Performance (last 3 years)</h2>")
    last = inst_df.sort_values('year', ascending=False).head(3)
    html.append("<table border='1' cellpadding='4'><tr><th>Year</th><th>Composite Score</th><th>Risk Level</th><th>Approval Recommendation</th><th>Mand. Suff.%</th><th>Overall Suff.%</th></tr>")
    for _,r in last.iterrows():
        html.append(f"<tr><td>{int(r['year'])}</td><td>{r['composite_score']}</td><td>{r['risk_level']}</td><td>{r['approval_recommendation']}</td><td>{r['mandatory_sufficiency_pct']}</td><td>{r['overall_document_sufficiency_pct']}</td></tr>")
    html.append("</table>")
    html.append("<h2>Summary Metrics (averaged)</h2>")
    avg = inst_df.mean(numeric_only=True).to_dict()
    html.append("<ul>")
    for k in ["research_publications_count","placements_pct","graduation_rate_pct","financial_stability_score","student_satisfaction_score"]:
        html.append(f"<li><b>{k}:</b> {avg.get(k, 'N/A'):.2f}</li>")
    html.append("</ul>")
    html.append("<h2>Document Checklist (latest year)</h2>")
    latest_year = inst_docs['year'].max()
    ddf = inst_docs[inst_docs['year']==latest_year]
    html.append(f"<p><b>Year:</b> {int(latest_year)}</p>")
    html.append("<table border='1' cellpadding='4'><tr><th>Document</th><th>Category</th><th>Submitted</th></tr>")
    for _,d in ddf.iterrows():
        html.append(f"<tr><td>{d['document_name']}</td><td>{d['category']}</td><td>{'Yes' if d['submitted'] else 'No'}</td></tr>")
    html.append("</table>")
    html.append("</body></html>")
    return "\n".join(html)

def html_to_bytes_dl(name: str, html_str: str):
    b = html_str.encode('utf-8')
    return b

# ---------------------------
# Database helpers (lightweight)
# ---------------------------
@st.cache_resource
def get_db_conn(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(conn: sqlite3.Connection):
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

# ---------------------------
# Build app UI
# ---------------------------
def main():
    st.title("🏛️ AI-based Institutional Approval — Hackathon 2025")
    st.markdown("""
    **Goal:** Generate AI analysis and performance indicators for higher education institutions 
    (10 years × 20 institutions), produce approval recommendations, compute document sufficiency, 
    provide RAG-style doc analysis, downloadable reports, and visual dashboards.
    """)

    # Generate or load dataset
    with st.expander("Dataset generation / load"):
        st.write("Generate or load the synthetic dataset (10 years × 20 institutions).")
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            if st.button("Generate dataset (force)"):
                df, df_docs, summary = generate_dummy_dataset(force=True)
                st.success("Dataset generated and saved.")
        with col2:
            if st.button("Load dataset (if exists)"):
                df, df_docs, summary = generate_dummy_dataset(force=False)
                st.success("Dataset loaded.")
        with col3:
            st.write("Files are saved to `/mnt/data/hackathon_generated` by default.")
        # ensure loaded for UI
    # ensure dataset available
    df, df_docs, summary = generate_dummy_dataset(force=False)

    # Initialize DB
    conn = get_db_conn()
    init_db(conn)

    # Sidebar: login/register (simple institution user concept)
    st.sidebar.header("User")
    if 'user' not in st.session_state:
        st.session_state['user'] = None

    if st.session_state['user'] is None:
        mode = st.sidebar.selectbox("Mode", ["Guest", "Institution User"])
        if mode == "Institution User":
            username = st.sidebar.text_input("Username")
            pwd = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login"):
                # simple mocked authentication: any username that matches institution id works (password ignored)
                possible = summary[summary['institution_id']==username]
                if not possible.empty:
                    st.session_state['user'] = {"institution_id": username, "institution_name": possible.iloc[0]['institution_name']}
                    st.sidebar.success("Logged in.")
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Username not found. Use 'Guest' or register in DB (not implemented).")
        else:
            st.sidebar.info("Browsing as Guest.")
    else:
        st.sidebar.success(f"Logged in: {st.session_state['user']['institution_name']}")
        if st.sidebar.button("Logout"):
            st.session_state['user'] = None
            st.experimental_rerun()

    # Top tabs
    tabs = st.tabs(["Dashboard", "Document Sufficiency", "RAG Analyzer", "Institution Portal", "Reports & Downloads", "System"])
    # ---- Dashboard
    with tabs[0]:
        st.header("📊 System Dashboard")
        # KPIs
        cur_year = st.selectbox("Select Year", sorted(df['year'].unique(), reverse=True), index=0)
        df_year = df[df['year']==cur_year]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Institutions", df_year['institution_id'].nunique())
        with col2:
            st.metric("Avg Composite Score", f"{df_year['composite_score'].mean():.2f}/100")
        with col3:
            st.metric("Avg Mandatory Suff %", f"{df_year['mandatory_sufficiency_pct'].mean():.1f}%")
        with col4:
            st.metric("Avg Overall Doc Suff %", f"{df_year['overall_document_sufficiency_pct'].mean():.1f}%")
        st.markdown("---")
        # histogram of composite scores
        fig_hist = px.histogram(df_year, x="composite_score", nbins=20, title=f"Composite Score Distribution ({cur_year})")
        st.plotly_chart(fig_hist, use_container_width=True)
        # scatter matrix - research vs placements
        fig_scatter = px.scatter(df_year, x="research_publications_count", y="placements_pct", color="institution_type",
                                 size="composite_score", hover_data=["institution_id","institution_name"], title="Publications vs Placements")
        st.plotly_chart(fig_scatter, use_container_width=True)
        # trend line for selected institution
        sel_inst = st.selectbox("Select Institution to show trend", df['institution_id'].unique())
        inst_ts = df[df['institution_id']==sel_inst].sort_values('year')
        fig_trend = px.line(inst_ts, x='year', y='composite_score', title=f"Composite Score Trend — {sel_inst}", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

    # ---- Document Sufficiency
    with tabs[1]:
        st.header("📂 Document Sufficiency & Checklist")
        st.markdown("Heatmap of document sufficiency (mandatory %) per institution (averaged across years or selected year).")
        agg_by_inst = summary.copy()
        year_or_avg = st.radio("View", ["Average (all years)", "Specific Year"])
        if year_or_avg == "Average (all years)":
            heat_df = summary[['institution_id','composite_score','mandatory_sufficiency_pct']].set_index('institution_id')
            # heatmap
            fig = px.imshow(heat_df[['mandatory_sufficiency_pct']].T, labels=dict(x="Institution", y="Metric"), x=heat_df.index, y=["Mand. Suff. %"], aspect="auto", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            sel_year = st.selectbox("Year", sorted(df['year'].unique()))
            tmp = df[df['year']==sel_year][['institution_id','mandatory_sufficiency_pct']].set_index('institution_id')
            fig = px.imshow(tmp.T, labels=dict(x="Institution", y="Metric"), x=tmp.index, y=["Mand. Suff. %"], aspect="auto", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Document details (per institution-year)")
        sel_inst = st.selectbox("Institution for document checklist", df['institution_id'].unique(), index=0, key="doc_inst")
        sel_year = st.selectbox("Year", sorted(df['year'].unique(), reverse=True), index=0, key="doc_year")
        docs_subset = df_docs[(df_docs['institution_id']==sel_inst) & (df_docs['year']==sel_year)]
        if docs_subset.empty:
            st.info("No document records for selection.")
        else:
            st.dataframe(docs_subset[['document_name','category','submitted']].sort_values(['category','document_name']), width=900)

    # ---- RAG Analyzer
    with tabs[2]:
        st.header("🤖 Document Analyzer (RAG-style)")
        st.markdown("Upload files (pdf, docx, txt). If `sentence-transformers` is installed, embeddings will be used for similarity queries.")
        rag = SimpleRAG()
        uploaded = st.file_uploader("Upload documents for analysis", accept_multiple_files=True, type=['pdf','docx','txt','xlsx'])
        if uploaded:
            if st.button("Run Extraction & Build Vectors"):
                with st.spinner("Extracting and building... (may take time if embeddings enabled)"):
                    rag.prepare(uploaded)
                    st.success(f"Prepared {len(rag.texts)} text chunks.")
                    if rag.model is not None and rag.embeddings is not None:
                        st.success("Embeddings built.")
                    else:
                        st.info("Embeddings not available or not built.")
            st.markdown("### Text preview")
            for i,f in enumerate(uploaded):
                txt = rag.extract_text_from_file(f)
                if len(txt) > 1000:
                    st.write(f"**{f.name}** - preview:")
                    st.text_area(f"preview_{i}", txt[:2000], height=150)
                else:
                    st.write(f"**{f.name}** - content:")
                    st.text(txt[:2000])
            st.markdown("### Ask a query (similarity search)")
            q = st.text_input("Enter query to search uploaded docs")
            topk = st.slider("Top K", 1, 10, 5)
            if st.button("Query documents"):
                if rag.model and rag.embeddings is not None:
                    results = rag.query(q, topk)
                    if results:
                        for r in results:
                            st.write(f"**Score:** {r['score']:.4f}")
                            st.write(r['text'][:800])
                            st.markdown("---")
                    else:
                        st.info("No matches found.")
                else:
                    st.info("Embeddings not available — show naive regex matches")
                    # naive
                    alltxt = " ".join([rag.extract_text_from_file(f) for f in uploaded])
                    matches = re.findall(r'.{0,100}' + re.escape(q) + r'.{0,100}', alltxt, flags=re.IGNORECASE)
                    if not matches:
                        st.info("No text matches found.")
                    else:
                        for m in matches[:topk]:
                            st.write(m)

    # ---- Institution Portal
    with tabs[3]:
        st.header("🏫 Institution Portal (upload/view)")
        user = st.session_state.get('user')
        if user is None:
            st.info("Login as an Institution (sidebar) to use the portal. You can still upload files as guest to the RAG analyzer.")
        else:
            inst_id = user['institution_id']
            st.subheader(f"Welcome {user['institution_name']} ({inst_id})")
            # upload documents and store metadata in DB
            docs = st.file_uploader("Upload documents (will save metadata)", accept_multiple_files=True, type=['pdf','docx','txt','xlsx'])
            doc_type_input = st.text_input("Document type / tag (e.g. affidavit_legal_status, land_documents)", value="other")
            year_input = st.number_input("Year for documents", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=int(df['year'].max()))
            if docs and st.button("Save document metadata"):
                conn = get_db_conn()
                cur = conn.cursor()
                for f in docs:
                    cur.execute('INSERT INTO institution_documents (institution_id, year, document_name, document_type) VALUES (?,?,?,?)',
                                (inst_id, int(year_input), f.name, doc_type_input))
                conn.commit()
                st.success("Document metadata saved.")
            st.markdown("#### Previously uploaded (metadata)")
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute('SELECT * FROM institution_documents WHERE institution_id = ? ORDER BY uploaded_at DESC', (inst_id,))
            rows = cur.fetchall()
            if rows:
                dd = pd.DataFrame([dict(r) for r in rows])
                st.dataframe(dd[['year','document_name','document_type','uploaded_at']])
            else:
                st.info("No documents uploaded (metadata).")

    # ---- Reports & Downloads
    with tabs[4]:
        st.header("📥 Reports & Downloads")
        st.markdown("Download CSVs, or generate per-institution HTML report.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download full time-series CSV", data=open(CSV_FULL,'rb').read(), file_name="institutions_10yrs_20inst.csv", mime="text/csv")
        with c2:
            st.download_button("Download documents CSV", data=open(CSV_DOCS,'rb').read(), file_name="institution_documents_10yrs_20inst.csv", mime="text/csv")
        with c3:
            st.download_button("Download summary CSV", data=open(CSV_SUM,'rb').read(), file_name="institutions_summary.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Per-institution report (HTML)")
        sel_inst = st.selectbox("Select institution", df['institution_id'].unique(), key="report_inst")
        inst_df = df[df['institution_id']==sel_inst].sort_values('year', ascending=False)
        inst_docs = df_docs[df_docs['institution_id']==sel_inst].sort_values(['year','category'])
        if st.button("Generate HTML report"):
            inst_info = {
                "institution_id": sel_inst,
                "institution_name": inst_df.iloc[0]['institution_name'],
                "institution_type": inst_df.iloc[0]['institution_type'],
                "heritage_category": inst_df.iloc[0]['heritage_category']
            }
            html = generate_html_report(inst_df, inst_docs, inst_info)
            b = html_to_bytes_dl(f"{sel_inst}_report.html", html)
            st.download_button("Download HTML report", data=b, file_name=f"{sel_inst}_report.html", mime="text/html")
            st.success("Report ready for download.")
        st.markdown("**(Optional)** If reportlab installed, generate a PDF snapshot (simple) — may be basic.")
        if 'generate_pdf' not in st.session_state:
            st.session_state['generate_pdf'] = False
        if reportlab:
            if st.button("Generate sample PDF (reportlab)"):
                inst_info = {
                    "institution_id": sel_inst,
                    "institution_name": inst_df.iloc[0]['institution_name'],
                }
                # create in-memory PDF
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                text = c.beginText(40, 750)
                text.textLine(f"Report: {inst_info['institution_name']} ({inst_info['institution_id']})")
                text.textLine(f"Generated: {datetime.now().isoformat()}")
                text.textLine("")
                # add last 3 years table
                last = inst_df.head(3)
                for _,r in last.iterrows():
                    text.textLine(f"Year {int(r['year'])} | Composite: {r['composite_score']} | Risk: {r['risk_level']} | Approval: {r['approval_recommendation']}")
                c.drawText(text)
                c.showPage()
                c.save()
                buffer.seek(0)
                st.download_button("Download PDF report", data=buffer, file_name=f"{sel_inst}_summary.pdf", mime="application/pdf")
        else:
            st.info("reportlab not installed. PDF export disabled.")

    # ---- System
    with tabs[5]:
        st.header("⚙️ System")
        st.write("Model availability & diagnostics")
        st.write("SentenceTransformers available:", SentenceTransformer is not None)
        st.write("PyPDF2 available:", PyPDF2 is not None)
        st.write("docx available:", docx is not None)
        st.write("Reportlab available:", 'reportlab' in globals() and reportlab is not None)
        if st.button("Force regenerate dataset"):
            df, df_docs, summary = generate_dummy_dataset(force=True)
            st.success("Dataset regenerated.")
        st.markdown("### Quick insights")
        # top 5 by average composite score
        top5 = summary.nlargest(5, "composite_score")
        st.write("Top 5 institutions (avg composite score):")
        st.table(top5[['institution_id','institution_name','composite_score','overall_document_sufficiency_pct']])

if __name__ == "__main__":
    main()
