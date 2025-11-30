# app_patched.py
"""
Patched SIH app (single-file)
Base: SIH-app_Final.py (UI & features preserved conceptually)
Upgrade: Append Appendix-1 framework (Input, Process, Outcome, Impact)
Fixes: single set_page_config at top, safe temp storage, graceful optional libs

Save this as app_patched.py and run:
    streamlit run app_patched.py

Dependencies (core):
    pip install streamlit pandas numpy plotly scikit-learn
Optional (improves features):
    pip install sentence-transformers PyPDF2 python-docx reportlab
"""

# Standard libs
import os
import io
import re
import json
import math
import sqlite3
import tempfile
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional

# Data + viz
import pandas as pd
import numpy as np
import plotly.express as px

# Streamlit
import streamlit as st

# Optional heavy libs
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMBED = True
except Exception:
    SentenceTransformer = None
    cosine_similarity = None
    HAS_EMBED = False

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

# Silence noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# Page config - MUST be first Streamlit command usage
# ---------------------------
st.set_page_config(page_title="AI-Powered Institutional Approval - UGC/AICTE", page_icon="🏛️", layout="wide")

# ---------------------------
# App storage config - use safe temp dir (no /mnt/data)
# ---------------------------
TMP = tempfile.gettempdir()
DATA_SUB = "sih_app_patched_data"
DATA_DIR = os.path.join(TMP, DATA_SUB)
os.makedirs(DATA_DIR, exist_ok=True)

TS_CSV = os.path.join(DATA_DIR, "institutions_10yrs_20inst.csv")
DOCS_CSV = os.path.join(DATA_DIR, "institution_documents_10yrs_20inst.csv")
SUMMARY_CSV = os.path.join(DATA_DIR, "institutions_summary.csv")
DB_PATH = os.path.join(DATA_DIR, "institutions_meta.db")
LOG_PATH = os.path.join(DATA_DIR, "app_patched.log")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

def log(msg: str):
    ts = datetime.utcnow().isoformat()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{ts} {msg}\n")
    except Exception:
        pass

# ---------------------------
# Appendix-1 framework definitions
# ---------------------------
APPENDIX1 = {
    "input": {
        "description": "Faculty, labs, library, digital readiness, finance, student support",
        "metrics": [
            "faculty_strength_index", "faculty_phd_pct", "lab_quality_index",
            "library_resources_index", "digital_readiness_index", "financial_resources_index",
            "student_support_index"
        ]
    },
    "process": {
        "description": "Teaching-learning quality, curriculum updates, governance, research promotion",
        "metrics": [
            "teaching_learning_quality", "curriculum_update_freq", "governance_transparency",
            "research_promotion_score", "community_engagement_score", "green_practices_score"
        ]
    },
    "outcome": {
        "description": "Graduation, placements, publications, citations, patents, satisfaction",
        "metrics": [
            "graduation_rate_pct", "placements_pct", "research_publications_count",
            "citations_per_pub", "patents_filed", "student_satisfaction_score"
        ]
    },
    "impact": {
        "description": "Entrepreneurship, social impact, SDG alignment, internationalization",
        "metrics": [
            "entrepreneurship_index", "social_impact_score", "sdg_alignment_score",
            "internationalization_score", "awards_innovations_count"
        ]
    }
}

MANDATORY_DOCS = [
    "Institution Profile (SSR)", "Faculty Roster", "Program Curriculum Document",
    "Accreditation/Approval Certificates", "Audited Financial Statements (last 3 years)",
    "Student Enrollment Data", "Examination Results Summary", "Library & Lab Inventory"
]

SUPPORTING_DOCS = [
    "Research Publications List", "Patents & IPR filings", "Alumni Placement Reports",
    "External Collaboration MOUs", "Community Engagement Reports", "Student Feedback Surveys",
    "Teacher Development Records", "Annual Reports"
]

# ---------------------------
# Data generation (Appendix-1 guided)
# ---------------------------
def generate_dummy_dataset(force=False, n_insts=20, start_year=2016, years=10):
    # load if exists and not forcing
    if not force and os.path.exists(TS_CSV) and os.path.exists(DOCS_CSV) and os.path.exists(SUMMARY_CSV):
        df = pd.read_csv(TS_CSV)
        df_docs = pd.read_csv(DOCS_CSV)
        summary = pd.read_csv(SUMMARY_CSV)
        return df, df_docs, summary

    random.seed(42)
    np.random.seed(42)

    # build institution list
    inst_list = []
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
    for i in range(n_insts):
        inst_id = f"HEI_{i+1:02d}"
        inst_name = f"Institution {i+1}"
        inst_type = random.choice(categories)
        inst_heritage = random.choice(heritage)
        inst_list.append((inst_id, inst_name, inst_type, inst_heritage))

    yrs = list(range(start_year, start_year + years))
    rows = []
    doc_rows = []

    for inst_id, inst_name, inst_type, inst_herit in inst_list:
        # biases
        research_bias = 1.2 if "Research" in inst_type or "Multi-disciplinary" in inst_type else 0.7
        teaching_bias = 1.1 if "Teaching" in inst_type else 0.9
        community_bias = 1.2 if "Community" in inst_type or "Rural" in inst_type else 0.9

        for year in yrs:
            # INPUT metrics
            faculty_strength_index = round(np.clip(np.random.normal(6 + research_bias, 1.5), 1, 10), 2)
            faculty_phd_pct = round(np.clip(np.random.normal(40 * research_bias, 15), 2, 95), 2)
            lab_quality_index = round(np.clip(np.random.normal(6 * research_bias, 1.8), 1, 10), 2)
            library_resources_index = round(np.clip(np.random.normal(6.5, 1.5), 1, 10), 2)
            digital_readiness_index = round(np.clip(np.random.normal(6.0, 2.0), 1, 10), 2)
            financial_resources_index = round(np.clip(np.random.normal(6.0, 1.8), 1, 10), 2)
            student_support_index = round(np.clip(np.random.normal(6.5, 1.2), 1, 10), 2)

            # PROCESS metrics
            teaching_learning_quality = round(np.clip(np.random.normal(6.5 + teaching_bias*0.2, 1.5), 1, 10), 2)
            curriculum_update_freq = int(np.clip(np.random.poisson(1 + (teaching_bias - 0.8)), 0, 6))
            governance_transparency = round(np.clip(np.random.normal(6.0, 1.5), 1, 10), 2)
            research_promotion_score = round(np.clip(np.random.normal(5.0 * research_bias, 1.8), 1, 10), 2)
            community_engagement_score = round(np.clip(np.random.normal(4.0 * community_bias, 2.0), 1, 10), 2)
            green_practices_score = round(np.clip(np.random.normal(5.5, 1.8), 1, 10), 2)

            # OUTCOME
            graduation_rate_pct = round(np.clip(np.random.normal(78, 10), 30, 99), 2)
            placements_pct = round(np.clip(np.random.normal(60 * research_bias, 20), 0, 100), 2)
            research_publications_count = int(np.clip(np.random.poisson(15 * research_bias), 0, 500))
            citations_per_pub = round(np.clip(np.random.normal(3 * research_bias, 2.0), 0, 50), 2)
            patents_filed = int(np.clip(np.random.poisson(1 * research_bias), 0, 50))
            student_satisfaction_score = round(np.clip(np.random.normal(7.0, 1.5), 1, 10), 2)

            # IMPACT
            entrepreneurship_index = round(np.clip(np.random.normal(4.0 + research_bias*0.5, 1.8), 1, 10), 2)
            social_impact_score = round(np.clip(np.random.normal(5.0*community_bias, 1.8), 1, 10), 2)
            sdg_alignment_score = round(np.clip(np.random.normal(5.0, 1.5), 1, 10), 2)
            internationalization_score = round(np.clip(np.random.normal(4.0, 2.0), 0, 10), 2)
            awards_innovations_count = int(np.clip(np.random.poisson(1.5 * research_bias), 0, 50))

            # compute subscores (normalize)
            def norm(v, a, b):
                if b <= a: return 0.0
                return (v - a) / (b - a)

            input_vals = {
                "faculty_strength_index": (faculty_strength_index, 1, 10),
                "faculty_phd_pct": (faculty_phd_pct, 0, 100),
                "lab_quality_index": (lab_quality_index, 1, 10),
                "library_resources_index": (library_resources_index, 1, 10),
                "digital_readiness_index": (digital_readiness_index, 1, 10),
                "financial_resources_index": (financial_resources_index, 1, 10),
                "student_support_index": (student_support_index, 1, 10)
            }
            input_score = round(sum(norm(v,a,b) for v,a,b in input_vals.values())/len(input_vals)*100,2)

            process_vals = {
                "teaching_learning_quality": (teaching_learning_quality,1,10),
                "curriculum_update_freq": (curriculum_update_freq,0,6),
                "governance_transparency": (governance_transparency,1,10),
                "research_promotion_score": (research_promotion_score,1,10),
                "community_engagement_score": (community_engagement_score,1,10),
                "green_practices_score": (green_practices_score,1,10)
            }
            process_score = round(sum(norm(v,a,b) for v,a,b in process_vals.values())/len(process_vals)*100,2)

            outcome_vals = {
                "graduation_rate_pct": (graduation_rate_pct,30,100),
                "placements_pct": (placements_pct,0,100),
                "research_publications_count": (research_publications_count,0,200),
                "citations_per_pub": (citations_per_pub,0,10),
                "patents_filed": (patents_filed,0,20),
                "student_satisfaction_score": (student_satisfaction_score,1,10)
            }
            outcome_score = round(sum(norm(v,a,b) for v,a,b in outcome_vals.values())/len(outcome_vals)*100,2)

            impact_vals = {
                "entrepreneurship_index": (entrepreneurship_index,1,10),
                "social_impact_score": (social_impact_score,1,10),
                "sdg_alignment_score": (sdg_alignment_score,1,10),
                "internationalization_score": (internationalization_score,0,10),
                "awards_innovations_count": (awards_innovations_count,0,20)
            }
            impact_score = round(sum(norm(v,a,b) for v,a,b in impact_vals.values())/len(impact_vals)*100,2)

            # final weighted overall
            w_input, w_proc, w_out, w_imp = 0.2, 0.25, 0.4, 0.15
            overall = round(w_input*input_score + w_proc*process_score + w_out*outcome_score + w_imp*impact_score,2)

            # decision mapping
            if overall >= 75:
                risk = "Low"
                rec = "Full Approval - 5 Years"
            elif overall >= 60:
                risk = "Medium"
                rec = "Provisional Approval - 3 Years"
            elif overall >= 45:
                risk = "High"
                rec = "Conditional Approval - 1 Year"
            else:
                risk = "Critical"
                rec = "Rejection / Major Improvement Required"

            # documents sampling
            doc_prob = overall / 120.0
            mand_present = 0
            for d in MANDATORY_DOCS:
                present = np.random.rand() < min(0.95, max(0.2, doc_prob + np.random.normal(0,0.1)))
                doc_rows.append({"institution_id":inst_id, "year":year, "document_name":d, "category":"mandatory", "submitted":bool(present)})
                if present: mand_present += 1
            supp_present = 0
            for d in SUPPORTING_DOCS:
                present = np.random.rand() < min(0.9, max(0.05, doc_prob - 0.1 + np.random.normal(0,0.18)))
                doc_rows.append({"institution_id":inst_id, "year":year, "document_name":d, "category":"supporting", "submitted":bool(present)})
                if present: supp_present += 1
            mand_pct = round(mand_present/len(MANDATORY_DOCS)*100,2)
            overall_doc_pct = round((mand_present + supp_present)/(len(MANDATORY_DOCS)+len(SUPPORTING_DOCS))*100,2)

            # append row
            row = {
                "institution_id":inst_id, "institution_name":inst_name, "year":year,
                "institution_type":inst_type, "heritage_category":inst_herit,
                # inputs
                "faculty_strength_index":faculty_strength_index, "faculty_phd_pct":faculty_phd_pct,
                "lab_quality_index":lab_quality_index, "library_resources_index":library_resources_index,
                "digital_readiness_index":digital_readiness_index, "financial_resources_index":financial_resources_index,
                "student_support_index":student_support_index,
                # process
                "teaching_learning_quality":teaching_learning_quality, "curriculum_update_freq":curriculum_update_freq,
                "governance_transparency":governance_transparency, "research_promotion_score":research_promotion_score,
                "community_engagement_score":community_engagement_score, "green_practices_score":green_practices_score,
                # outcomes
                "graduation_rate_pct":graduation_rate_pct, "placements_pct":placements_pct,
                "research_publications_count":research_publications_count, "citations_per_pub":citations_per_pub,
                "patents_filed":patents_filed, "student_satisfaction_score":student_satisfaction_score,
                # impact
                "entrepreneurship_index":entrepreneurship_index, "social_impact_score":social_impact_score,
                "sdg_alignment_score":sdg_alignment_score, "internationalization_score":internationalization_score,
                "awards_innovations_count":awards_innovations_count,
                # subscores
                "input_score":input_score, "process_score":process_score, "outcome_score":outcome_score, "impact_score":impact_score,
                "overall_score":overall, "risk_level":risk, "approval_recommendation":rec,
                # docs
                "mandatory_documents_present":mand_present, "mandatory_sufficiency_pct":mand_pct,
                "supporting_documents_present":supp_present, "overall_document_sufficiency_pct":overall_doc_pct
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df_docs = pd.DataFrame(doc_rows)
    summary = df.groupby(["institution_id","institution_name","institution_type","heritage_category"]).agg({
        "overall_score":"mean","mandatory_sufficiency_pct":"mean","overall_document_sufficiency_pct":"mean",
        "research_publications_count":"mean","placements_pct":"mean","graduation_rate_pct":"mean"
    }).reset_index().round(2)
    # save CSVs
    df.to_csv(TS_CSV, index=False)
    df_docs.to_csv(DOCS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    return df, df_docs, summary

# ---------------------------
# Simple AI insights (rule-based)
# ---------------------------
def generate_ai_insights(record: Dict[str,Any]) -> Dict[str,Any]:
    # record is one row dict
    strengths = []
    weaknesses = []
    recs = []
    input_s = record.get("input_score",0)
    proc_s = record.get("process_score",0)
    out_s = record.get("outcome_score",0)
    imp_s = record.get("impact_score",0)
    overall = record.get("overall_score",0)

    if input_s >= 75:
        strengths.append("Strong inputs: faculty and infrastructure.")
    if proc_s >= 70:
        strengths.append("Robust academic processes and governance.")
    if out_s >= 75:
        strengths.append("Good outcomes: placements, graduation, research.")
    if imp_s >= 65:
        strengths.append("Impact initiatives and SDG alignment evident.")

    if input_s < 50:
        weaknesses.append("Input gaps: labs/library/digital readiness need attention.")
    if proc_s < 50:
        weaknesses.append("Process gaps: curriculum and governance need strengthening.")
    if out_s < 50:
        weaknesses.append("Outcome gaps: placements and research performance low.")
    if imp_s < 40:
        weaknesses.append("Low measurable societal impact.")

    if record.get("faculty_phd_pct",0) < 20:
        recs.append("Increase PhD-qualified faculty hiring and capacity building.")
    if record.get("mandatory_sufficiency_pct",0) < 80:
        recs.append("Improve document submission and completeness.")
    if record.get("placements_pct",0) < 50:
        recs.append("Strengthen industry partnerships and placement support.")

    risk_score = round(10 - overall/10,2)
    if risk_score < 3.5: level = "Low"
    elif risk_score < 5.5: level = "Medium"
    elif risk_score < 7.5: level = "High"
    else: level = "Critical"

    return {"strengths":strengths, "weaknesses":weaknesses, "recommendations":recs, "risk":{"score":risk_score,"level":level}}

# ---------------------------
# Simple RAG extractor (optional)
# ---------------------------
class SimpleRAG:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model = None
        self.texts = []
        self.embeddings = None
        if HAS_EMBED:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                log(f"Embedding load failed: {e}")
                self.model = None

    def extract_text(self, uploaded_file) -> str:
        name = uploaded_file.name.lower()
        try:
            if name.endswith(".pdf") and HAS_PYPDF2:
                reader = PyPDF2.PdfReader(uploaded_file)
                texts = [p.extract_text() or "" for p in reader.pages]
                return "\n".join(texts)
            elif name.endswith(".docx") and HAS_DOCX:
                d = docx.Document(uploaded_file)
                return "\n".join([p.text for p in d.paragraphs])
            else:
                raw = uploaded_file.getvalue()
                try:
                    return raw.decode("utf-8")
                except Exception:
                    return raw.decode("latin-1", errors="ignore")
        except Exception as e:
            log(f"extract_text error: {e}")
            return ""

    def prepare(self, files: List):
        self.texts = []
        for f in files:
            t = self.extract_text(f)
            if t:
                # chunk by sentences
                chunks = re.split(r'(?<=[\.\?\!])\s+', t)
                self.texts.extend([c.strip() for c in chunks if c.strip()])
        if self.model and self.texts:
            try:
                self.embeddings = self.model.encode(self.texts, show_progress_bar=False)
            except Exception as e:
                log(f"embeddings failed: {e}")
                self.embeddings = None

    def query(self, q: str, topk:int=5):
        if not self.embeddings or not self.model:
            return []
        qemb = self.model.encode([q])
        sims = cosine_similarity(qemb, self.embeddings)[0]
        idxs = np.argsort(sims)[-topk:][::-1]
        return [{"text": self.texts[i], "score": float(sims[i])} for i in idxs if sims[i] > 0]

# ---------------------------
# Light DB metadata functions
# ---------------------------
def init_meta_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS rag_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            institution_id TEXT,
            analysis_type TEXT,
            extracted_data TEXT,
            confidence_score REAL,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    return conn

# ---------------------------
# UI helpers & main
# ---------------------------
def sidebar_login(summary_df):
    st.sidebar.title("User / Institution")
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    if st.session_state['user'] is None:
        mode = st.sidebar.radio("Mode", ["Guest", "Institution User"])
        if mode == "Institution User":
            username = st.sidebar.text_input("Institution ID (e.g., HEI_01)")
            pwd = st.sidebar.text_input("Password (demo)", type="password")
            if st.sidebar.button("Login"):
                if username in summary_df['institution_id'].values:
                    row = summary_df[summary_df['institution_id'] == username].iloc[0]
                    st.session_state['user'] = {"institution_id":username, "institution_name":row['institution_name']}
                    st.sidebar.success(f"Signed in: {row['institution_name']}")
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Institution ID not found.")
        else:
            st.sidebar.info("Browsing as Guest.")
    else:
        st.sidebar.success(f"Signed in: {st.session_state['user']['institution_name']}")
        if st.sidebar.button("Logout"):
            st.session_state['user'] = None
            st.experimental_rerun()

def main():
    st.title("🏛️ SIH - UGC/AICTE — Patched (Appendix-1 added)")
    st.markdown("This app is the SIH UI base upgraded with Appendix-1 (Input → Process → Outcome → Impact).")

    # Data controls
    with st.expander("Dataset generation / load"):
        col1, col2 = st.columns(2)
        if col1.button("Generate dataset (force)"):
            df, df_docs, summary = generate_dummy_dataset(force=True)
            st.success("Dataset generated.")
        if col2.button("Load dataset (if exists)"):
            df, df_docs, summary = generate_dummy_dataset(force=False)
            st.success("Dataset loaded.")
        st.write(f"Temporary data folder: `{DATA_DIR}` (ephemeral on some hosts)")

    df, df_docs, summary = generate_dummy_dataset(force=False)
    conn = init_meta_db()

    # sidebar login
    sidebar_login(summary)

    # Tabs similar to SIH-app_Final.py
    tabs = st.tabs(["Dashboard", "Appendix-1", "Documents & RAG", "Institution Portal", "Reports", "Admin"])
    # Dashboard
    with tabs[0]:
        st.header("Dashboard")
        year = st.selectbox("Select year", sorted(df['year'].unique(), reverse=True))
        df_year = df[df['year'] == year]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Institutions", df_year['institution_id'].nunique())
        c2.metric("Avg Overall Score", f"{df_year['overall_score'].mean():.2f}/100")
        c3.metric("Avg Mandatory Suff %", f"{df_year['mandatory_sufficiency_pct'].mean():.1f}%")
        c4.metric("Avg Overall Doc Suff %", f"{df_year['overall_document_sufficiency_pct'].mean():.1f}%")
        st.markdown("---")
        st.subheader("Top institutions (selected year)")
        st.dataframe(df_year.sort_values("overall_score", ascending=False).head(10)[["institution_id","institution_name","overall_score","approval_recommendation"]])

    # Appendix-1 tab — added framework view & AI insights
    with tabs[1]:
        st.header("Appendix-1 (Input → Process → Outcome → Impact)")
        inst = st.selectbox("Select institution", df['institution_id'].unique(), key="apx_inst")
        yr = st.selectbox("Select year", sorted(df['year'].unique(), reverse=True), key="apx_year")
        row = df[(df['institution_id']==inst) & (df['year']==yr)]
        if row.empty:
            st.info("No data for selection.")
        else:
            r = row.iloc[0].to_dict()
            st.subheader(f"{r['institution_name']} — {yr}")
            seq = st.columns(4)
            seq[0].metric("Input score", f"{r['input_score']:.2f}/100")
            seq[1].metric("Process score", f"{r['process_score']:.2f}/100")
            seq[2].metric("Outcome score", f"{r['outcome_score']:.2f}/100")
            seq[3].metric("Impact score", f"{r['impact_score']:.2f}/100")
            st.markdown("### Metric groups (values shown)")
            for group in ["input","process","outcome","impact"]:
                st.subheader(group.capitalize())
                cols = st.columns(3)
                metrics = APPENDIX1[group]["metrics"]
                for i, m in enumerate(metrics):
                    val = r.get(m, "N/A")
                    cols[i%3].write(f"**{m.replace('_',' ').title()}**: {val}")

            st.markdown("### AI Insights (rule-based from Appendix-1 subscores)")
            insights = generate_ai_insights(r)
            st.write("**Strengths**")
            for s in insights["strengths"]: st.write("- " + s)
            st.write("**Weaknesses**")
            for s in insights["weaknesses"]: st.write("- " + s)
            st.write("**Recommendations**")
            for s in insights["recommendations"]: st.write("- " + s)
            st.write("**Risk assessment**")
            st.json(insights["risk"])

    # Documents & RAG
    with tabs[2]:
        st.header("Documents & RAG")
        st.markdown("Document sufficiency and (optional) RAG extraction.")
        agg = df.groupby("institution_id").agg({"mandatory_sufficiency_pct":"mean","overall_document_sufficiency_pct":"mean"}).reset_index()
        fig = px.bar(agg, x="institution_id", y="mandatory_sufficiency_pct", title="Average Mandatory Document Sufficiency %")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Per-institution check")
        inst_sel = st.selectbox("Institution", df['institution_id'].unique(), key="docs_inst")
        yr_sel = st.selectbox("Year", sorted(df['year'].unique(), reverse=True), key="docs_year")
        docs_df = df_docs[(df_docs['institution_id']==inst_sel) & (df_docs['year']==yr_sel)]
        if docs_df.empty:
            st.info("No document records.")
        else:
            st.dataframe(docs_df[['document_name','category','submitted']].sort_values(['category','document_name']))

        st.markdown("### RAG (optional)")
        rag = SimpleRAG()
        uploaded = st.file_uploader("Upload docs (pdf/docx/txt)", accept_multiple_files=True)
        if uploaded:
            if st.button("Prepare RAG"):
                rag.prepare(uploaded)
                st.success(f"Prepared {len(rag.texts)} chunks.")
            q = st.text_input("Query uploaded docs", key="rag_q")
            if q and st.button("Query RAG"):
                if rag.embeddings is not None and HAS_EMBED:
                    res = rag.query(q, topk=5)
                    for r in res:
                        st.write(f"Score: {r['score']:.4f}")
                        st.write(r['text'][:800])
                        st.markdown("---")
                else:
                    st.info("Embeddings not available. Doing naive substring search.")
                    alltxt = " ".join([rag.extract_text(f) for f in uploaded])
                    matches = re.findall(r'.{0,200}'+re.escape(q)+r'.{0,200}', alltxt, flags=re.IGNORECASE)
                    if matches:
                        for m in matches[:5]:
                            st.write(m)
                    else:
                        st.info("No matches found.")

    # Institution Portal (metadata save)
    with tabs[3]:
        st.header("Institution Portal")
        user = st.session_state.get('user', None)
        if not user:
            st.info("Login as Institution via sidebar to access portal features.")
        else:
            st.subheader(f"Welcome {user['institution_name']} ({user['institution_id']})")
            uploaded = st.file_uploader("Upload documents (metadata only)", accept_multiple_files=True)
            doc_type = st.text_input("Document type", value="other")
            year_input = st.number_input("Year", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=int(df['year'].max()))
            if uploaded and st.button("Save metadata"):
                meta_file = os.path.join(DATA_DIR, "uploaded_metadata.csv")
                rows = []
                for f in uploaded:
                    rows.append({"institution_id":user['institution_id'],"year":int(year_input),"file_name":f.name,"doc_type":doc_type,"uploaded_at":datetime.utcnow().isoformat()})
                if os.path.exists(meta_file):
                    old = pd.read_csv(meta_file)
                    new = pd.DataFrame(rows)
                    pd.concat([old,new], ignore_index=True).to_csv(meta_file, index=False)
                else:
                    pd.DataFrame(rows).to_csv(meta_file, index=False)
                st.success("Metadata saved.")
            meta_file = os.path.join(DATA_DIR, "uploaded_metadata.csv")
            if os.path.exists(meta_file):
                saved = pd.read_csv(meta_file)
                my = saved[saved['institution_id']==user['institution_id']]
                if not my.empty:
                    st.dataframe(my.sort_values('uploaded_at', ascending=False))
                else:
                    st.info("No metadata uploaded by your institution.")
            else:
                st.info("No metadata uploaded yet.")

    # Reports
    with tabs[4]:
        st.header("Reports")
        c1, c2, c3 = st.columns(3)
        if os.path.exists(TS_CSV):
            with open(TS_CSV,"rb") as f:
                c1.download_button("Download time-series CSV", data=f.read(), file_name="institutions_10yrs_20inst.csv", mime="text/csv")
        else:
            c1.info("Time-series CSV not found (generate data).")
        if os.path.exists(DOCS_CSV):
            with open(DOCS_CSV,"rb") as f:
                c2.download_button("Download docs CSV", data=f.read(), file_name="institution_documents_10yrs_20inst.csv", mime="text/csv")
        else:
            c2.info("Docs CSV not found.")
        if os.path.exists(SUMMARY_CSV):
            with open(SUMMARY_CSV,"rb") as f:
                c3.download_button("Download summary CSV", data=f.read(), file_name="institutions_summary.csv", mime="text/csv")
        else:
            c3.info("Summary CSV not found.")

        st.markdown("---")
        st.subheader("Per-institution HTML report (Appendix-1)")
        sel_inst = st.selectbox("Institution for report", df['institution_id'].unique(), key="rep_inst")
        inst_df = df[df['institution_id']==sel_inst].sort_values("year", ascending=False)
        inst_docs = df_docs[df_docs['institution_id']==sel_inst] if not df_docs.empty else pd.DataFrame()
        if st.button("Generate HTML report"):
            h = ["<html><head><meta charset='utf-8'><title>Report</title></head><body>"]
            h.append(f"<h1>{inst_df.iloc[0]['institution_name']} ({sel_inst})</h1>")
            h.append("<h2>Recent years</h2><table border='1'><tr><th>Year</th><th>Overall</th><th>Input</th><th>Process</th><th>Outcome</th><th>Impact</th></tr>")
            for _,r in inst_df.head(3).iterrows():
                h.append(f"<tr><td>{int(r['year'])}</td><td>{r['overall_score']}</td><td>{r['input_score']}</td><td>{r['process_score']}</td><td>{r['outcome_score']}</td><td>{r['impact_score']}</td></tr>")
            h.append("</table>")
            # insights
            insights = generate_ai_insights(inst_df.iloc[0].to_dict())
            h.append("<h3>AI Insights (latest)</h3><ul>")
            for s in insights["strengths"]: h.append(f"<li><b>Strength:</b> {s}</li>")
            for s in insights["weaknesses"]: h.append(f"<li><b>Weakness:</b> {s}</li>")
            for s in insights["recommendations"]: h.append(f"<li><b>Recommendation:</b> {s}</li>")
            h.append("</ul>")
            # docs latest
            if not inst_docs.empty:
                ly = int(inst_docs['year'].max())
                h.append(f"<h3>Document checklist (Year {ly})</h3><table border='1'><tr><th>Doc</th><th>Category</th><th>Submitted</th></tr>")
                for _,d in inst_docs[inst_docs['year']==ly].iterrows():
                    h.append(f"<tr><td>{d['document_name']}</td><td>{d['category']}</td><td>{'Yes' if d['submitted'] else 'No'}</td></tr>")
                h.append("</table>")
            h.append("</body></html>")
            b = "\n".join(h).encode("utf-8")
            st.download_button("Download HTML report", data=b, file_name=f"{sel_inst}_report.html", mime="text/html")
            st.success("Report ready.")

    # Admin tab - diagnostics
    with tabs[5]:
        st.header("Admin & Diagnostics")
        st.write("Optional components availability:")
        st.write("sentence-transformers:", HAS_EMBED)
        st.write("PyPDF2:", HAS_PYPDF2)
        st.write("python-docx:", HAS_DOCX)
        st.write("reportlab:", HAS_REPORTLAB)
        st.write("Data dir:", DATA_DIR)
        if st.button("Show recent app log"):
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    st.text("".join(f.readlines()[-200:]))
            else:
                st.info("No logs yet.")
        if st.button("Force regenerate dataset (overwrite)"):
            df, df_docs, summary = generate_dummy_dataset(force=True)
            st.success("Regenerated dataset.")

if __name__ == "__main__":
    main()
