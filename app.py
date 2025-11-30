# app_upgraded.py
"""
Upgraded SIH app (single-file)
- Retains UI & features of SIH-app_Final.py
- Adds Appendix-1 (Input, Process, Outcome, Impact) framework
- Generates 20 institutions x 10 years dummy data
- Document compliance engine (mandatory/supporting)
- Composite scoring & AI insights (rule-based)
- Optional RAG embeddings (sentence-transformers) and file parsing (PyPDF2, python-docx)
- Uses tempfile for writable storage (no /mnt/data)
"""

# Standard imports
import os
import io
import sys
import json
import math
import random
import tempfile
import warnings
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Data + plotting
import pandas as pd
import numpy as np
import plotly.express as px

# Streamlit
import streamlit as st

# Optional packages (graceful)
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

# Silence some warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# App configuration - safe temp dir (works on Streamlit Cloud)
APP_LABEL = "SIH - UGC/AICTE Institutional Approval (Upgraded)"
TMP = tempfile.gettempdir()
APP_DATA_SUB = "sih_app_upgraded_data"
DATA_DIR = os.path.join(TMP, APP_DATA_SUB)
os.makedirs(DATA_DIR, exist_ok=True)

CSV_TIME_SERIES = os.path.join(DATA_DIR, "institutions_10yrs_20inst.csv")
CSV_DOCS = os.path.join(DATA_DIR, "institution_documents_10yrs_20inst.csv")
CSV_SUMMARY = os.path.join(DATA_DIR, "institutions_summary.csv")
DB_PATH = os.path.join(DATA_DIR, "institutions_meta.db")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
st.set_page_config(page_title=APP_LABEL, page_icon="🏛️", layout="wide")

# -------------------------------
# Utilities
# -------------------------------
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def load_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def now_iso():
    return datetime.utcnow().isoformat()

# -------------------------------
# Appendix-1 parameter definitions
# -------------------------------
APPENDIX1 = {
    "input": {
        "description": "Institution inputs: faculty, infrastructure, library, digital readiness, funding, student support",
        "metrics": [
            "faculty_strength_index",
            "faculty_phd_pct",
            "lab_quality_index",
            "library_resources_index",
            "digital_readiness_index",
            "financial_resources_index",
            "student_support_index"
        ]
    },
    "process": {
        "description": "Teaching-learning, curriculum updates, governance, research promotion, engagement",
        "metrics": [
            "teaching_learning_quality",
            "curriculum_update_freq",
            "governance_transparency",
            "research_promotion_score",
            "community_engagement_score",
            "green_practices_score"
        ]
    },
    "outcome": {
        "description": "Graduation, placements, publications, patents, student satisfaction",
        "metrics": [
            "graduation_rate_pct",
            "placements_pct",
            "research_publications_count",
            "citations_per_pub",
            "patents_filed",
            "student_satisfaction_score"
        ]
    },
    "impact": {
        "description": "Societal impact, entrepreneurship, SDG alignment, internationalization",
        "metrics": [
            "entrepreneurship_index",
            "social_impact_score",
            "sdg_alignment_score",
            "internationalization_score",
            "awards_innovations_count"
        ]
    }
}

# Mandatory and supporting documents (from earlier)
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

# -------------------------------
# Dummy data generator (20 inst x 10 years)
# -------------------------------
def generate_dummy_institutions(n_insts=20, start_year=2016, years=10, force=False):
    # load existing if present and not forcing
    if not force and os.path.exists(CSV_TIME_SERIES) and os.path.exists(CSV_DOCS) and os.path.exists(CSV_SUMMARY):
        df = pd.read_csv(CSV_TIME_SERIES)
        df_docs = pd.read_csv(CSV_DOCS)
        summary = pd.read_csv(CSV_SUMMARY)
        return df, df_docs, summary

    random.seed(42)
    np.random.seed(42)

    # construct 20 names with categories similar to Appendix-1
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

    institutions = []
    for i in range(n_insts):
        inst_id = f"HEI_{i+1:02d}"
        inst_name = f"Institute {i+1}"
        inst_type = random.choice(categories)
        inst_heritage = random.choice(heritage)
        institutions.append((inst_id, inst_name, inst_type, inst_heritage))

    yrs = list(range(start_year, start_year + years))
    rows = []
    doc_rows = []

    for inst_id, inst_name, inst_type, inst_heritage in institutions:
        # bias factors by type
        research_bias = 1.2 if "Research" in inst_type or "Multi-disciplinary" in inst_type else 0.6
        teaching_bias = 1.1 if "Teaching" in inst_type else 0.8
        community_bias = 1.2 if "Community" in inst_type or "Rural" in inst_type else 0.8

        for year in yrs:
            # Input metrics
            faculty_strength_index = round(np.clip(np.random.normal(6 + research_bias, 1.5), 1, 10), 2)
            faculty_phd_pct = round(np.clip(np.random.normal(40 * research_bias, 15), 2, 95), 2)
            lab_quality_index = round(np.clip(np.random.normal(6 * research_bias, 1.8), 1, 10), 2)
            library_resources_index = round(np.clip(np.random.normal(6.5, 1.5), 1, 10), 2)
            digital_readiness_index = round(np.clip(np.random.normal(6.0, 2.0), 1, 10), 2)
            financial_resources_index = round(np.clip(np.random.normal(6.0, 1.8), 1, 10), 2)
            student_support_index = round(np.clip(np.random.normal(6.5, 1.2), 1, 10), 2)

            # Process metrics
            teaching_learning_quality = round(np.clip(np.random.normal(6.5 + teaching_bias*0.2, 1.5), 1, 10), 2)
            curriculum_update_freq = int(np.clip(np.random.poisson(1 + (teaching_bias - 0.8)), 0, 6))
            governance_transparency = round(np.clip(np.random.normal(6.0, 1.5), 1, 10), 2)
            research_promotion_score = round(np.clip(np.random.normal(5.0 * research_bias, 1.8), 1, 10), 2)
            community_engagement_score = round(np.clip(np.random.normal(4.0 * community_bias, 2.0), 1, 10), 2)
            green_practices_score = round(np.clip(np.random.normal(5.5, 1.8), 1, 10), 2)

            # Outcome metrics
            graduation_rate_pct = round(np.clip(np.random.normal(78, 10), 30, 99), 2)
            placements_pct = round(np.clip(np.random.normal(60 * research_bias, 20), 0, 100), 2)
            research_publications_count = int(np.clip(np.random.poisson(15 * research_bias), 0, 500))
            citations_per_pub = round(np.clip(np.random.normal(3 * research_bias, 2.0), 0, 50), 2)
            patents_filed = int(np.clip(np.random.poisson(1 * research_bias), 0, 50))
            student_satisfaction_score = round(np.clip(np.random.normal(7.0, 1.5), 1, 10), 2)

            # Impact metrics
            entrepreneurship_index = round(np.clip(np.random.normal(4.0 + research_bias*0.5, 1.8), 1, 10), 2)
            social_impact_score = round(np.clip(np.random.normal(5.0*community_bias, 1.8), 1, 10), 2)
            sdg_alignment_score = round(np.clip(np.random.normal(5.0, 1.5), 1, 10), 2)
            internationalization_score = round(np.clip(np.random.normal(4.0, 2.0), 0, 10), 2)
            awards_innovations_count = int(np.clip(np.random.poisson(1.5 * research_bias), 0, 50))

            # Composite scoring: calculate input/process/outcome/impact subscores (normalized)
            # We map each metric to 0-1 using reasonable bounds, then weighted sum
            def norm(val, vmin, vmax):
                if vmax <= vmin: return 0.0
                return float((val - vmin) / (vmax - vmin))

            # Input subscore
            input_metrics = {
                "faculty_strength_index": (faculty_strength_index, 1, 10),
                "faculty_phd_pct": (faculty_phd_pct, 0, 100),
                "lab_quality_index": (lab_quality_index, 1, 10),
                "library_resources_index": (library_resources_index, 1, 10),
                "digital_readiness_index": (digital_readiness_index, 1, 10),
                "financial_resources_index": (financial_resources_index, 1, 10),
                "student_support_index": (student_support_index, 1, 10)
            }
            input_score = 0.0
            # equal weighting across inputs (can be tuned)
            for v, mn, mx in input_metrics.values():
                input_score += norm(v, mn, mx)
            input_score = round((input_score / len(input_metrics)) * 100, 2)

            # Process subscore
            process_metrics = {
                "teaching_learning_quality": (teaching_learning_quality, 1, 10),
                "curriculum_update_freq": (curriculum_update_freq, 0, 6),
                "governance_transparency": (governance_transparency, 1, 10),
                "research_promotion_score": (research_promotion_score, 1, 10),
                "community_engagement_score": (community_engagement_score, 1, 10),
                "green_practices_score": (green_practices_score, 1, 10)
            }
            process_score = 0.0
            for v, mn, mx in process_metrics.values():
                process_score += norm(v, mn, mx)
            process_score = round((process_score / len(process_metrics)) * 100, 2)

            # Outcome subscore
            outcome_metrics = {
                "graduation_rate_pct": (graduation_rate_pct, 30, 100),
                "placements_pct": (placements_pct, 0, 100),
                "research_publications_count": (research_publications_count, 0, 200),
                "citations_per_pub": (citations_per_pub, 0, 10),
                "patents_filed": (patents_filed, 0, 20),
                "student_satisfaction_score": (student_satisfaction_score, 1, 10)
            }
            outcome_score = 0.0
            for v, mn, mx in outcome_metrics.values():
                outcome_score += norm(v, mn, mx)
            outcome_score = round((outcome_score / len(outcome_metrics)) * 100, 2)

            # Impact subscore
            impact_metrics = {
                "entrepreneurship_index": (entrepreneurship_index, 1, 10),
                "social_impact_score": (social_impact_score, 1, 10),
                "sdg_alignment_score": (sdg_alignment_score, 1, 10),
                "internationalization_score": (internationalization_score, 0, 10),
                "awards_innovations_count": (awards_innovations_count, 0, 20)
            }
            impact_score = 0.0
            for v, mn, mx in impact_metrics.values():
                impact_score += norm(v, mn, mx)
            impact_score = round((impact_score / len(impact_metrics)) * 100, 2)

            # Overall score — weighted (you can tune weights)
            w_input, w_process, w_outcome, w_impact = 0.2, 0.25, 0.4, 0.15
            overall_score = round(w_input * input_score + w_process * process_score + w_outcome * outcome_score + w_impact * impact_score, 2)

            # Risk & recommendation
            if overall_score >= 75:
                risk_level = "Low"
                approval_recommendation = "Full Approval - 5 Years"
            elif overall_score >= 60:
                risk_level = "Medium"
                approval_recommendation = "Provisional Approval - 3 Years"
            elif overall_score >= 45:
                risk_level = "High"
                approval_recommendation = "Conditional Approval - 1 Year"
            else:
                risk_level = "Critical"
                approval_recommendation = "Rejection / Major Improvement Required"

            # Document submission simulation
            doc_prob_base = overall_score / 120.0  # scale in ~0-0.9
            mand_present = 0
            for doc in MANDATORY_DOCS:
                present = np.random.rand() < min(0.95, max(0.2, doc_prob_base + np.random.normal(0, 0.1)))
                doc_rows.append({"institution_id": inst_id, "year": year, "document_name": doc, "category": "mandatory", "submitted": bool(present)})
                if present: mand_present += 1
            supp_present = 0
            for doc in SUPPORTING_DOCS:
                present = np.random.rand() < min(0.9, max(0.05, doc_prob_base - 0.1 + np.random.normal(0, 0.18)))
                doc_rows.append({"institution_id": inst_id, "year": year, "document_name": doc, "category": "supporting", "submitted": bool(present)})
                if present: supp_present += 1
            mand_pct = round(mand_present / len(MANDATORY_DOCS) * 100, 2)
            overall_doc_pct = round((mand_present + supp_present) / (len(MANDATORY_DOCS) + len(SUPPORTING_DOCS)) * 100, 2)

            # assemble the row
            row = {
                "institution_id": inst_id,
                "institution_name": inst_name,
                "year": year,
                "institution_type": inst_type,
                "heritage_category": inst_heritage,
                # inputs
                "faculty_strength_index": faculty_strength_index,
                "faculty_phd_pct": faculty_phd_pct,
                "lab_quality_index": lab_quality_index,
                "library_resources_index": library_resources_index,
                "digital_readiness_index": digital_readiness_index,
                "financial_resources_index": financial_resources_index,
                "student_support_index": student_support_index,
                # process
                "teaching_learning_quality": teaching_learning_quality,
                "curriculum_update_freq": curriculum_update_freq,
                "governance_transparency": governance_transparency,
                "research_promotion_score": research_promotion_score,
                "community_engagement_score": community_engagement_score,
                "green_practices_score": green_practices_score,
                # outcomes
                "graduation_rate_pct": graduation_rate_pct,
                "placements_pct": placements_pct,
                "research_publications_count": research_publications_count,
                "citations_per_pub": citations_per_pub,
                "patents_filed": patents_filed,
                "student_satisfaction_score": student_satisfaction_score,
                # impact
                "entrepreneurship_index": entrepreneurship_index,
                "social_impact_score": social_impact_score,
                "sdg_alignment_score": sdg_alignment_score,
                "internationalization_score": internationalization_score,
                "awards_innovations_count": awards_innovations_count,
                # subscores & composite
                "input_score": input_score,
                "process_score": process_score,
                "outcome_score": outcome_score,
                "impact_score": impact_score,
                "overall_score": overall_score,
                "risk_level": risk_level,
                "approval_recommendation": approval_recommendation,
                # docs
                "mandatory_documents_present": mand_present,
                "mandatory_sufficiency_pct": mand_pct,
                "supporting_documents_present": supp_present,
                "overall_document_sufficiency_pct": overall_doc_pct
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df_docs = pd.DataFrame(doc_rows)
    summary = df.groupby(["institution_id", "institution_name", "institution_type", "heritage_category"]).agg({
        "overall_score": "mean",
        "mandatory_sufficiency_pct": "mean",
        "overall_document_sufficiency_pct": "mean",
        "research_publications_count": "mean",
        "placements_pct": "mean",
        "graduation_rate_pct": "mean"
    }).reset_index()
    # rename overall_score -> composite like previous apps
    summary.rename(columns={"overall_score":"composite_score"}, inplace=True)
    summary = summary.round(2)

    # save CSVs
    save_csv(df, CSV_TIME_SERIES)
    save_csv(df_docs, CSV_DOCS)
    save_csv(summary, CSV_SUMMARY)

    return df, df_docs, summary

# -------------------------------
# AI Insights (rule-based) - maps Appendix-1 subscores to strengths/weaknesses
# -------------------------------
def generate_ai_insights_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # inputs: row is a dict (single-year)
    insights = {"strengths": [], "weaknesses": [], "recommendations": [], "risk_assessment": {}}
    # use subscores
    input_s = row.get("input_score", 0)
    process_s = row.get("process_score", 0)
    outcome_s = row.get("outcome_score", 0)
    impact_s = row.get("impact_score", 0)
    overall = row.get("overall_score", row.get("composite_score", 0))

    # strengths
    if input_s >= 75:
        insights["strengths"].append("Strong institutional inputs (faculty/infrastructure).")
    if process_s >= 70:
        insights["strengths"].append("Robust teaching-learning and governance processes.")
    if outcome_s >= 75:
        insights["strengths"].append("Excellent outcomes: placements/graduation/pubs.")
    if impact_s >= 65:
        insights["strengths"].append("Good societal and SDG alignment impact.")

    # weaknesses
    if input_s < 50:
        insights["weaknesses"].append("Weak inputs — consider hiring, labs & library upgrades.")
    if process_s < 50:
        insights["weaknesses"].append("Processes need improvement — curriculum updates & governance.")
    if outcome_s < 50:
        insights["weaknesses"].append("Poor outcomes — focus on placements and student support.")
    if impact_s < 40:
        insights["weaknesses"].append("Low community/SDG impact; engage with local stakeholders.")

    # recommendations (simple rules)
    if row.get("faculty_phd_pct", 0) < 15:
        insights["recommendations"].append("Invest in PhD hiring or faculty development.")
    if row.get("research_publications_count", 0) < 10:
        insights["recommendations"].append("Promote research & grant-seeking; incentivize publications.")
    if row.get("mandatory_sufficiency_pct", 0) < 60:
        insights["recommendations"].append("Improve document submission & compliance processes.")
    if row.get("placements_pct", 0) < 40:
        insights["recommendations"].append("Strengthen industry linkages and placement training.")

    # risk score & level (simple mapping)
    risk_score = 10.0 - (overall / 10.0)  # higher overall -> lower risk
    if risk_score < 3.5:
        level = "Low"
    elif risk_score < 5.5:
        level = "Medium"
    elif risk_score < 7.5:
        level = "High"
    else:
        level = "Critical"

    insights["risk_assessment"] = {"score": round(risk_score, 2), "level": level, "computed_on": now_iso()}
    return insights

# -------------------------------
# Simple RAG utilities (optional)
# -------------------------------
class SimpleRAG:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model = None
        self.text_chunks = []
        self.embeddings = None
        if HAS_EMBED:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None

    def extract_text_from_file(self, uploaded_file) -> str:
        name = uploaded_file.name.lower()
        try:
            if name.endswith(".pdf") and HAS_PYPDF2:
                reader = PyPDF2.PdfReader(uploaded_file)
                texts = []
                for p in reader.pages:
                    texts.append(p.extract_text() or "")
                return "\n".join(texts)
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
            return ""

    def chunk_text(self, text: str, chunk_size=800, overlap=200) -> List[str]:
        if not text:
            return []
        sentences = re.split(r'(?<=[\.\?\!])\s+', text)
        chunks = []
        cur = ""
        for s in sentences:
            if len(cur) + len(s) + 1 > chunk_size:
                chunks.append(cur.strip())
                cur = s[-overlap:] if overlap > 0 else s
            else:
                cur = (cur + " " + s).strip() if cur else s
        if cur:
            chunks.append(cur.strip())
        return chunks

    def prepare(self, uploaded_files: List[Any]):
        self.text_chunks = []
        for f in uploaded_files:
            t = self.extract_text_from_file(f)
            if t:
                self.text_chunks.extend(self.chunk_text(t))
        # build embeddings if model available
        if self.model and self.text_chunks:
            try:
                self.embeddings = self.model.encode(self.text_chunks, show_progress_bar=False)
            except Exception:
                self.embeddings = None

    def query(self, q: str, topk=5):
        if self.model is None or self.embeddings is None or len(self.text_chunks) == 0:
            return []
        qemb = self.model.encode([q])
        sims = cosine_similarity(qemb, self.embeddings)[0]
        idxs = np.argsort(sims)[-topk:][::-1]
        return [{"text": self.text_chunks[i], "score": float(sims[i])} for i in idxs if sims[i] > 0]

# -------------------------------
# App UI
# -------------------------------
def sidebar_login(summary_df: pd.DataFrame):
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
                    inst_row = summary_df[summary_df['institution_id'] == username].iloc[0]
                    st.session_state['user'] = {"institution_id": username, "institution_name": inst_row['institution_name']}
                    st.sidebar.success(f"Logged in as {inst_row['institution_name']}")
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Institution ID not found. Use Guest or correct ID.")
        else:
            st.sidebar.info("Browsing as Guest.")
    else:
        st.sidebar.success(f"Signed in: {st.session_state['user']['institution_name']}")
        if st.sidebar.button("Logout"):
            st.session_state['user'] = None
            st.experimental_rerun()

def main():
    st.title("🏛️ SIH — UGC/AICTE Institutional Approval (Upgraded)")
    st.markdown("This app has appended **Appendix-1** (Input → Process → Outcome → Impact) framework to your working SIH UI. It generates dummy 10-year data for 20 institutions and calculates composite scores, document sufficiency and AI insights.")

    # dataset generation controls
    with st.expander("Data generation / load"):
        st.write("Generate or load dummy dataset (20 institutions × 10 years). Uses a safe temp folder.")
        col1, col2 = st.columns(2)
        if col1.button("Generate dataset (force)"):
            df, df_docs, summary = generate_dummy_institutions(force=True)
            st.success("Dummy dataset generated.")
        if col2.button("Load dataset if exists"):
            df, df_docs, summary = generate_dummy_institutions(force=False)
            st.success("Dataset loaded (if present).")
        st.write("Data stored in temp directory (ephemeral on some hosts).")

    # ensure dataset exists
    df, df_docs, summary = generate_dummy_institutions(force=False)

    # sidebar login
    sidebar_login(summary)

    # main tabs (keeps style similar to your SIH app)
    tabs = st.tabs(["Dashboard", "Appendix-1 Insights", "Documents & RAG", "Institution Portal", "Reports"])
    # Dashboard
    with tabs[0]:
        st.header("📊 Dashboard")
        year_sel = st.selectbox("Select year", sorted(df['year'].unique(), reverse=True))
        df_year = df[df['year'] == year_sel]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Institutions", df_year['institution_id'].nunique())
        with c2:
            st.metric("Avg Overall Score", f"{df_year['overall_score'].mean():.2f}/100")
        with c3:
            st.metric("Avg Mandatory Suff %", f"{df_year['mandatory_sufficiency_pct'].mean():.1f}%")
        with c4:
            st.metric("Avg Overall Doc Suff %", f"{df_year['overall_document_sufficiency_pct'].mean():.1f}%")

        st.markdown("---")
        st.subheader("Top Institutions (selected year)")
        st.dataframe(df_year.sort_values("overall_score", ascending=False).head(10)[['institution_id','institution_name','overall_score','approval_recommendation']])

        st.subheader("Score Distribution")
        fig = px.histogram(df_year, x="overall_score", nbins=20)
        st.plotly_chart(fig, use_container_width=True)

    # Appendix-1 Insights tab
    with tabs[1]:
        st.header("📚 Appendix-1: Input → Process → Outcome → Impact Insights")
        st.markdown("Choose an institution and year to see Appendix-1 subscores, supporting metrics, AI insights and recommendations.")
        inst = st.selectbox("Select institution", df['institution_id'].unique())
        year = st.selectbox("Select year", sorted(df['year'].unique(), reverse=True), key="apx1_year")
        row = df[(df['institution_id'] == inst) & (df['year'] == year)]
        if row.empty:
            st.info("No data for selection.")
        else:
            r = row.iloc[0].to_dict()
            st.subheader(f"{r['institution_name']} — {year}")
            # show subscores
            cols = st.columns(4)
            cols[0].metric("Input (subscore)", f"{r['input_score']:.2f}/100")
            cols[1].metric("Process (subscore)", f"{r['process_score']:.2f}/100")
            cols[2].metric("Outcome (subscore)", f"{r['outcome_score']:.2f}/100")
            cols[3].metric("Impact (subscore)", f"{r['impact_score']:.2f}/100")
            st.markdown("### Key Appendix-1 Metrics")
            # display groups
            for group in ["input","process","outcome","impact"]:
                st.subheader(group.capitalize())
                metrics = APPENDIX1[group]["metrics"]
                metric_cols = st.columns(3)
                for i, m in enumerate(metrics):
                    val = r.get(m, "N/A")
                    metric_cols[i % 3].write(f"**{m.replace('_',' ').title()}:** {val}")
            # AI insights
            insights = generate_ai_insights_from_row(r)
            st.subheader("AI Insights")
            st.write("**Strengths**")
            for s in insights["strengths"]:
                st.write("- " + s)
            st.write("**Weaknesses**")
            for s in insights["weaknesses"]:
                st.write("- " + s)
            st.write("**Recommendations**")
            for s in insights["recommendations"]:
                st.write("- " + s)
            st.write("**Risk Assessment**")
            st.json(insights["risk_assessment"])

    # Documents & RAG
    with tabs[2]:
        st.header("📂 Documents & RAG Analyzer")
        st.markdown("Document sufficiency heatmap & RAG (optional) for uploaded files.")
        # heatmap summary (avg)
        agg = df.groupby("institution_id").agg({"mandatory_sufficiency_pct":"mean","overall_document_sufficiency_pct":"mean"}).reset_index()
        heat = agg.pivot_table(index=lambda x: 0, columns="institution_id", values="mandatory_sufficiency_pct")
        # simplified heat: show mandatory suff average across institutions
        insts = agg['institution_id'].tolist()
        mand_vals = agg['mandatory_sufficiency_pct'].tolist()
        fig = px.bar(agg, x="institution_id", y="mandatory_sufficiency_pct", title="Average Mandatory Document Sufficiency (%)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Inspect per-institution document checklist")
        inst_sel = st.selectbox("Institution", df['institution_id'].unique(), key="doc_inst")
        yr_sel = st.selectbox("Year", sorted(df['year'].unique(), reverse=True), key="doc_year")
        docs_for = df_docs[(df_docs['institution_id']==inst_sel) & (df_docs['year']==yr_sel)]
        if docs_for.empty:
            st.info("No document records found for selection.")
        else:
            st.dataframe(docs_for[['document_name','category','submitted']].sort_values(['category','document_name']))

        st.markdown("### RAG Analyzer (optional embeddings)")
        rag = SimpleRAG()
        uploaded = st.file_uploader("Upload documents to analyze (pdf/docx/txt)", accept_multiple_files=True)
        if uploaded:
            if st.button("Run RAG prepare"):
                with st.spinner("Extracting text & building embeddings (if available)..."):
                    rag.prepare(uploaded)
                    st.success(f"Prepared {len(rag.text_chunks)} chunks.")
            if st.text_input("Ask a question about uploaded docs", key="rag_query"):
                q = st.session_state.get("rag_query","")
                if q and st.button("Query RAG", key="rag_query_btn"):
                    if HAS_EMBED and rag.embeddings is not None:
                        res = rag.query(q, topk=5)
                        if not res:
                            st.info("No relevant matches.")
                        for r in res:
                            st.write(f"Score: {r['score']:.4f}")
                            st.write(r['text'][:1000])
                            st.markdown("---")
                    else:
                        st.info("Embeddings not available — showing naive substring matches.")
                        fulltext = " ".join([rag.extract_text_from_file(f) for f in uploaded])
                        matches = [m for m in [fulltext[max(0,i-200):i+200] for i in [fulltext.lower().find(q.lower())]] if m]
                        if matches:
                            for m in matches:
                                st.write(m)
                        else:
                            st.info("No matches found.")

    # Institution Portal
    with tabs[3]:
        st.header("🏫 Institution Portal (Upload metadata & view submissions)")
        user = st.session_state.get('user', None)
        if user is None:
            st.info("Login as an Institution (sidebar) to use portal features.")
        else:
            st.subheader(f"Welcome {user['institution_name']} ({user['institution_id']})")
            uploaded_meta = st.file_uploader("Upload documents (metadata saved only)", accept_multiple_files=True)
            doc_type = st.text_input("Document tag", value="other")
            year = st.number_input("Year", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=int(df['year'].max()))
            if uploaded_meta and st.button("Save metadata"):
                # store metadata in local CSV (append)
                conn_path = DB_PATH
                # use simple CSV to record metadata for portability
                meta_file = os.path.join(DATA_DIR, "uploaded_metadata.csv")
                rows = []
                for f in uploaded_meta:
                    rows.append({"institution_id": user['institution_id'], "year": int(year), "file_name": f.name, "doc_type": doc_type, "uploaded_at": now_iso()})
                if os.path.exists(meta_file):
                    old = pd.read_csv(meta_file)
                    new = pd.DataFrame(rows)
                    combined = pd.concat([old, new], ignore_index=True)
                    combined.to_csv(meta_file, index=False)
                else:
                    pd.DataFrame(rows).to_csv(meta_file, index=False)
                st.success("Metadata saved.")

            # show saved metadata if present
            meta_file = os.path.join(DATA_DIR, "uploaded_metadata.csv")
            if os.path.exists(meta_file):
                saved = pd.read_csv(meta_file)
                my_saved = saved[saved['institution_id'] == user['institution_id']]
                if not my_saved.empty:
                    st.dataframe(my_saved.sort_values('uploaded_at', ascending=False))
                else:
                    st.info("No metadata uploaded by your institution yet.")
            else:
                st.info("No uploads yet.")

    # Reports tab
    with tabs[4]:
        st.header("📄 Reports & Downloads")
        st.markdown("Download CSVs and generate per-institution HTML report with Appendix-1 insights.")
        c1, c2, c3 = st.columns(3)
        with c1:
            if os.path.exists(CSV_TIME_SERIES):
                with open(CSV_TIME_SERIES, "rb") as f:
                    st.download_button("Download time-series CSV", data=f.read(), file_name="institutions_10yrs_20inst.csv", mime="text/csv")
            else:
                st.info("Time-series CSV not found (generate dataset).")
        with c2:
            if os.path.exists(CSV_DOCS):
                with open(CSV_DOCS, "rb") as f:
                    st.download_button("Download documents CSV", data=f.read(), file_name="institution_documents_10yrs_20inst.csv", mime="text/csv")
            else:
                st.info("Docs CSV not found.")
        with c3:
            if os.path.exists(CSV_SUMMARY):
                with open(CSV_SUMMARY, "rb") as f:
                    st.download_button("Download summary CSV", data=f.read(), file_name="institutions_summary.csv", mime="text/csv")
            else:
                st.info("Summary CSV not found.")

        st.markdown("---")
        st.subheader("Per-institution HTML report (Appendix-1 summary)")
        sel_inst = st.selectbox("Select institution for report", df['institution_id'].unique(), key="report_inst")
        inst_df = df[df['institution_id'] == sel_inst].sort_values("year", ascending=False)
        inst_docs = df_docs[df_docs['institution_id'] == sel_inst].sort_values(['year','category'])
        if st.button("Generate HTML report"):
            # build HTML
            meta = {"institution_id": sel_inst, "institution_name": inst_df.iloc[0]['institution_name'], "year": int(inst_df.iloc[0]['year'])}
            html = ["<html><head><meta charset='utf-8'><title>Report</title></head><body>"]
            html.append(f"<h1>Report: {meta['institution_name']} ({meta['institution_id']})</h1>")
            html.append("<h2>Last 3 years summary</h2><table border='1'><tr><th>Year</th><th>Overall Score</th><th>Input</th><th>Process</th><th>Outcome</th><th>Impact</th><th>Docs Mand. %</th></tr>")
            for _, r in inst_df.head(3).iterrows():
                html.append(f"<tr><td>{int(r['year'])}</td><td>{r['overall_score']}</td><td>{r['input_score']}</td><td>{r['process_score']}</td><td>{r['outcome_score']}</td><td>{r['impact_score']}</td><td>{r['mandatory_sufficiency_pct']}</td></tr>")
            html.append("</table>")
            html.append("<h3>AI Insights (latest year)</h3>")
            insights = generate_ai_insights_from_row(inst_df.iloc[0].to_dict())
            html.append("<ul>")
            for s in insights["strengths"]:
                html.append(f"<li><b>Strength:</b> {s}</li>")
            for w in insights["weaknesses"]:
                html.append(f"<li><b>Weakness:</b> {w}</li>")
            for rec in insights["recommendations"]:
                html.append(f"<li><b>Recommendation:</b> {rec}</li>")
            html.append("</ul>")
            html.append("<h3>Document checklist (latest year)</h3>")
            latest_year = int(inst_docs['year'].max()) if not inst_docs.empty else None
            if latest_year:
                html.append(f"<p>Year: {latest_year}</p><table border='1'><tr><th>Document</th><th>Category</th><th>Submitted</th></tr>")
                for _, d in inst_docs[inst_docs['year'] == latest_year].iterrows():
                    html.append(f"<tr><td>{d['document_name']}</td><td>{d['category']}</td><td>{'Yes' if d['submitted'] else 'No'}</td></tr>")
                html.append("</table>")
            html.append("</body></html>")
            html_bytes = "\n".join(html).encode("utf-8")
            st.download_button("Download HTML report", data=html_bytes, file_name=f"{sel_inst}_report.html", mime="text/html")
            st.success("Report generated.")

    st.markdown("---")
    st.info("This upgraded app integrates Appendix-1 and produces analytic outputs suitable for a Smart India Hackathon demo. If you want the UI to match exactly the original SIH-app_Final.py components (icons, wording), provide the specific code snippets and I will integrate them into this single-file version.")

if __name__ == "__main__":
    main()
