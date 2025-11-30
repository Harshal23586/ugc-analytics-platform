# app_optimized.py
"""
Optimized single-file Streamlit app:
- Institutional approval system with RAG-based doc extraction (simple)
- SQLite-backed sample data
- Streamlined UI (login, upload, RAG analysis, dashboards)
- Caching for embeddings & heavy ops
- Single-file, readable structure
"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import io
import os

# Optional heavy deps
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

# Visualization
import plotly.express as px

# -----------------------------
# Utility & Config
# -----------------------------
st.set_page_config(page_title="AI-Powered Institutional Approval (Optimized)", layout="wide", page_icon="🏛️")

DB_PATH = "institutions.db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------
# Simple helpers
# -----------------------------
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def safe_get(d: dict, k: str, default=None):
    return d.get(k, default)

# -----------------------------
# Caching: create DB connection once
# -----------------------------
@st.cache_resource
def get_db_conn(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# -----------------------------
# Embedding model manager (cached)
# -----------------------------
@st.cache_resource
def load_embedding_model(model_name: str = EMBED_MODEL_NAME):
    if SentenceTransformer is None:
        return None
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.warning(f"Embedding model load failed: {e}")
        return None

# -----------------------------
# Simple Vector Store (in-memory)
# -----------------------------
class SimpleVectorStore:
    def __init__(self, texts: List[str] = None, embeddings: np.ndarray = None):
        self.texts = texts or []
        self.embeddings = np.array(embeddings) if embeddings is not None else np.array([])

    def build(self, texts: List[str], embeddings: np.ndarray):
        self.texts = texts
        self.embeddings = np.array(embeddings)

    def is_empty(self):
        return self.embeddings.size == 0

    def query(self, model, query_text: str, k: int = 5):
        if self.is_empty() or model is None:
            return []
        q_emb = model.encode([query_text])
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        idxs = np.argsort(sims)[-k:][::-1]
        results = []
        for i in idxs:
            if sims[i] > 0:
                results.append({"text": self.texts[i], "score": float(sims[i])})
        return results

# -----------------------------
# RAG extractor simplified (text extraction + optional vector store)
# -----------------------------
class RAGExtractor:
    def __init__(self, model: Optional[SentenceTransformer] = None):
        self.model = model
        self.texts: List[str] = []
        self.vector_store = SimpleVectorStore()

    def extract_text_from_file(self, uploaded_file) -> str:
        # file-like object
        name = uploaded_file.name.lower()
        try:
            if name.endswith(".pdf") and PyPDF2:
                reader = PyPDF2.PdfReader(uploaded_file)
                pages = []
                for p in reader.pages:
                    txt = p.extract_text()
                    if txt:
                        pages.append(txt)
                return "\n".join(pages)
            elif (name.endswith(".docx") or name.endswith(".doc")) and docx:
                doc = docx.Document(uploaded_file)
                return "\n".join([p.text for p in doc.paragraphs])
            elif name.endswith(".txt"):
                raw = uploaded_file.getvalue()
                try:
                    return raw.decode("utf-8")
                except Exception:
                    return raw.decode("latin-1", errors="ignore")
            elif name.endswith((".xlsx", ".xls")):
                # convert sheets to csv-like text
                try:
                    df = pd.read_excel(uploaded_file)
                    return df.to_string()
                except Exception:
                    return ""
            else:
                # unknown binary - try decode
                try:
                    return uploaded_file.getvalue().decode("utf-8", errors="ignore")
                except Exception:
                    return ""
        except Exception as e:
            st.error(f"Failed to extract {uploaded_file.name}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def split_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if not text:
            return []
        sentences = re.split(r'(?<=[\.\?\!])\s+', text)
        chunks = []
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 > chunk_size:
                chunks.append(current.strip())
                # overlap last N chars
                if overlap > 0:
                    current = current[-overlap:] + " " + s
                else:
                    current = s
            else:
                current = current + " " + s if current else s
        if current:
            chunks.append(current.strip())
        return chunks

    def prepare_documents(self, files: List) -> List[str]:
        texts = []
        for f in files:
            raw = self.extract_text_from_file(f)
            cleaned = self.clean_text(raw)
            if cleaned:
                chunks = self.split_into_chunks(cleaned)
                texts.extend(chunks if chunks else [cleaned])
        self.texts = texts
        return texts

    def build_vector_store(self, batch_size: int = 32):
        if self.model is None or not self.texts:
            return
        # batch encode
        embeddings = []
        for i in range(0, len(self.texts), batch_size):
            batch = self.texts[i:i+batch_size]
            try:
                emb = self.model.encode(batch)
                embeddings.extend(emb)
            except Exception as e:
                st.warning(f"Batch embedding failed: {e}")
                return
        self.vector_store.build(self.texts, np.vstack(embeddings))

    def query(self, query_text: str, k: int = 5):
        return self.vector_store.query(self.model, query_text, k)

# -----------------------------
# Analyzer: scoring & DB
# -----------------------------
class InstitutionalAIAnalyzer:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.ensure_schema()
        self.historical_data = self.load_or_generate_sample_data()
        self.embedding_model = load_embedding_model()
        # RAG extractor per session
        self.rag = RAGExtractor(self.embedding_model)

    def ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS institutions (
            id INTEGER PRIMARY KEY,
            institution_id TEXT UNIQUE,
            institution_name TEXT,
            year INTEGER,
            institution_type TEXT,
            heritage_category TEXT,
            state TEXT,
            established_year INTEGER,
            overall_score REAL,
            approval_recommendation TEXT,
            risk_level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        cur.execute('''
        CREATE TABLE IF NOT EXISTS institution_users (
            id INTEGER PRIMARY KEY,
            institution_id TEXT,
            username TEXT UNIQUE,
            password_hash TEXT,
            contact_person TEXT,
            email TEXT,
            phone TEXT,
            role TEXT DEFAULT 'Institution',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        cur.execute('''
        CREATE TABLE IF NOT EXISTS rag_analysis (
            id INTEGER PRIMARY KEY,
            institution_id TEXT,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            extracted_data TEXT,
            ai_insights TEXT,
            confidence_score REAL
        )''')
        cur.execute('''
        CREATE TABLE IF NOT EXISTS institution_documents (
            id INTEGER PRIMARY KEY,
            institution_id TEXT,
            document_name TEXT,
            document_type TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'Uploaded'
        )''')
        self.conn.commit()

    def seed_system_users(self):
        # Create some system users if missing (non-sensitive simple seed)
        cur = self.conn.cursor()
        seeds = [
            ("ugc_officer", "ugc123", "UGC Officer", "ugc.officer@ugc.gov.in"),
            ("aicte_officer", "aicte123", "AICTE Officer", "aicte.officer@aicte.gov.in"),
        ]
        for username, pwd, role, email in seeds:
            cur.execute("SELECT * FROM institution_users WHERE username = ?", (username,))
            if cur.fetchone() is None:
                cur.execute('''
                    INSERT INTO institution_users (institution_id, username, password_hash, contact_person, email, phone, role)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', ("SYSTEM", username, sha256(pwd), role + " Contact", email, "", "System"))
        self.conn.commit()

    def load_or_generate_sample_data(self) -> pd.DataFrame:
        # try load; if empty, generate sample dataset for 20 institutions and multiple years
        try:
            df = pd.read_sql('SELECT * FROM institutions', self.conn)
            if len(df) > 0:
                return df
        except Exception:
            pass
        # generate compact sample
        np.random.seed(42)
        base_insts = [
            ("HEI_01", "IIT Varanasi", "Multi-disciplinary", "Old"),
            ("HEI_02", "NIT Srinagar", "Research-Intensive", "Old"),
            ("HEI_03", "State University Bengaluru", "Teaching-Intensive", "Old"),
            ("HEI_04", "National Law School Delhi", "Specialised", "New"),
            ("HEI_05", "NSDI Pune", "Vocational", "New"),
            ("HEI_06", "RGU Community Health", "Community", "New"),
            ("HEI_07", "Himalayan Rural Institute", "Rural", "Old"),
        ]
        rows = []
        for inst_id, name, itype, heritage in base_insts:
            for y in range(2018, 2024):
                overall = round(np.clip(np.random.normal(7.0, 1.0), 3.0, 9.5), 2)
                rows.append({
                    "institution_id": inst_id,
                    "institution_name": name,
                    "year": y,
                    "institution_type": itype,
                    "heritage_category": heritage,
                    "state": "State",
                    "established_year": 1990,
                    "overall_score": overall,
                    "approval_recommendation": self.generate_approval_recommendation(overall),
                    "risk_level": self.assess_risk_level(overall)
                })
        df = pd.DataFrame(rows)
        df.to_sql('institutions', self.conn, if_exists='replace', index=False)
        return df

    # scoring helpers
    def generate_approval_recommendation(self, score: float) -> str:
        if score >= 8.0:
            return "Full Approval - 5 Years"
        elif score >= 7.0:
            return "Provisional Approval - 3 Years"
        elif score >= 6.0:
            return "Conditional Approval - 1 Year"
        elif score >= 5.0:
            return "Approval w/ Monitoring - 1 Year"
        else:
            return "Rejection"

    def assess_risk_level(self, score: float) -> str:
        if score >= 8.0:
            return "Low Risk"
        elif score >= 6.5:
            return "Medium Risk"
        elif score >= 5.0:
            return "High Risk"
        else:
            return "Critical Risk"

    # user management
    def create_institution_user(self, institution_id: str, username: str, password: str, contact_person: str, email: str, phone: str) -> bool:
        cur = self.conn.cursor()
        try:
            cur.execute('''
            INSERT INTO institution_users (institution_id, username, password_hash, contact_person, email, phone)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (institution_id, username, sha256(password), contact_person, email, phone))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def authenticate_institution_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute('''
        SELECT iu.*, i.institution_name FROM institution_users iu
        LEFT JOIN institutions i ON iu.institution_id = i.institution_id
        WHERE iu.username = ? AND iu.is_active = 1
        ''', (username,))
        row = cur.fetchone()
        if row:
            rowd = dict(row)
            if rowd.get("password_hash") == sha256(password):
                return {
                    "institution_id": rowd.get("institution_id"),
                    "username": rowd.get("username"),
                    "contact_person": rowd.get("contact_person"),
                    "email": rowd.get("email"),
                    "institution_name": rowd.get("institution_name") or rowd.get("institution_id")
                }
        return None

    # document saving
    def save_uploaded_documents(self, institution_id: str, uploaded_files: List, document_types: List[str]):
        cur = self.conn.cursor()
        for f, dtype in zip(uploaded_files, document_types):
            cur.execute('''
            INSERT INTO institution_documents (institution_id, document_name, document_type)
            VALUES (?, ?, ?)
            ''', (institution_id, f.name, dtype))
        self.conn.commit()

    def get_institution_documents(self, institution_id: str) -> pd.DataFrame:
        try:
            return pd.read_sql('SELECT * FROM institution_documents WHERE institution_id = ? ORDER BY upload_date DESC', self.conn, params=(institution_id,))
        except Exception:
            return pd.DataFrame([])

    # RAG analysis orchestration (simple)
    def analyze_documents_with_rag(self, institution_id: str, uploaded_files: List) -> Dict[str, Any]:
        # prepare texts
        texts = self.rag.prepare_documents(uploaded_files)
        if not texts:
            return {"status": "Failed", "message": "No text extracted from files."}

        # build vectors if possible
        if self.embedding_model:
            self.rag.build_vector_store()
        else:
            st.warning("Embedding model not available. RAG similarity disabled.")

        # simple structured extraction using regex heuristics
        combined = " ".join(texts)
        extracted = self.extract_structured_data(combined)

        # insights
        insights = self.generate_ai_insights(extracted)

        # persist rag_analysis record
        cur = self.conn.cursor()
        cur.execute('''
        INSERT INTO rag_analysis (institution_id, extracted_data, ai_insights, confidence_score)
        VALUES (?, ?, ?, ?)
        ''', (institution_id, json.dumps(extracted), json.dumps(insights), 0.85))
        self.conn.commit()

        return {"status": "Analysis Complete", "extracted_data": extracted, "ai_insights": insights, "confidence_score": 0.85}

    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        # minimal set of regex-based extractions for demo purposes
        data = {
            "academic_metrics": {},
            "research_metrics": {},
            "infrastructure_metrics": {},
            "governance_metrics": {},
            "financial_metrics": {},
            "raw_text": text[:5000]
        }
        # examples
        m = re.search(r'naac.*grade[:\s]*([A\+\-]+)', text, re.IGNORECASE)
        if m:
            data["academic_metrics"]["naac_grade"] = m.group(1)

        m = re.search(r'research.*publications[:\s]*([0-9]+)', text, re.IGNORECASE)
        if m:
            data["research_metrics"]["research_publications"] = int(m.group(1))

        m = re.search(r'patents.*filed[:\s]*([0-9]+)', text, re.IGNORECASE)
        if m:
            data["research_metrics"]["patents_filed"] = int(m.group(1))

        m = re.search(r'financial.*stability[:\s]*([0-9]+(?:\.[0-9]+)?)', text, re.IGNORECASE)
        if m:
            data["financial_metrics"]["financial_stability_score"] = float(m.group(1))

        # fallback heuristics (counts)
        data["research_metrics"].setdefault("research_publications", int(re.findall(r'\b\d+\s+publications\b', text, re.IGNORECASE).__len__()))
        return data

    def generate_ai_insights(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        insights = {"strengths": [], "weaknesses": [], "recommendations": [], "risk_assessment": {}}
        academic = extracted_data.get("academic_metrics", {})
        research = extracted_data.get("research_metrics", {})
        financial = extracted_data.get("financial_metrics", {})

        # simple rules
        naac = academic.get("naac_grade")
        if naac in ("A++", "A+", "A"):
            insights["strengths"].append(f"Strong NAAC accreditation: {naac}")
        pubs = research.get("research_publications", 0)
        if pubs and pubs > 50:
            insights["strengths"].append("Robust research publication output")
        if pubs and pubs < 5:
            insights["weaknesses"].append("Low publication count")

        patents = research.get("patents_filed", 0)
        if patents < 2:
            insights["recommendations"].append("Increase patenting and IPR activities")

        fin = financial.get("financial_stability_score", 0)
        if fin and fin < 6:
            insights["weaknesses"].append("Financial stability score is low")
            insights["recommendations"].append("Improve financial planning")

        risk_score = 5.0
        if naac in ("A++", "A+", "A"):
            risk_score -= 1.5
        if pubs > 30:
            risk_score -= 1.0
        if patents < 2:
            risk_score += 1.0
        level = "Low" if risk_score < 4 else "Medium" if risk_score < 7 else "High"
        insights["risk_assessment"] = {"score": round(risk_score, 2), "level": level, "factors": []}
        return insights

# -----------------------------
# Streamlit UI: pages and components
# -----------------------------
def sidebar_login_ui(analyzer: InstitutionalAIAnalyzer):
    st.sidebar.title("🔐 Login / Register")
    mode = st.sidebar.radio("Mode", ["Institution Login", "Register"])
    if mode == "Institution Login":
        user = st.sidebar.text_input("Username", key="ui_login_username")
        pwd = st.sidebar.text_input("Password", type="password", key="ui_login_password")
        if st.sidebar.button("Login"):
            auth = analyzer.authenticate_institution_user(user, pwd)
            if auth:
                st.session_state['institution_user'] = auth
                st.success(f"Welcome {auth.get('contact_person')}")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.sidebar.markdown("Create a new institution user")
        inst = st.sidebar.selectbox("Institution ID (sample)", analyzer.historical_data['institution_id'].unique())
        username = st.sidebar.text_input("Username (new)", key="ui_reg_username")
        pwd = st.sidebar.text_input("Password (new)", type="password", key="ui_reg_password")
        confirm = st.sidebar.text_input("Confirm Password", type="password", key="ui_reg_confirm")
        contact = st.sidebar.text_input("Contact Person", key="ui_reg_contact")
        email = st.sidebar.text_input("Email", key="ui_reg_email")
        phone = st.sidebar.text_input("Phone", key="ui_reg_phone")
        if st.sidebar.button("Register"):
            if not username or not pwd or not contact or not email:
                st.sidebar.error("Fill required fields")
            elif pwd != confirm:
                st.sidebar.error("Passwords do not match")
            else:
                ok = analyzer.create_institution_user(inst, username, pwd, contact, email, phone)
                if ok:
                    st.sidebar.success("User created. Please login.")
                else:
                    st.sidebar.error("Username exists. Pick another.")

def main_dashboard(analyzer: InstitutionalAIAnalyzer):
    st.title("🏛️ AI-Powered Institutional Approval System (Optimized)")
    st.markdown("A simplified and optimized single-file version of your institutional approval app.")

    # top-level tabs
    tabs = st.tabs(["Dashboard", "RAG Analyzer", "My Institution", "System"])
    with tabs[0]:
        performance_dashboard_ui(analyzer)
    with tabs[1]:
        rag_analyzer_ui(analyzer)
    with tabs[2]:
        institution_portal_ui(analyzer)
    with tabs[3]:
        system_ui(analyzer)

# Dashboard UI
def performance_dashboard_ui(analyzer: InstitutionalAIAnalyzer):
    st.header("📊 Institutional Performance Dashboard")
    df = analyzer.historical_data
    if df.empty:
        st.info("No data available.")
        return
    cur_year = st.selectbox("Select Year", sorted(df['year'].unique(), reverse=True), index=0)
    df_year = df[df['year'] == cur_year]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Institutions", df_year['institution_id'].nunique())
    with col2:
        st.metric("Avg Score", f"{df_year['overall_score'].mean():.2f}/10")
    with col3:
        st.metric("High Risk Count", int((df_year['overall_score'] < 6.0).sum()))
    with col4:
        st.metric("Max Score", f"{df_year['overall_score'].max():.2f}/10")
    # histogram
    fig = px.histogram(df_year, x="overall_score", nbins=15, title=f"Score Distribution ({cur_year})")
    st.plotly_chart(fig, use_container_width=True)
    # top institutions
    st.subheader("Top Institutions")
    top = df_year.nlargest(10, "overall_score")[['institution_id', 'institution_name', 'overall_score', 'approval_recommendation']]
    st.dataframe(top.reset_index(drop=True))

# RAG analyzer UI
def rag_analyzer_ui(analyzer: InstitutionalAIAnalyzer):
    st.header("🤖 RAG Document Analyzer")
    inst_id = st.selectbox("Select Institution", analyzer.historical_data['institution_id'].unique())
    uploaded_files = st.file_uploader("Upload Documents (pdf, docx, txt, xlsx)", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'xlsx'])
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")
        # show file names
        for f in uploaded_files:
            st.write(f"- {f.name}")
        # assign types quickly: default "other"
        if st.button("Start RAG Analysis"):
            with st.spinner("Extracting & analyzing..."):
                # Save file metadata to DB
                analyzer.save_uploaded_documents(inst_id, uploaded_files, ["uploaded"]*len(uploaded_files))
                result = analyzer.analyze_documents_with_rag(inst_id, uploaded_files)
                if result.get("status") == "Analysis Complete":
                    st.success("RAG Analysis Complete")
                    # show extracted pieces
                    extracted = result.get("extracted_data", {})
                    st.subheader("Extracted Metrics (Preview)")
                    for k, v in extracted.items():
                        if k == "raw_text":
                            st.text_area("Raw Text Preview", v[:4000], height=200)
                        else:
                            st.write(f"**{k}**")
                            st.json(v)
                    st.subheader("AI Insights")
                    st.json(result.get("ai_insights", {}))
                else:
                    st.error("RAG analysis failed: " + str(result.get("message", "Unknown")))

    else:
        st.info("Upload one or more documents to run RAG analysis.")

# Institution portal UI
def institution_portal_ui(analyzer: InstitutionalAIAnalyzer):
    user = st.session_state.get("institution_user")
    if not user:
        st.info("Please login/create a user in the sidebar to access your institution portal.")
        return
    st.header(f"🏛️ Institution Portal — {user.get('institution_name')}")
    inst_id = user.get("institution_id")
    tabs = st.tabs(["Documents", "Submissions", "Profile"])
    with tabs[0]:
        st.subheader("Uploaded Documents")
        df_docs = analyzer.get_institution_documents(inst_id)
        if df_docs.empty:
            st.info("No documents uploaded.")
        else:
            st.dataframe(df_docs[['document_name', 'document_type', 'upload_date', 'status']])
        # quick upload
        upl = st.file_uploader("Upload more docs", accept_multiple_files=True, type=['pdf','docx','txt','xlsx'])
        if upl:
            types = ["other"] * len(upl)
            if st.button("Upload Documents for Institution"):
                analyzer.save_uploaded_documents(inst_id, upl, types)
                st.success("Uploaded.")
    with tabs[1]:
        st.subheader("Submissions")
        subs = pd.read_sql('SELECT * FROM rag_analysis WHERE institution_id = ? ORDER BY analysis_date DESC', analyzer.conn, params=(inst_id,))
        if subs.empty:
            st.info("No previous analyses.")
        else:
            st.dataframe(subs[['analysis_date', 'confidence_score']])
            if st.checkbox("Show last analysis details"):
                last = subs.iloc[0]
                st.json({"extracted_data": json.loads(last['extracted_data']), "ai_insights": json.loads(last['ai_insights'])})
    with tabs[2]:
        st.subheader("Profile")
        st.write(f"Contact: {user.get('contact_person')}")
        st.write(f"Email: {user.get('email')}")
        if st.button("Logout"):
            st.session_state.pop("institution_user", None)
            st.experimental_rerun()

# System admin UI (read-only)
def system_ui(analyzer: InstitutionalAIAnalyzer):
    st.header("⚙️ System Overview")
    st.subheader("DB Summary")
    df = analyzer.historical_data
    st.metric("Institutions in DB", df['institution_id'].nunique())
    st.metric("Total records", len(df))
    st.write("System embedding model available:", bool(analyzer.embedding_model))
    if st.button("Force reload embedding model"):
        # clear cache (simple trick)
        st.cache_resource.clear()
        st.experimental_rerun()

# -----------------------------
# Main
# -----------------------------
def main():
    conn = get_db_conn()
    analyzer = InstitutionalAIAnalyzer(conn)
    analyzer.seed_system_users()

    # sidebar login / register
    sidebar_login_ui(analyzer)

    # main dashboard and pages
    main_dashboard(analyzer)

if __name__ == "__main__":
    main()
