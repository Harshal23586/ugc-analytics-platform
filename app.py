import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Additional imports for enhanced functionality
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Any, Optional
import hashlib
import sqlite3
import io
import base64
import os
import tempfile
from pathlib import Path

# RAG-specific imports
import PyPDF2
import docx
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add these instead:
import re
from typing import List

# Initialize session state at module level
if 'session_initialized' not in st.session_state:
    st.session_state.session_initialized = True
    st.session_state.institution_user = None
    st.session_state.user_role = None
    st.session_state.rag_analysis = None
    st.session_state.selected_institution = None
    st.session_state.system_user = None

# Page configuration with enhanced UI
st.set_page_config(
    page_title="AI-Powered Institutional Approval System - UGC/AICTE",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS from app.py
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .accredited { color: #28a745; font-weight: bold; }
    .awaiting { color: #ffc107; font-weight: bold; }
    .not-accredited { color: #dc3545; font-weight: bold; }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# RAG and Analyzer classes
class RAGDocument:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Simple text splitter that splits by sentences and chunks"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                if self.chunk_overlap > 0:
                    overlap_sentences = current_chunk.split('.')[-3:]
                    current_chunk = '.'.join(overlap_sentences) + '. ' + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += '. ' + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class SimpleVectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = []
    
    def from_embeddings(self, text_embeddings):
        """Create vector store from text-embedding pairs"""
        texts, embeddings = zip(*text_embeddings)
        self.documents = list(texts)
        self.embeddings = np.array(embeddings)
        return self
    
    def similarity_search_with_score(self, query: str, k: int = 5):
        """Simple similarity search using cosine similarity"""
        if not self.embeddings.size:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[-k:][::-1]
        results = []
        
        for idx in top_indices:
            if similarities[idx] > 0:
                doc = RAGDocument(
                    page_content=self.documents[idx],
                    metadata={"similarity_score": float(similarities[idx])}
                )
                results.append((doc, float(similarities[idx])))
        
        return results

class RAGDataExtractor:
    def __init__(self):
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.text_splitter = SimpleTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.vector_store = None
            self.documents = []
        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")
            self.embedding_model = None
            self.text_splitter = SimpleTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.vector_store = None
            self.documents = []

    def build_vector_store(self, documents: List[RAGDocument]):
        """Build simple vector store from documents"""
        if not documents or self.embedding_model is None:
            return None
        
        try:
            texts = [doc.page_content for doc in documents]
            if not texts:
                return None
            
            embeddings = self.embedding_model.encode(texts)
            text_embeddings = list(zip(texts, embeddings))
            self.vector_store = SimpleVectorStore(self.embedding_model).from_embeddings(text_embeddings)
            self.documents = documents
        except Exception as e:
            st.error(f"Error building vector store: {e}")
            return None
        
    def extract_text_from_file(self, file) -> str:
        """Extract text from various file formats"""
        text = ""
        file_extension = file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'pdf':
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    
            elif file_extension in ['doc', 'docx']:
                doc = docx.Document(file)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                    
            elif file_extension in ['txt']:
                text = file.getvalue().decode('utf-8')
                
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file)
                text = df.to_string()
                
        except Exception as e:
            st.error(f"Error extracting text from {file.name}: {str(e)}")
            
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text.strip()
    
    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from text using pattern matching"""
        data = {
            'academic_metrics': {},
            'research_metrics': {},
            'infrastructure_metrics': {},
            'governance_metrics': {},
            'student_metrics': {},
            'financial_metrics': {}
        }
        
        academic_patterns = {
            'naac_grade': r'NAAC\s*(?:grade|accreditation|score)[:\s]*([A+]+)',
            'nirf_ranking': r'NIRF\s*(?:rank|ranking)[:\s]*(\d+)',
            'student_faculty_ratio': r'(?:student|student-faculty)\s*(?:ratio|ratio:)[:\s]*(\d+(?:\.\d+)?)',
            'phd_faculty_ratio': r'PhD\s*(?:faculty|faculty ratio)[:\s]*(\d+(?:\.\d+)?)%?',
            'placement_rate': r'placement\s*(?:rate|percentage)[:\s]*(\d+(?:\.\d+)?)%?'
        }
        
        research_patterns = {
            'research_publications': r'research\s*(?:publications|papers)[:\s]*(\d+)',
            'research_grants': r'research\s*(?:grants|funding)[:\s]*[₹$]?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            'patents_filed': r'patents?\s*(?:filed|granted)[:\s]*(\d+)',
            'industry_collaborations': r'industry\s*(?:collaborations|partnerships)[:\s]*(\d+)'
        }
        
        for key, pattern in academic_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['academic_metrics'][key] = match.group(1)
        
        for key, pattern in research_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['research_metrics'][key] = match.group(1)
        
        self.extract_contextual_data(text, data)
        
        return data
    
    def extract_contextual_data(self, text: str, data: Dict):
        """Extract data based on contextual patterns"""
        patterns = [
            (r'library.*?(\d+(?:,\d+)*)\s*(?:volumes|books)', 'library_volumes'),
            (r'campus.*?(\d+(?:\.\d+)?)\s*(?:acres|hectares)', 'campus_area'),
            (r'financial.*?stability.*?(\d+(?:\.\d+)?)\s*(?:out of|/)', 'financial_stability_score'),
            (r'digital.*?infrastructure.*?(\d+(?:\.\d+)?)\s*(?:out of|/)', 'digital_infrastructure_score'),
            (r'community.*?projects.*?(\d+)', 'community_projects')
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if 'infrastructure' in key:
                    data['infrastructure_metrics'][key] = match.group(1)
                elif 'financial' in key:
                    data['financial_metrics'][key] = match.group(1)
                else:
                    data['governance_metrics'][key] = match.group(1)
    
    def query_documents(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Query documents using semantic search"""
        if not self.vector_store:
            return []
            
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def extract_comprehensive_data(self, uploaded_files: List) -> Dict[str, Any]:
        """Extract comprehensive data from all uploaded files"""
        all_text = ""
        all_structured_data = {
            'academic_metrics': {},
            'research_metrics': {},
            'infrastructure_metrics': {},
            'governance_metrics': {},
            'student_metrics': {},
            'financial_metrics': {},
            'raw_text': "",
            'file_names': []
        }
    
        documents = []
    
        for file in uploaded_files:
            try:
                text = self.extract_text_from_file(file)
                cleaned_text = self.preprocess_text(text)
                all_text += cleaned_text + "\n\n"
            
                doc = RAGDocument(
                    page_content=cleaned_text,
                    metadata={"source": file.name, "type": "institutional_data"}
                )
                documents.append(doc)
            
                file_data = self.extract_structured_data(cleaned_text)
            
                for category in file_data:
                    if category in all_structured_data:
                        all_structured_data[category].update(file_data[category])
            
                all_structured_data['file_names'].append(file.name)
            
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                continue
    
        if documents and self.embedding_model is not None:
            try:
                self.build_vector_store(documents)
            except Exception as e:
                st.warning(f"Vector store creation skipped: {e}")
    
        all_structured_data['raw_text'] = all_text
    
        return all_structured_data

class AccreditationAnalyzer:
    """Enhanced analyzer with beautiful UI components from app.py"""
    def __init__(self):
        self.parameters = [
            "Curriculum", "Faculty Resources", "Learning and Teaching", 
            "Research and Innovation", "Extracurricular Activities",
            "Community Engagement", "Green Initiatives", 
            "Governance and Administration", "Infrastructure Development",
            "Financial Resources"
        ]
        
    def calculate_parameter_scores(self, institution_data):
        """Calculate scores for each parameter based on institutional data"""
        scores = {}
        
        # Academic Excellence Score
        scores['Curriculum'] = (
            institution_data.get('placement_rate', 75) * 0.4 +
            (100 - institution_data.get('dropout_rate', 5)) * 0.3 +
            institution_data.get('pass_percentage', 80) * 0.3
        )
        
        # Faculty Resources Score
        faculty_ratio = institution_data.get('student_faculty_ratio', 20)
        scores['Faculty Resources'] = (
            min((1/faculty_ratio) * 1000, 100) * 0.4 +
            institution_data.get('phd_faculty_ratio', 0.7) * 100 * 0.4 +
            min(institution_data.get('research_publications', 50) / 2, 100) * 0.2
        )
        
        # Learning and Teaching Score
        scores['Learning and Teaching'] = (
            institution_data.get('pass_percentage', 80) * 0.5 +
            (100 - institution_data.get('dropout_rate', 5)) * 0.3 +
            min(institution_data.get('student_strength', 2000) / 50, 100) * 0.2
        )
        
        # Research and Innovation Score
        scores['Research and Innovation'] = (
            min(institution_data.get('research_publications', 50) * 0.5, 100) * 0.6 +
            min(institution_data.get('research_grants_amount', 1000000) / 20000, 100) * 0.4
        )
        
        # Infrastructure Development
        scores['Infrastructure Development'] = (
            institution_data.get('digital_infrastructure_score', 7) * 10 * 0.4 +
            min(institution_data.get('library_volumes', 20000) / 500, 100) * 0.3 +
            institution_data.get('laboratory_equipment_score', 7) * 10 * 0.3
        )
        
        # Governance and Administration
        scores['Governance and Administration'] = (
            institution_data.get('financial_stability_score', 8) * 10 * 0.4 +
            institution_data.get('compliance_score', 8) * 10 * 0.3 +
            institution_data.get('administrative_efficiency', 7) * 10 * 0.3
        )
        
        # Other parameters
        for param in self.parameters[6:]:
            base_score = np.random.uniform(65, 85)
            trend_factor = institution_data.get('performance_score', 7) * 2
            scores[param] = min(max(base_score + trend_factor, 0), 100)
        
        return scores
    
    def predict_accreditation_status(self, scores):
        """Predict accreditation status based on parameter scores"""
        overall_score = np.mean(list(scores.values()))
        
        if overall_score >= 80:
            status = "Accredited"
            status_class = "accredited"
        elif overall_score >= 60:
            status = "Awaiting Accreditation"
            status_class = "awaiting"
        else:
            status = "Not Accredited"
            status_class = "not-accredited"
        
        return overall_score, status, status_class
    
    def assess_maturity_level(self, scores):
        """Assess maturity level (1-5) based on scores"""
        overall_score = np.mean(list(scores.values()))
        
        if overall_score >= 90:
            return 5, "Global Standards"
        elif overall_score >= 80:
            return 4, "National Excellence"
        elif overall_score >= 70:
            return 3, "Excellence in Some Areas"
        elif overall_score >= 60:
            return 2, "Good Practices"
        else:
            return 1, "Basic Compliance"

    def create_accreditation_dashboard(self, institution_data, institution_name):
        """Create beautiful accreditation dashboard for an institution"""
        st.markdown(f'<div class="section-header">🎓 {institution_name} - Accreditation Analysis</div>', unsafe_allow_html=True)
        
        # Calculate scores
        scores = self.calculate_parameter_scores(institution_data)
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        overall_score, status, status_class = self.predict_accreditation_status(scores)
        maturity_level, maturity_desc = self.assess_maturity_level(scores)
        
        with col1:
            st.metric("Overall Score", f"{overall_score:.1f}/100")
        with col2:
            st.markdown(f'<div class="metric-card">Accreditation Status: <span class="{status_class}">{status}</span></div>', unsafe_allow_html=True)
        with col3:
            st.metric("Maturity Level", f"Level {maturity_level}")
        with col4:
            st.metric("Maturity Description", maturity_desc)
        
        # Parameter scores visualization
        st.markdown('<div class="section-header">Parameter-wise Performance</div>', unsafe_allow_html=True)
        
        fig = go.Figure(data=[
            go.Bar(name='Scores', x=list(scores.keys()), y=list(scores.values()),
                  marker_color=['#1f77b4' if x >= 80 else '#ff7f0e' if x >= 60 else '#d62728' for x in scores.values()])
        ])
        fig.update_layout(
            title='Parameter-wise Performance Scores',
            xaxis_title='Parameters',
            yaxis_title='Score (0-100)',
            yaxis_range=[0, 100],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return scores, overall_score, status

class InstitutionalAIAnalyzer:
    def __init__(self):
        self.init_database()
        self.historical_data = self.load_or_generate_data()
        self.performance_metrics = self.define_performance_metrics()
        self.document_requirements = self.define_document_requirements()
        self.rag_extractor = RAGDataExtractor()
        self.accreditation_analyzer = AccreditationAnalyzer()
        self.create_dummy_institution_users()
        self.create_dummy_system_users()

    def init_database(self):
        """Initialize SQLite database for storing institutional data"""
        self.conn = sqlite3.connect('institutions.db', check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()

        # Create institutions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS institutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                institution_id TEXT UNIQUE,
                institution_name TEXT,
                year INTEGER,
                institution_type TEXT,
                state TEXT,
                established_year INTEGER,
                naac_grade TEXT,
                nirf_ranking INTEGER,
                student_faculty_ratio REAL,
                phd_faculty_ratio REAL,
                research_publications INTEGER,
                research_grants_amount REAL,
                patents_filed INTEGER,
                industry_collaborations INTEGER,
                digital_infrastructure_score REAL,
                library_volumes INTEGER,
                laboratory_equipment_score REAL,
                financial_stability_score REAL,
                compliance_score REAL,
                administrative_efficiency REAL,
                placement_rate REAL,
                higher_education_rate REAL,
                entrepreneurship_cell_score REAL,
                community_projects INTEGER,
                rural_outreach_score REAL,
                inclusive_education_index REAL,
                rusa_participation INTEGER,
                nmeict_participation INTEGER,
                fist_participation INTEGER,
                dst_participation INTEGER,
                performance_score REAL,
                approval_recommendation TEXT,
                risk_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create other tables (documents, rag_analysis, etc.)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS institution_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                institution_id TEXT,
                document_name TEXT,
                document_type TEXT,
                file_path TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'Pending',
                extracted_data TEXT,
                FOREIGN KEY (institution_id) REFERENCES institutions (institution_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                institution_id TEXT,
                analysis_type TEXT,
                extracted_data TEXT,
                ai_insights TEXT,
                confidence_score REAL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (institution_id) REFERENCES institutions (institution_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS institution_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                institution_id TEXT,
                submission_type TEXT,
                submission_data TEXT,
                status TEXT DEFAULT 'Under Review',
                submitted_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_by TEXT,
                review_date TIMESTAMP,
                review_comments TEXT,
                FOREIGN KEY (institution_id) REFERENCES institutions (institution_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS institution_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                institution_id TEXT,
                username TEXT UNIQUE,
                password_hash TEXT,
                contact_person TEXT,
                email TEXT,
                phone TEXT,
                role TEXT DEFAULT 'Institution',
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (institution_id) REFERENCES institutions (institution_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password_hash TEXT,
                full_name TEXT,
                email TEXT,
                role TEXT,
                department TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()

    def create_dummy_system_users(self):
        """Create dummy system users for testing"""
        system_users = [
            {
                'username': 'ugc_officer',
                'password': 'ugc123',
                'full_name': 'UGC Department Officer',
                'email': 'ugc.officer@ugc.gov.in',
                'role': 'UGC Officer',
                'department': 'UGC Approval Division'
            },
            {
                'username': 'aicte_officer',
                'password': 'aicte123',
                'full_name': 'AICTE Department Officer',
                'email': 'aicte.officer@aicte.gov.in',
                'role': 'AICTE Officer',
                'department': 'AICTE Approval Division'
            },
            {
                'username': 'system_admin',
                'password': 'admin123',
                'full_name': 'System Administrator',
                'email': 'admin@ugc-aicte.gov.in',
                'role': 'System Admin',
                'department': 'IT Department'
            },
            {
                'username': 'review_committee',
                'password': 'review123',
                'full_name': 'Review Committee Member',
                'email': 'review.committee@ugc-aicte.gov.in',
                'role': 'Review Committee',
                'department': 'Review Committee'
            }
        ]
    
        for user_data in system_users:
            try:
                cursor = self.conn.cursor()
                cursor.execute('SELECT * FROM system_users WHERE username = ?', (user_data['username'],))
                existing_user = cursor.fetchone()
        
                if not existing_user:
                    self.create_system_user(
                        user_data['username'],
                        user_data['password'],
                        user_data['full_name'],
                        user_data['email'],
                        user_data['role'],
                        user_data['department']
                    )
            except Exception as e:
                print(f"Error creating system user {user_data['username']}: {e}")    

    def create_dummy_institution_users(self):
        """Create dummy institution users for testing"""
        dummy_users = [
            {
                'institution_id': 'INST_0001',
                'username': 'inst001_admin',
                'password': 'password123',
                'contact_person': 'Dr. Rajesh Kumar',
                'email': 'rajesh.kumar@university001.edu.in',
                'phone': '+91-9876543210'
            },
            {
                'institution_id': 'INST_0050',
                'username': 'inst050_registrar',
                'password': 'testpass456',
                'contact_person': 'Ms. Priya Sharma',
                'email': 'priya.sharma@college050.edu.in',
                'phone': '+91-8765432109'
            }
        ]
    
        for user_data in dummy_users:
            try:
                cursor = self.conn.cursor()
                cursor.execute('SELECT * FROM institution_users WHERE username = ?', (user_data['username'],))
                existing_user = cursor.fetchone()
            
                if not existing_user:
                    self.create_institution_user(
                        user_data['institution_id'],
                        user_data['username'],
                        user_data['password'],
                        user_data['contact_person'],
                        user_data['email'],
                        user_data['phone']
                    )
            except Exception as e:
                print(f"Error creating user {user_data['username']}: {e}")

    def create_institution_user(self, institution_id: str, username: str, password: str, 
                          contact_person: str, email: str, phone: str):
        """Create new institution user account"""
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO institution_users 
                (institution_id, username, password_hash, contact_person, email, phone)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (institution_id, username, self.hash_password(password), 
                  contact_person, email, phone))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def hash_password(self, password: str) -> str:
        """Simple password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_system_user(self, username: str, password: str, full_name: str, 
                          email: str, role: str, department: str):
        """Create new system user account"""
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO system_users 
                (username, password_hash, full_name, email, role, department)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, self.hash_password(password), full_name, email, role, department))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def authenticate_institution_user(self, username: str, password: str) -> Dict:
        """Authenticate institution user"""
        if not username or not password:
            return None
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT iu.*, i.institution_name 
            FROM institution_users iu 
            JOIN institutions i ON iu.institution_id = i.institution_id 
            WHERE iu.username = ? AND iu.is_active = 1
        ''', (username,))
    
        user = cursor.fetchone()
        if user:
            columns = [description[0] for description in cursor.description]
            user_dict = dict(zip(columns, user))
        
            password_hash = user_dict.get('password_hash')
            if password_hash and password_hash == self.hash_password(password):
                return {
                    'institution_id': user_dict.get('institution_id'),
                    'institution_name': user_dict.get('institution_name'),
                    'username': user_dict.get('username'),
                    'role': user_dict.get('role', 'Institution'),
                    'contact_person': user_dict.get('contact_person', ''),
                    'email': user_dict.get('email', '')
                }
        return None

    def authenticate_system_user(self, username: str, password: str, role: str) -> Dict:
        """Authenticate system user"""
        if not username or not password:
            return None
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM system_users 
            WHERE username = ? AND role = ? AND is_active = 1
        ''', (username, role))
    
        user = cursor.fetchone()
        if user:
            columns = [description[0] for description in cursor.description]
            user_dict = dict(zip(columns, user))
        
            password_hash = user_dict.get('password_hash')
            if password_hash and password_hash == self.hash_password(password):
                return {
                    'username': user_dict.get('username'),
                    'full_name': user_dict.get('full_name'),
                    'role': user_dict.get('role'),
                    'department': user_dict.get('department'),
                    'email': user_dict.get('email')
                }
        return None

    def load_or_generate_data(self):
        """Load data from database or generate sample data"""
        try:
            df = pd.read_sql('SELECT * FROM institutions', self.conn)
            if len(df) > 0:
                return df
        except:
            pass
        
        sample_data = self.generate_comprehensive_historical_data()
        sample_data.to_sql('institutions', self.conn, if_exists='replace', index=False)
        return sample_data

    def generate_comprehensive_historical_data(self) -> pd.DataFrame:
        """Generate comprehensive historical data for institutions"""
        np.random.seed(42)
        n_institutions = 200
        years_of_data = 5

        institutions_data = []

        for inst_id in range(1, n_institutions + 1):
            base_quality = np.random.uniform(0.3, 0.9)
        
            institution_type = np.random.choice(
                ['State University', 'Deemed University', 'Private University', 'Autonomous College'], 
                p=[0.3, 0.2, 0.3, 0.2]
            )
            state = np.random.choice(
                ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Uttar Pradesh', 'Kerala', 'Gujarat'], 
                p=[0.2, 0.15, 0.15, 0.1, 0.2, 0.1, 0.1]
            )
            established_year = np.random.randint(1950, 2015)
        
            for year_offset in range(years_of_data):
                year = 2023 - year_offset
                inst_trend = base_quality + (year_offset * 0.02)

                naac_grades = ['A++', 'A+', 'A', 'B++', 'B+', 'B', 'C']
                naac_probs = [0.05, 0.10, 0.15, 0.25, 0.25, 0.15, 0.05]
                naac_grade = np.random.choice(naac_grades, p=naac_probs)

                nirf_choices = list(range(1, 201)) + [None] * 50
                nirf_probs = [0.005] * 200 + [0.01] * 50
                nirf_probs = [p / sum(nirf_probs) for p in nirf_probs]
                nirf_rank = np.random.choice(nirf_choices, p=nirf_probs)

                student_faculty_ratio = max(10, np.random.normal(20, 5))
                phd_faculty_ratio = np.random.beta(2, 2) * 0.6 + 0.3

                publications = max(5, int(np.random.poisson(inst_trend * 30)))
                research_grants = max(100000, int(np.random.exponential(inst_trend * 500000)))
                patents = max(0, int(np.random.poisson(inst_trend * 3)))

                digital_infrastructure_score = max(3, min(10, np.random.normal(7, 1.5)))
                library_volumes = max(5000, int(np.random.normal(20000, 10000)))

                financial_stability = max(4, min(10, np.random.normal(7.5, 1.2)))
                compliance_score = max(5, min(10, np.random.normal(8, 1)))

                placement_rate = max(40, min(98, np.random.normal(75, 10)))
                higher_education_rate = max(5, min(50, np.random.normal(20, 8)))

                community_projects = max(1, int(np.random.poisson(inst_trend * 8)))

                faculty_count = max(30, np.random.randint(30, 150))
                performance_score = self.calculate_performance_score({
                    'naac_grade': naac_grade,
                    'nirf_ranking': nirf_rank,
                    'student_faculty_ratio': student_faculty_ratio,
                    'phd_faculty_ratio': phd_faculty_ratio,
                    'publications_per_faculty': publications / faculty_count,
                    'research_grants': research_grants,
                    'digital_infrastructure': digital_infrastructure_score,
                    'financial_stability': financial_stability,
                    'placement_rate': placement_rate,
                    'community_engagement': community_projects
                })

                if performance_score >= 8.0:
                    risk_level = "Low Risk"
                elif performance_score >= 6.5:
                    risk_level = "Medium Risk"
                elif performance_score >= 5.0:
                    risk_level = "High Risk"
                else:
                    risk_level = "Critical Risk"

                institution_data = {
                    'institution_id': f'INST_{inst_id:04d}',
                    'institution_name': f'University/College {inst_id:03d}',
                    'year': year,
                    'institution_type': institution_type,
                    'state': state,
                    'established_year': established_year,

                    'naac_grade': naac_grade,
                    'nirf_ranking': nirf_rank,
                    'student_faculty_ratio': round(student_faculty_ratio, 1),
                    'phd_faculty_ratio': round(phd_faculty_ratio, 3),

                    'research_publications': publications,
                    'research_grants_amount': research_grants,
                    'patents_filed': patents,
                    'industry_collaborations': max(1, int(np.random.poisson(inst_trend * 6))),

                    'digital_infrastructure_score': round(digital_infrastructure_score, 1),
                    'library_volumes': library_volumes,
                    'laboratory_equipment_score': round(max(3, min(10, np.random.normal(7, 1.3))), 1),

                    'financial_stability_score': round(financial_stability, 1),
                    'compliance_score': round(compliance_score, 1),
                    'administrative_efficiency': round(max(4, min(10, np.random.normal(7.2, 1.1))), 1),

                    'placement_rate': round(placement_rate, 1),
                    'higher_education_rate': round(higher_education_rate, 1),
                    'entrepreneurship_cell_score': round(max(3, min(10, np.random.normal(6.5, 1.5))), 1),

                    'community_projects': community_projects,
                    'rural_outreach_score': round(max(3, min(10, np.random.normal(6.8, 1.4))), 1),
                    'inclusive_education_index': round(max(4, min(10, np.random.normal(7.5, 1.2))), 1),

                    'rusa_participation': np.random.choice([0, 1], p=[0.4, 0.6]),
                    'nmeict_participation': np.random.choice([0, 1], p=[0.5, 0.5]),
                    'fist_participation': np.random.choice([0, 1], p=[0.6, 0.4]),
                    'dst_participation': np.random.choice([0, 1], p=[0.7, 0.3]),

                    'performance_score': round(performance_score, 2),
                    'approval_recommendation': self.generate_approval_recommendation(performance_score),
                    'risk_level': risk_level
                }

                institutions_data.append(institution_data)
        return pd.DataFrame(institutions_data)
        
    def calculate_performance_score(self, metrics: Dict) -> float:
        """Calculate overall performance score based on weighted metrics"""
        score = 0
        
        naac_scores = {'A++': 10, 'A+': 9, 'A': 8, 'B++': 7, 'B+': 6, 'B': 5, 'C': 4}
        naac_score = naac_scores.get(metrics['naac_grade'], 5)
        score += naac_score * 0.15
        
        nirf_score = 0
        if metrics['nirf_ranking'] and metrics['nirf_ranking'] <= 200:
            nirf_score = (201 - metrics['nirf_ranking']) / 200 * 10
        score += nirf_score * 0.10
        
        sf_ratio_score = max(0, 10 - max(0, metrics['student_faculty_ratio'] - 15) / 3)
        score += sf_ratio_score * 0.10
        
        phd_score = metrics['phd_faculty_ratio'] * 10
        score += phd_score * 0.10
        
        pub_score = min(10, metrics['publications_per_faculty'] * 3)
        score += pub_score * 0.10
        
        grant_score = min(10, np.log1p(metrics['research_grants'] / 100000) * 2.5)
        score += grant_score * 0.10
        
        infra_score = metrics['digital_infrastructure']
        score += infra_score * 0.10
        
        financial_score = metrics['financial_stability']
        score += financial_score * 0.10
        
        placement_score = metrics['placement_rate'] / 10
        score += placement_score * 0.10
        
        community_score = min(10, metrics['community_engagement'] / 1.5)
        score += community_score * 0.05
        
        return min(10, score)
    
    def generate_approval_recommendation(self, performance_score: float) -> str:
        """Generate approval recommendation based on performance score"""
        if performance_score >= 8.0:
            return "Full Approval - 5 Years"
        elif performance_score >= 7.0:
            return "Provisional Approval - 3 Years"
        elif performance_score >= 6.0:
            return "Conditional Approval - 1 Year"
        elif performance_score >= 5.0:
            return "Approval with Strict Monitoring - 1 Year"
        else:
            return "Rejection - Significant Improvements Required"

    def define_performance_metrics(self) -> Dict[str, Dict]:
        """Define key performance indicators for institutional evaluation"""
        return {
            "academic_excellence": {
                "weight": 0.25,
                "sub_metrics": {
                    "naac_grade": 0.30,
                    "nirf_ranking": 0.25,
                    "student_faculty_ratio": 0.20,
                    "phd_faculty_ratio": 0.15,
                    "curriculum_innovation": 0.10
                }
            },
            "research_innovation": {
                "weight": 0.20,
                "sub_metrics": {
                    "publications_per_faculty": 0.30,
                    "research_grants": 0.25,
                    "patents_filed": 0.20,
                    "conferences_organized": 0.15,
                    "industry_collaborations": 0.10
                }
            },
            "infrastructure_facilities": {
                "weight": 0.15,
                "sub_metrics": {
                    "campus_area": 0.25,
                    "digital_infrastructure": 0.25,
                    "library_resources": 0.20,
                    "laboratory_equipment": 0.20,
                    "hostel_facilities": 0.10
                }
            }
        }

    def define_document_requirements(self) -> Dict[str, Dict]:
        """Define document requirements for different approval types"""
        return {
            "new_approval": {
                "mandatory": [
                    "affidavit_legal_status", "land_documents", "building_plan_approval",
                    "infrastructure_details", "financial_solvency_certificate",
                    "faculty_recruitment_plan", "academic_curriculum", "governance_structure"
                ],
                "supporting": [
                    "feasibility_report", "market_demand_analysis", "five_year_development_plan",
                    "industry_partnerships", "research_facilities_plan"
                ]
            }
        }

    def analyze_documents_with_rag(self, institution_id: str, uploaded_files: List) -> Dict[str, Any]:
        """Analyze uploaded documents using RAG and extract structured data"""
        try:
            # Extract data using RAG
            extracted_data = self.rag_extractor.extract_comprehensive_data(uploaded_files)
        
            # Ensure extracted_data has all required keys
            if not extracted_data:
                extracted_data = {
                    'academic_metrics': {},
                    'research_metrics': {},
                    'infrastructure_metrics': {},
                    'governance_metrics': {},
                    'student_metrics': {},
                    'financial_metrics': {},
                    'raw_text': "",
                    'file_names': [f.name for f in uploaded_files]
                }
        
            # Save extracted data to database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO rag_analysis 
                (institution_id, analysis_type, extracted_data, confidence_score)
                VALUES (?, ?, ?, ?)
            ''', (institution_id, 'document_analysis', json.dumps(extracted_data), 0.85))
        
            self.conn.commit()
        
            # Generate AI insights
            ai_insights = self.generate_ai_insights(extracted_data)
        
            return {
                'extracted_data': extracted_data,
                'ai_insights': ai_insights,
                'confidence_score': 0.85,
                'status': 'Analysis Complete'
            }
        
        except Exception as e:
            st.error(f"Error in RAG analysis: {str(e)}")
            # Return a safe default structure even on error
            return {
                'extracted_data': {
                    'academic_metrics': {},
                    'research_metrics': {},
                    'infrastructure_metrics': {},
                    'governance_metrics': {},
                    'student_metrics': {},
                    'financial_metrics': {},
                    'raw_text': "",
                    'file_names': [f.name for f in uploaded_files] if uploaded_files else []
                },
                'ai_insights': {
                    'strengths': [],
                    'weaknesses': [],
                    'recommendations': [],
                    'risk_assessment': {'score': 5.0, 'level': 'Medium', 'factors': []},
                    'compliance_status': {}
                },
                'confidence_score': 0.0,
                'status': 'Analysis Failed'
            }

    def generate_ai_insights(self, extracted_data: Dict) -> Dict[str, Any]:
        """Generate AI insights from extracted data"""
        # Ensure extracted_data is not None
        if not extracted_data:
            extracted_data = {}
    
        insights = {
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'risk_assessment': {},
            'compliance_status': {}
        }
    
        # Safe access to nested dictionaries
        academic_data = extracted_data.get('academic_metrics', {})
        research_data = extracted_data.get('research_metrics', {})
        financial_data = extracted_data.get('financial_metrics', {})
        
        # Analyze academic metrics
        academic_data = extracted_data.get('academic_metrics', {})
        research_data = extracted_data.get('research_metrics', {})
        
        # Strength analysis
        if academic_data.get('naac_grade') in ['A++', 'A+', 'A']:
            insights['strengths'].append(f"Strong NAAC accreditation: {academic_data['naac_grade']}")
        
        if research_data.get('research_publications', 0) > 50:
            insights['strengths'].append("Robust research publication output")
        
        # Weakness analysis
        if academic_data.get('student_faculty_ratio', 0) > 25:
            insights['weaknesses'].append("High student-faculty ratio needs improvement")
        
        if research_data.get('patents_filed', 0) < 5:
            insights['weaknesses'].append("Limited patent filings - need to strengthen IPR culture")
        
        # Recommendations
        if not academic_data.get('nirf_ranking'):
            insights['recommendations'].append("Consider participating in NIRF ranking for better visibility")
        
        if research_data.get('industry_collaborations', 0) < 3:
            insights['recommendations'].append("Increase industry collaborations for practical exposure")
        
        # Risk assessment
        risk_score = self.calculate_risk_score(extracted_data)
        insights['risk_assessment'] = {
            'score': risk_score,
            'level': 'Low' if risk_score < 4 else 'Medium' if risk_score < 7 else 'High',
            'factors': self.identify_risk_factors(extracted_data)
        }
        
        return insights
    
    def calculate_risk_score(self, extracted_data: Dict) -> float:
        """Calculate risk score based on extracted data"""
        score = 5.0  # Default medium risk
        
        academic_data = extracted_data.get('academic_metrics', {})
        research_data = extracted_data.get('research_metrics', {})
        financial_data = extracted_data.get('financial_metrics', {})
        
        # Positive factors (reduce risk)
        if academic_data.get('naac_grade') in ['A++', 'A+', 'A']:
            score -= 1.5
        if research_data.get('research_publications', 0) > 50:
            score -= 1.0
        if financial_data.get('financial_stability_score', 0) > 7:
            score -= 1.0
        
        # Negative factors (increase risk)
        if academic_data.get('student_faculty_ratio', 0) > 25:
            score += 1.5
        if research_data.get('patents_filed', 0) < 2:
            score += 1.0
        if not academic_data.get('nirf_ranking'):
            score += 0.5
        
        return max(1.0, min(10.0, score))
    
    def identify_risk_factors(self, extracted_data: Dict) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []
        academic_data = extracted_data.get('academic_metrics', {})
        research_data = extracted_data.get('research_metrics', {})
        
        if academic_data.get('student_faculty_ratio', 0) > 25:
            risk_factors.append("High student-faculty ratio affecting quality")
        if research_data.get('industry_collaborations', 0) < 2:
            risk_factors.append("Limited industry exposure")
        if not academic_data.get('naac_grade'):
            risk_factors.append("No NAAC accreditation")
        
        return risk_factors

    def save_institution_submission(self, institution_id: str, submission_type: str, 
                                  submission_data: Dict):
        """Save institution submission data"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO institution_submissions 
            (institution_id, submission_type, submission_data)
            VALUES (?, ?, ?)
        ''', (institution_id, submission_type, json.dumps(submission_data)))
        self.conn.commit()

    def get_institution_submissions(self, institution_id: str) -> pd.DataFrame:
        """Get submissions for a specific institution"""
        return pd.read_sql('''
            SELECT * FROM institution_submissions 
            WHERE institution_id = ? 
            ORDER BY submitted_date DESC
        ''', self.conn, params=(institution_id,))

    def save_uploaded_documents(self, institution_id: str, uploaded_files: List, document_types: List[str]):
        """Save uploaded documents to database"""
        cursor = self.conn.cursor()
        for i, uploaded_file in enumerate(uploaded_files):
            cursor.execute('''
                INSERT INTO institution_documents (institution_id, document_name, document_type, status)
                VALUES (?, ?, ?, ?)
            ''', (institution_id, uploaded_file.name, document_types[i], 'Uploaded'))
        self.conn.commit()
    
    def get_institution_documents(self, institution_id: str) -> pd.DataFrame:
        """Get documents for a specific institution"""
        return pd.read_sql('''
            SELECT * FROM institution_documents 
            WHERE institution_id = ? 
            ORDER BY upload_date DESC
        ''', self.conn, params=(institution_id,))
    
    def analyze_document_sufficiency(self, uploaded_docs: List[str], approval_type: str) -> Dict:
        """Analyze document sufficiency percentage"""
        requirements = self.document_requirements[approval_type]
        
        # Count present documents
        mandatory_present = 0
        for doc in requirements['mandatory']:
            for uploaded_doc in uploaded_docs:
                if doc.lower() in uploaded_doc.lower():
                    mandatory_present += 1
                    break
        
        supporting_present = 0
        for doc in requirements['supporting']:
            for uploaded_doc in uploaded_docs:
                if doc.lower() in uploaded_doc.lower():
                    supporting_present += 1
                    break
        
        total_mandatory = len(requirements['mandatory'])
        total_supporting = len(requirements['supporting'])
        
        mandatory_sufficiency = (mandatory_present / total_mandatory) * 100 if total_mandatory > 0 else 0
        overall_sufficiency = ((mandatory_present + supporting_present) / 
                             (total_mandatory + total_supporting)) * 100 if (total_mandatory + total_supporting) > 0 else 0
        
        return {
            'mandatory_sufficiency': mandatory_sufficiency,
            'overall_sufficiency': overall_sufficiency,
            'missing_mandatory': [doc for doc in requirements['mandatory'] 
                                if not any(doc.lower() in uploaded_doc.lower() for uploaded_doc in uploaded_docs)],
            'missing_supporting': [doc for doc in requirements['supporting'] 
                                 if not any(doc.lower() in uploaded_doc.lower() for uploaded_doc in uploaded_docs)],
            'recommendations': self.generate_document_recommendations(mandatory_sufficiency)
        }
    
    def generate_document_recommendations(self, mandatory_sufficiency: float) -> List[str]:
        """Generate recommendations based on document sufficiency"""
        recommendations = []
        
        if mandatory_sufficiency < 100:
            recommendations.append("Upload all mandatory documents to proceed with approval process")
        
        if mandatory_sufficiency < 80:
            recommendations.append("Critical documents missing - application cannot be processed")
        
        if mandatory_sufficiency >= 100:
            recommendations.append("All mandatory documents present - ready for comprehensive evaluation")
        
        return recommendations

# Enhanced UI Functions with Beautiful Styling
def create_institution_login(analyzer):
    st.markdown('<div class="main-header">🏛️ Institution Portal Login</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Existing Institution Users</div>', unsafe_allow_html=True)
        username = st.text_input("Username", key="inst_login_username")
        password = st.text_input("Password", type="password", key="inst_login_password")
        
        if st.button("Login", key="inst_login_button", use_container_width=True):
            user = analyzer.authenticate_institution_user(username, password)
            if user:
                st.session_state.institution_user = user
                st.session_state.user_role = "Institution"
                st.success(f"Welcome, {user['contact_person']} from {user['institution_name']}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with col2:
        st.markdown('<div class="section-header">New Institution Registration</div>', unsafe_allow_html=True)
        
        available_institutions = analyzer.historical_data[
            analyzer.historical_data['year'] == 2023
        ][['institution_id', 'institution_name']].drop_duplicates()
        
        selected_institution = st.selectbox(
            "Select Your Institution",
            available_institutions['institution_id'].tolist(),
            format_func=lambda x: available_institutions[
                available_institutions['institution_id'] == x
            ]['institution_name'].iloc[0],
            key="inst_reg_institution"
        )
        
        new_username = st.text_input("Choose Username", key="inst_reg_username")
        new_password = st.text_input("Choose Password", type="password", key="inst_reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="inst_reg_confirm")
        contact_person = st.text_input("Contact Person Name", key="inst_reg_contact")
        email = st.text_input("Email Address", key="inst_reg_email")
        phone = st.text_input("Phone Number", key="inst_reg_phone")
        
        if st.button("Register Institution Account", key="inst_reg_button", use_container_width=True):
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            elif not all([new_username, new_password, contact_person, email]):
                st.error("Please fill all required fields!")
            else:
                success = analyzer.create_institution_user(
                    selected_institution, new_username, new_password,
                    contact_person, email, phone
                )
                if success:
                    st.success("Institution account created successfully! You can now login.")
                else:
                    st.error("Username already exists. Please choose a different username.")

def create_system_login(analyzer):
    st.markdown('<div class="main-header">🔐 System Login</div>', unsafe_allow_html=True)
    
    role = st.selectbox(
        "Select Your Role",
        ["UGC Officer", "AICTE Officer", "System Admin", "Review Committee"],
        key="system_login_role"
    )

    username = st.text_input("Username", key="system_login_username")
    password = st.text_input("Password", type="password", key="system_login_password")
    
    if st.button("Login", key="system_login_button", use_container_width=True):
        user = analyzer.authenticate_system_user(username, password, role)
        if user:
            st.session_state.system_user = user
            st.session_state.user_role = role
            st.success(f"Welcome, {user['full_name']} ({role})!")
            st.rerun()
        else:
            st.error("Invalid credentials for selected role!")

def create_institution_dashboard(analyzer, user):
    if not user:
        st.error("No user data available")
        return
        
    st.markdown(f'<div class="main-header">🏛️ {user.get("institution_name", "Unknown")} Dashboard</div>', unsafe_allow_html=True)
    
    # Get institution data
    institution_data = analyzer.historical_data[
        analyzer.historical_data['institution_id'] == user['institution_id']
    ].iloc[0] if not analyzer.historical_data[
        analyzer.historical_data['institution_id'] == user['institution_id']
    ].empty else None
    
    if institution_data is not None:
        # Create beautiful accreditation dashboard
        scores, overall_score, status = analyzer.accreditation_analyzer.create_accreditation_dashboard(
            institution_data, user['institution_name']
        )
    
    # Navigation for institution users with enhanced UI
    institution_tabs = st.tabs([
        "📤 Document Upload", 
        "📝 Data Submission", 
        "📊 My Submissions",
        "📋 Requirements Guide"
    ])
    
    with institution_tabs[0]:
        create_institution_document_upload(analyzer, user)
    
    with institution_tabs[1]:
        create_institution_data_submission(analyzer, user)
    
    with institution_tabs[2]:
        create_institution_submissions_view(analyzer, user)
    
    with institution_tabs[3]:
        create_institution_requirements_guide(analyzer)

def create_institution_document_upload(analyzer, user):
    st.subheader("📤 Document Upload Portal")
    
    st.info("Upload required documents for approval processes")
    
    approval_type = st.selectbox(
        "Select Approval Type",
        ["new_approval", "renewal_approval", "expansion_approval"],
        format_func=lambda x: x.replace('_', ' ').title(),
        key="inst_approval_type"
    )
    
    uploaded_files = st.file_uploader(
        "Upload Institutional Documents",
        type=['pdf', 'doc', 'docx', 'xlsx', 'jpg', 'png'],
        accept_multiple_files=True,
        help="Upload all required documents for your application"
    )
    
    if uploaded_files:
        # Document type mapping
        st.subheader("📝 Document Type Assignment")
        document_types = []
        for i, file in enumerate(uploaded_files):
            doc_type = st.selectbox(
                f"Document type for: {file.name}",
                ["affidavit_legal_status", "land_documents", "building_plan_approval", 
                 "financial_solvency_certificate", "faculty_recruitment_plan", 
                 "academic_curriculum", "annual_reports", "research_publications",
                 "placement_records", "other"],
                key=f"inst_doc_type_{i}"
            )
            document_types.append(doc_type)
        
        if st.button("💾 Upload Documents"):
            # Save documents
            analyzer.save_uploaded_documents(user['institution_id'], uploaded_files, document_types)
            st.success("✅ Documents uploaded successfully!")
            
            # Analyze document sufficiency
            file_names = [file.name for file in uploaded_files]
            analysis_result = analyzer.analyze_document_sufficiency(file_names, approval_type)
            
            # Display results
            st.subheader("📊 Upload Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Mandatory Documents", 
                    f"{analysis_result['mandatory_sufficiency']:.1f}%",
                    delta=f"{analysis_result['mandatory_sufficiency'] - 100:.1f}%" if analysis_result['mandatory_sufficiency'] < 100 else None,
                    delta_color="inverse"
                )
            
            with col2:
                st.metric(
                    "Overall Sufficiency", 
                    f"{analysis_result['overall_sufficiency']:.1f}%"
                )
            
            # Show missing documents
            if analysis_result['missing_mandatory']:
                st.error("**❌ Missing Mandatory Documents:**")
                for doc in analysis_result['missing_mandatory']:
                    st.write(f"• {doc.replace('_', ' ').title()}")
            
            # Recommendations
            st.info("**💡 Next Steps:**")
            for recommendation in analysis_result['recommendations']:
                st.write(f"• {recommendation}")

def create_institution_data_submission(analyzer, user):
    st.subheader("📝 Data Submission Form")
    
    st.info("Submit institutional data and performance metrics through this form")
    
    with st.form("institution_data_submission"):
        st.write("### Academic Performance Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            naac_grade = st.selectbox(
                "NAAC Grade",
                ["A++", "A+", "A", "B++", "B+", "B", "C"]
            )
            student_faculty_ratio = st.number_input(
                "Student-Faculty Ratio",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=0.1
            )
            phd_faculty_ratio = st.number_input(
                "PhD Faculty Ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0
            ) / 100
        
        with col2:
            nirf_ranking = st.number_input(
                "NIRF Ranking (if applicable)",
                min_value=1,
                max_value=200,
                value=None,
                placeholder="Leave blank if not ranked"
            )
            placement_rate = st.number_input(
                "Placement Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )
        
        st.write("### Research & Infrastructure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            research_publications = st.number_input(
                "Research Publications (Last Year)",
                min_value=0,
                value=50
            )
            research_grants = st.number_input(
                "Research Grants Amount (₹)",
                min_value=0,
                value=1000000,
                step=100000
            )
        
        with col2:
            digital_infrastructure_score = st.slider(
                "Digital Infrastructure Score",
                min_value=1,
                max_value=10,
                value=7
            )
            library_volumes = st.number_input(
                "Library Volumes",
                min_value=0,
                value=20000
            )
        
        st.write("### Governance & Social Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            financial_stability_score = st.slider(
                "Financial Stability Score",
                min_value=1,
                max_value=10,
                value=8
            )
            community_projects = st.number_input(
                "Community Projects (Last Year)",
                min_value=0,
                value=10
            )
        
        with col2:
            compliance_score = st.slider(
                "Compliance Score",
                min_value=1,
                max_value=10,
                value=8
            )
            administrative_efficiency = st.slider(
                "Administrative Efficiency",
                min_value=1,
                max_value=10,
                value=7
            )
        
        submission_notes = st.text_area(
            "Additional Notes / Comments",
            placeholder="Add any additional information or context for your submission..."
        )
        
        submitted = st.form_submit_button("📤 Submit Data")
        
        if submitted:
            submission_data = {
                "academic_data": {
                    "naac_grade": naac_grade,
                    "nirf_ranking": nirf_ranking,
                    "student_faculty_ratio": student_faculty_ratio,
                    "phd_faculty_ratio": phd_faculty_ratio,
                    "placement_rate": placement_rate
                },
                "research_data": {
                    "research_publications": research_publications,
                    "research_grants": research_grants,
                    "digital_infrastructure_score": digital_infrastructure_score,
                    "library_volumes": library_volumes
                },
                "governance_data": {
                    "financial_stability_score": financial_stability_score,
                    "compliance_score": compliance_score,
                    "administrative_efficiency": administrative_efficiency,
                    "community_projects": community_projects
                },
                "submission_notes": submission_notes,
                "submission_date": datetime.now().isoformat()
            }
            
            analyzer.save_institution_submission(
                user['institution_id'],
                "annual_performance_data",
                submission_data
            )
            
            st.success("✅ Data submitted successfully! Your submission is under review.")
            st.balloons()

def create_institution_submissions_view(analyzer, user):
    st.subheader("📊 My Submissions & Status")
    
    submissions = analyzer.get_institution_submissions(user['institution_id'])
    
    if len(submissions) > 0:
        for _, submission in submissions.iterrows():
            with st.expander(f"{submission['submission_type']} - {submission['submitted_date']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Status:** {submission['status']}")
                with col2:
                    st.write(f"**Submitted:** {submission['submitted_date']}")
                with col3:
                    if submission['reviewed_by']:
                        st.write(f"**Reviewed by:** {submission['reviewed_by']}")
                
                if submission['review_comments']:
                    st.info(f"**Review Comments:** {submission['review_comments']}")
                
                # Display submission data
                try:
                    submission_data = json.loads(submission['submission_data'])
                    st.json(submission_data)
                except:
                    st.write("Submission data format not available")
    else:
        st.info("No submissions found. Use the Data Submission tab to submit your institutional data.")

def create_institution_requirements_guide(analyzer):
    st.subheader("📋 Approval Requirements Guide")
    
    requirements = analyzer.document_requirements
    
    for approval_type, docs in requirements.items():
        with st.expander(f"{approval_type.replace('_', ' ').title()} Requirements"):
            st.write("**Mandatory Documents:**")
            for doc in docs['mandatory']:
                st.write(f"• {doc.replace('_', ' ').title()}")
            
            st.write("**Supporting Documents:**")
            for doc in docs['supporting']:
                st.write(f"• {doc.replace('_', ' ').title()}")

def create_accreditation_analytics_dashboard(analyzer):
    """Enhanced accreditation analytics dashboard for multiple institutions"""
    st.markdown('<div class="main-header">🎓 Multi-Institution Accreditation Analytics</div>', unsafe_allow_html=True)
    
    # Institution selection
    current_institutions = analyzer.historical_data[
        analyzer.historical_data['year'] == 2023
    ]['institution_id'].unique()
    
    selected_institution = st.selectbox(
        "Select Institution for Detailed Analysis",
        current_institutions,
        key="accreditation_institution"
    )
    
    if selected_institution:
        institution_data = analyzer.historical_data[
            (analyzer.historical_data['institution_id'] == selected_institution) & 
            (analyzer.historical_data['year'] == 2023)
        ].iloc[0]
        
        institution_name = institution_data['institution_name']
        
        # Create beautiful accreditation dashboard for selected institution
        scores, overall_score, status = analyzer.accreditation_analyzer.create_accreditation_dashboard(
            institution_data, institution_name
        )
        
        # Additional analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">📈 Performance Trends</div>', unsafe_allow_html=True)
            
            # Historical performance
            historical_data = analyzer.historical_data[
                analyzer.historical_data['institution_id'] == selected_institution
            ]
            
            if len(historical_data) > 1:
                fig = px.line(
                    historical_data, 
                    x='year', 
                    y='performance_score',
                    title=f'Performance Score Trend for {institution_name}',
                    markers=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header">🎯 AI Recommendations</div>', unsafe_allow_html=True)
            
            # Strengths and weaknesses analysis
            strengths = []
            weaknesses = []
            
            for param, score in scores.items():
                if score >= 80:
                    strengths.append(f"{param}: {score:.1f}/100")
                elif score <= 60:
                    weaknesses.append(f"{param}: {score:.1f}/100")
            
            if strengths:
                st.success("**✅ Key Strengths**")
                for strength in strengths[:3]:
                    st.write(f"• {strength}")
            
            if weaknesses:
                st.error("**⚠️ Areas for Improvement**")
                for weakness in weaknesses[:3]:
                    st.write(f"• {weakness}")
            
            # Risk assessment
            st.markdown("**🔍 Risk Assessment**")
            risks = [
                ("Curriculum Relevance", "Medium", "Regular industry interaction needed"),
                ("Faculty Retention", "Low", "Good retention rates observed"),
                ("Research Funding", "High", "Need to diversify funding sources"),
            ]
            
            for risk, level, description in risks:
                risk_color = {"High": "red", "Medium": "orange", "Low": "green"}[level]
                st.markdown(f"""
                <div style="border-left: 4px solid {risk_color}; padding: 10px; margin: 5px 0;">
                    <strong>{risk}</strong> | <span style="color: {risk_color}">{level} Risk</span><br>
                    {description}
                </div>
                """, unsafe_allow_html=True)

def create_performance_dashboard(analyzer):
    """Enhanced performance dashboard with beautiful UI"""
    st.markdown('<div class="main-header">📊 Institutional Performance Analytics Dashboard</div>', unsafe_allow_html=True)
    
    df = analyzer.historical_data
    current_year_data = df[df['year'] == 2023]
    
    if len(current_year_data) == 0:
        st.warning("No data available for the current year.")
        return
    
    # Key Performance Indicators with enhanced styling
    st.markdown('<div class="section-header">🏆 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_performance = current_year_data['performance_score'].mean()
        st.metric("Average Performance Score", f"{avg_performance:.2f}/10")
    
    with col2:
        approval_rate = (current_year_data['performance_score'] >= 6.0).mean()
        st.metric("Approval Eligibility Rate", f"{approval_rate:.1f}%")
    
    with col3:
        high_risk_count = (current_year_data['risk_level'] == 'High Risk').sum() + (
            current_year_data['risk_level'] == 'Critical Risk').sum()
        st.metric("High/Critical Risk Institutions", high_risk_count)
    
    with col4:
        avg_placement = current_year_data['placement_rate'].mean()
        st.metric("Average Placement Rate", f"{avg_placement:.1f}%")
    
    with col5:
        research_intensity = current_year_data['research_publications'].sum() / len(current_year_data)
        st.metric("Avg Research Publications", f"{research_intensity:.1f}")
    
    # Performance Analysis with enhanced charts
    st.markdown('<div class="section-header">📈 Performance Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not current_year_data['performance_score'].empty:
            fig1 = px.histogram(
                current_year_data, 
                x='performance_score',
                title="Distribution of Institutional Performance Scores",
                nbins=20,
                color_discrete_sequence=['#1f77b4'],
                opacity=0.8
            )
            fig1.update_layout(
                xaxis_title="Performance Score", 
                yaxis_title="Number of Institutions",
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if not current_year_data.empty and 'institution_type' in current_year_data.columns:
            filtered_data = current_year_data.dropna(subset=['institution_type', 'performance_score'])
            if not filtered_data.empty:
                fig2 = px.box(
                    filtered_data,
                    x='institution_type',
                    y='performance_score',
                    title="Performance Score by Institution Type",
                    color='institution_type'
                )
                fig2.update_layout(
                    xaxis_title="Institution Type",
                    yaxis_title="Performance Score",
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)

def create_document_analysis_module(analyzer):
    st.markdown('<div class="main-header">📋 AI-Powered Document Sufficiency Analysis</div>', unsafe_allow_html=True)
    
    st.info("Analyze document completeness and generate sufficiency reports for approval processes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Document Upload & Analysis</div>', unsafe_allow_html=True)
        
        current_institutions = analyzer.historical_data[analyzer.historical_data['year'] == 2023]['institution_id'].unique()
        selected_institution = st.selectbox(
            "Select Institution",
            current_institutions
        )
        
        approval_type = st.selectbox(
            "Select Approval Type",
            ["new_approval", "renewal_approval", "expansion_approval"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        uploaded_files = st.file_uploader(
            "Upload Institutional Documents",
            type=['pdf', 'doc', 'docx', 'xlsx', 'jpg', 'png'],
            accept_multiple_files=True,
            help="Upload all required documents for AI analysis"
        )
        
        if uploaded_files:
            st.markdown('<div class="section-header">📝 Document Type Assignment</div>', unsafe_allow_html=True)
            document_types = []
            for i, file in enumerate(uploaded_files):
                doc_type = st.selectbox(
                    f"Document type for: {file.name}",
                    ["affidavit_legal_status", "land_documents", "building_plan_approval", 
                     "financial_solvency_certificate", "faculty_recruitment_plan", 
                     "academic_curriculum", "annual_reports", "research_publications",
                     "placement_records", "other"],
                    key=f"doc_type_{i}"
                )
                document_types.append(doc_type)
            
            if st.button("💾 Save Documents & Analyze", use_container_width=True):
                # Save documents
                analyzer.save_uploaded_documents(selected_institution, uploaded_files, document_types)
                st.success("✅ Documents saved successfully!")
                
                # Analyze document sufficiency
                file_names = [file.name for file in uploaded_files]
                analysis_result = analyzer.analyze_document_sufficiency(file_names, approval_type)
                
                # Display results
                st.subheader("📊 Document Sufficiency Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Mandatory Documents", 
                        f"{analysis_result['mandatory_sufficiency']:.1f}%",
                        delta=f"{analysis_result['mandatory_sufficiency'] - 100:.1f}%" if analysis_result['mandatory_sufficiency'] < 100 else None,
                        delta_color="inverse"
                    )
                
                with col2:
                    st.metric(
                        "Overall Sufficiency", 
                        f"{analysis_result['overall_sufficiency']:.1f}%"
                    )
                
                # Visual representation
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode = "gauge+number+delta",
                    value = analysis_result['mandatory_sufficiency'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Mandatory Documents Sufficiency"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Missing documents
                if analysis_result['missing_mandatory']:
                    st.error("**❌ Missing Mandatory Documents:**")
                    for doc in analysis_result['missing_mandatory']:
                        st.write(f"• {doc.replace('_', ' ').title()}")
                
                if analysis_result['missing_supporting']:
                    st.warning("**📝 Missing Supporting Documents:**")
                    for doc in analysis_result['missing_supporting']:
                        st.write(f"• {doc.replace('_', ' ').title()}")
                
                # Recommendations
                st.info("**💡 AI Recommendations:**")
                for recommendation in analysis_result['recommendations']:
                    st.write(f"• {recommendation}")
    
    with col2:
        st.markdown('<div class="section-header">Document Requirements Guide</div>', unsafe_allow_html=True)
        
        requirements = analyzer.document_requirements
        
        for approval_type, docs in requirements.items():
            with st.expander(f"{approval_type.replace('_', ' ').title()} Requirements"):
                st.write("**Mandatory Documents:**")
                for doc in docs['mandatory']:
                    st.write(f"• {doc.replace('_', ' ').title()}")
                
                st.write("**Supporting Documents:**")
                for doc in docs['supporting']:
                    st.write(f"• {doc.replace('_', ' ').title()}")
        
        # Show uploaded documents for selected institution
        if selected_institution:
            st.subheader("📁 Previously Uploaded Documents")
            existing_docs = analyzer.get_institution_documents(selected_institution)
            if len(existing_docs) > 0:
                st.dataframe(existing_docs[['document_name', 'document_type', 'upload_date', 'status']])
            else:
                st.info("No documents uploaded yet for this institution.")

def create_ai_analysis_reports(analyzer):
    st.header("🤖 Comprehensive AI Analysis Reports")
    
    df = analyzer.historical_data
    current_institutions = df[df['year'] == 2023]['institution_id'].unique()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_institution = st.selectbox(
            "Select Institution for Detailed Analysis",
            current_institutions
        )
        
        if selected_institution:
            # Generate comprehensive report using the existing method
            inst_data = analyzer.historical_data[
                analyzer.historical_data['institution_id'] == selected_institution
            ]
            
            if inst_data.empty:
                st.error("Institution not found")
                return
            
            latest_data = inst_data[inst_data['year'] == inst_data['year'].max()].iloc[0]
            historical_trend = inst_data.groupby('year')['performance_score'].mean()
            
            # Performance trends
            if len(historical_trend) > 1:
                if historical_trend.iloc[-1] > historical_trend.iloc[-2]:
                    trend_analysis = "Improving"
                elif historical_trend.iloc[-1] == historical_trend.iloc[-2]:
                    trend_analysis = "Stable"
                else:
                    trend_analysis = "Declining"
            else:
                trend_analysis = "Insufficient Data"
            
            # Create a simple report structure
            report = {
                "institution_info": {
                    "name": latest_data['institution_name'],
                    "type": latest_data['institution_type'],
                    "state": latest_data['state'],
                    "established": latest_data['established_year']
                },
                "performance_analysis": {
                    "current_score": latest_data['performance_score'],
                    "historical_trend": historical_trend.to_dict(),
                    "trend_analysis": trend_analysis,
                    "approval_recommendation": latest_data['approval_recommendation'],
                    "risk_level": latest_data['risk_level']
                }
            }
            
            if "error" not in report:
                st.subheader(f"🏛️ AI Analysis Report: {report['institution_info']['name']}")
                
                # Institution Overview
                st.info("**Institution Overview**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Type", report['institution_info']['type'])
                with col2:
                    st.metric("State", report['institution_info']['state'])
                with col3:
                    st.metric("Established", report['institution_info']['established'])
                with col4:
                    st.metric("Performance Score", f"{report['performance_analysis']['current_score']:.2f}/10")
                
                # Approval Recommendation with colored indicator
                recommendation = report['performance_analysis']['approval_recommendation']
                if "Full Approval" in recommendation:
                    st.success(f"**✅ {recommendation}**")
                elif "Provisional" in recommendation:
                    st.warning(f"**🟡 {recommendation}**")
                elif "Conditional" in recommendation or "Monitoring" in recommendation:
                    st.error(f"**🟠 {recommendation}**")
                else:
                    st.error(f"**🔴 {recommendation}**")
                
                # Risk Level
                risk_level = report['performance_analysis']['risk_level']
                if risk_level == "Low Risk":
                    st.success(f"**Risk Level: {risk_level}**")
                elif risk_level == "Medium Risk":
                    st.warning(f"**Risk Level: {risk_level}**")
                else:
                    st.error(f"**Risk Level: {risk_level}**")
                
                # Performance Trend
                st.metric(
                    "Performance Trend", 
                    report['performance_analysis']['trend_analysis'],
                    delta=report['performance_analysis']['trend_analysis'],
                    delta_color="normal" if report['performance_analysis']['trend_analysis'] == "Improving" else "off" if report['performance_analysis']['trend_analysis'] == "Stable" else "inverse"
                )
                
                # Historical Performance Chart
                if len(report['performance_analysis']['historical_trend']) > 1:
                    st.subheader("📈 Historical Performance Trend")
                    trend_df = pd.DataFrame(list(report['performance_analysis']['historical_trend'].items()), 
                                          columns=['Year', 'Performance Score'])
                    fig = px.line(trend_df, x='Year', y='Performance Score', 
                                title=f"Performance Trend for {report['institution_info']['name']}",
                                markers=True)
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Quick Institutional Insights")
        
        # Top performers
        top_performers = df[df['year'] == 2023].nlargest(5, 'performance_score')[
            ['institution_name', 'performance_score', 'approval_recommendation']
        ]
        
        st.write("**🏆 Top Performing Institutions**")
        for _, inst in top_performers.iterrows():
            st.write(f"• **{inst['institution_name']}** ({inst['performance_score']:.2f})")
            st.write(f"  _{inst['approval_recommendation']}_")
        
        st.markdown("---")
        
        # High risk institutions
        high_risk = df[
            (df['year'] == 2023) & 
            (df['risk_level'].isin(['High Risk', 'Critical Risk']))
        ].head(5)
        
        if not high_risk.empty:
            st.write("**🚨 High Risk Institutions**")
            for _, inst in high_risk.iterrows():
                st.write(f"• **{inst['institution_name']}** - {inst['risk_level']}")
        
        # Quick stats
        st.markdown("---")
        st.write("**📊 Quick Statistics**")
        total_inst = len(df[df['year'] == 2023])
        approved = len(df[(df['year'] == 2023) & (df['performance_score'] >= 7.0)])
        st.write(f"• Total Institutions: {total_inst}")
        st.write(f"• High Performing: {approved}")
        st.write(f"• Approval Rate: {(approved/total_inst*100):.1f}%")

def create_data_management_module(analyzer):
    st.header("💾 Data Management & Upload")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload New Data", "🔍 View Current Data", "⚙️ Database Management"])
    
    with tab1:
        st.subheader("Upload Institutional Data")
        
        uploaded_file = st.file_uploader("Upload CSV file with institutional data", type=['csv'])
        
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(new_data)} records")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(new_data.head())
                
                # Data validation
                required_columns = ['institution_id', 'institution_name', 'year', 'institution_type']
                missing_columns = [col for col in required_columns if col not in new_data.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {missing_columns}")
                else:
                    st.success("✅ All required columns present")
                    
                    if st.button("💾 Save to Database"):
                        try:
                            new_data.to_sql('institutions', analyzer.conn, if_exists='append', index=False)
                            st.success("✅ Data successfully saved to database!")
                            # Refresh the data
                            analyzer.historical_data = analyzer.load_or_generate_data()
                        except Exception as e:
                            st.error(f"❌ Error saving to database: {str(e)}")
                            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with tab2:
        st.subheader("Current Database Contents")
        
        # Show database statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_records = len(analyzer.historical_data)
            st.metric("Total Records", total_records)
        with col2:
            unique_institutions = analyzer.historical_data['institution_id'].nunique()
            st.metric("Unique Institutions", unique_institutions)
        with col3:
            years_covered = analyzer.historical_data['year'].nunique()
            st.metric("Years Covered", years_covered)
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(analyzer.historical_data.head(10))
        
        # Export data
        if st.button("📥 Export Current Data as CSV"):
            csv = analyzer.historical_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="institutional_data_export.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.subheader("Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Regenerate Sample Data", help="Replace current data with new sample data"):
                new_data = analyzer.generate_comprehensive_historical_data()
                new_data.to_sql('institutions', analyzer.conn, if_exists='replace', index=False)
                analyzer.historical_data = new_data
                st.success("✅ Sample data regenerated successfully!")
        
        with col2:
            if st.button("🗑️ Clear All Data", help="Remove all data from the database"):
                cursor = analyzer.conn.cursor()
                cursor.execute('DELETE FROM institutions')
                analyzer.conn.commit()
                analyzer.historical_data = pd.DataFrame()
                st.success("✅ All data cleared successfully!")

def create_approval_workflow(analyzer):
    st.header("🔄 AI-Enhanced Approval Workflow")
    
    st.info("Streamlined approval process with AI-powered decision support")
    
    # Workflow steps with AI integration
    workflow_steps = [
        {
            "step": 1,
            "title": "Document Submission & AI Verification",
            "description": "Institutions submit documents through portal, AI verifies completeness",
            "ai_features": ["Document classification", "Completeness check", "Sufficiency scoring"],
            "output": "Document Sufficiency Report"
        },
        {
            "step": 2,
            "title": "Historical Data Analysis",
            "description": "AI analyzes 5+ years of institutional performance data",
            "ai_features": ["Trend analysis", "Performance scoring", "Risk assessment"],
            "output": "Performance Analytics Report"
        },
        {
            "step": 3,
            "title": "Comparative Benchmarking",
            "description": "AI compares institution with similar peers and standards",
            "ai_features": ["Peer comparison", "Benchmark analysis", "Percentile ranking"],
            "output": "Comparative Analysis Report"
        },
        {
            "step": 4,
            "title": "AI Recommendation Generation",
            "description": "AI generates approval recommendations with justifications",
            "ai_features": ["Decision support", "Risk mitigation", "Improvement suggestions"],
            "output": "AI Recommendation Report"
        },
        {
            "step": 5,
            "title": "Expert Committee Review",
            "description": "UGC/AICTE committee reviews AI recommendations and makes final decision",
            "ai_features": ["Decision tracking", "Approval workflow", "Monitoring setup"],
            "output": "Final Approval Decision"
        }
    ]
    
    for step in workflow_steps:
        with st.expander(f"Step {step['step']}: {step['title']}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Description:** {step['description']}")
                st.write(f"**Output:** {step['output']}")
            with col2:
                st.write("**AI Features:**")
                for feature in step['ai_features']:
                    st.write(f"• {feature}")

def create_rag_data_management(analyzer):
    st.header("🤖 RAG-Powered Data Management & Analysis")
    
    st.info("""
    **Retrieval Augmented Generation (RAG) System**: 
    Upload institutional documents and let AI automatically extract, analyze, and structure data 
    for comprehensive institutional evaluation.
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Extract", 
        "🔍 View Extracted Data", 
        "📊 AI Analysis",
        "⚙️ RAG Settings"
    ])
    
    with tab1:
        st.subheader("Document Upload & Data Extraction")
        
        # Institution selection
        current_institutions = analyzer.historical_data[analyzer.historical_data['year'] == 2023]['institution_id'].unique()
        selected_institution = st.selectbox(
            "Select Institution",
            current_institutions,
            key="rag_institution"
        )
        
        uploaded_files = st.file_uploader(
            "Upload Institutional Documents for RAG Analysis",
            type=['pdf', 'doc', 'docx', 'txt', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload all relevant documents: Annual reports, NAAC reports, Research data, etc."
        )
        
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} documents ready for analysis")
            
            # Show document preview
            with st.expander("📋 Document Preview"):
                for i, file in enumerate(uploaded_files):
                    st.write(f"**{i+1}. {file.name}** ({file.size} bytes)")
            
            if st.button("🚀 Start RAG Analysis", type="primary"):
                with st.spinner("🤖 AI is analyzing documents and extracting data..."):
                    # Perform RAG analysis
                    analysis_result = analyzer.analyze_documents_with_rag(
                        selected_institution, 
                        uploaded_files
                    )
                    
                    # SAFE ACCESS: Check if analysis_result is valid
                    if analysis_result and analysis_result.get('status') == 'Analysis Complete':
                        st.success("✅ RAG Analysis Completed Successfully!")
                        
                        # Store results in session state for other tabs
                        st.session_state.rag_analysis = analysis_result
                        st.session_state.selected_institution = selected_institution
                        
                        # Show quick insights with safe access
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            confidence = analysis_result.get('confidence_score', 0.0)
                            st.metric("Confidence Score", f"{confidence:.2f}")
                        with col2:
                            extracted_data = analysis_result.get('extracted_data', {})
                            extracted_categories = len([k for k in extracted_data.keys() if k not in ['raw_text', 'file_names']])
                            st.metric("Data Categories", extracted_categories)
                        with col3:
                            ai_insights = analysis_result.get('ai_insights', {})
                            risk_assessment = ai_insights.get('risk_assessment', {})
                            risk_level = risk_assessment.get('level', 'Unknown')
                            st.metric("Risk Level", risk_level)
                    
                    else:
                        st.error("❌ RAG Analysis Failed. Please try again.")
                        # Still store the result even if failed for debugging
                        if analysis_result:
                            st.session_state.rag_analysis = analysis_result
    
    with tab2:
        st.subheader("Extracted Data View")
        
        # SAFE ACCESS: Check if rag_analysis exists and has extracted_data
        if 'rag_analysis' in st.session_state and st.session_state.rag_analysis:
            analysis_result = st.session_state.rag_analysis
            extracted_data = analysis_result.get('extracted_data', {})
            
            if not extracted_data:
                st.warning("No extracted data available. Please run RAG analysis first.")
                return
            
            # Show extracted data by category
            categories = [
                ('Academic Metrics', 'academic_metrics'),
                ('Research Metrics', 'research_metrics'), 
                ('Infrastructure Metrics', 'infrastructure_metrics'),
                ('Governance Metrics', 'governance_metrics'),
                ('Financial Metrics', 'financial_metrics')
            ]
            
            for category_name, category_key in categories:
                with st.expander(f"📈 {category_name}"):
                    category_data = extracted_data.get(category_key, {})
                    if category_data:
                        for key, value in category_data.items():
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.write(f"**{key.replace('_', ' ').title()}:**")
                            with col2:
                                st.write(value)
                    else:
                        st.info(f"No {category_name.lower()} extracted from documents")
            
            # Show raw text preview
            with st.expander("📝 Extracted Text Preview"):
                raw_text = extracted_data.get('raw_text', '')
                if raw_text:
                    preview_text = raw_text[:2000] + "..." if len(raw_text) > 2000 else raw_text
                    st.text_area("Extracted Text", preview_text, height=200, key="raw_text_preview")
                else:
                    st.info("No text extracted")
                    
        else:
            st.info("👆 Upload documents and run RAG analysis to view extracted data")
    
    with tab3:
        st.subheader("AI Insights & Analysis")
        
        # SAFE ACCESS: Check if rag_analysis exists and has ai_insights
        if 'rag_analysis' in st.session_state and st.session_state.rag_analysis:
            analysis_result = st.session_state.rag_analysis
            insights = analysis_result.get('ai_insights', {})
            
            if not insights:
                st.warning("No AI insights available. Please run RAG analysis first.")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Strengths")
                strengths = insights.get('strengths', [])
                if strengths:
                    for strength in strengths:
                        st.success(f"• {strength}")
                else:
                    st.info("No significant strengths identified")
                
                st.subheader("🎯 Recommendations")
                recommendations = insights.get('recommendations', [])
                if recommendations:
                    for rec in recommendations:
                        st.warning(f"• {rec}")
                else:
                    st.info("No specific recommendations")
            
            with col2:
                st.subheader("⚠️ Areas for Improvement")
                weaknesses = insights.get('weaknesses', [])
                if weaknesses:
                    for weakness in weaknesses:
                        st.error(f"• {weakness}")
                else:
                    st.info("No major weaknesses identified")
                
                st.subheader("📊 Risk Assessment")
                risk_assessment = insights.get('risk_assessment', {})
                risk_score = risk_assessment.get('score', 5.0)
                st.metric("Risk Score", f"{risk_score:.1f}/10")
                
                risk_level = risk_assessment.get('level', 'Unknown')
                st.write(f"**Risk Level:** {risk_level}")
                
                risk_factors = risk_assessment.get('factors', [])
                if risk_factors:
                    st.write("**Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
            
            # Generate approval recommendation
            st.subheader("🏛️ Approval Recommendation")
            risk_level = risk_assessment.get('level', 'Medium')
            if risk_level == 'Low':
                st.success("**✅ RECOMMENDED: Full Approval - 5 Years**")
                st.write("Institution demonstrates strong performance across all parameters with minimal risk factors.")
            elif risk_level == 'Medium':
                st.warning("**🟡 CONDITIONAL: Provisional Approval - 3 Years**")
                st.write("Institution shows promise but has some areas requiring improvement and monitoring.")
            else:
                st.error("**🔴 NOT RECOMMENDED: Requires Significant Improvements**")
                st.write("Critical risk factors identified. Institution needs substantial improvements before approval.")
                
        else:
            st.info("👆 Run RAG analysis to generate AI insights")
    
    with tab4:
        st.subheader("RAG System Settings")
        st.info("RAG system configuration and settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Embedding Model:** all-MiniLM-L6-v2")
            st.write("**Chunk Size:** 1000 characters")
            st.write("**Chunk Overlap:** 200 characters")
        
        with col2:
            st.write("**Vector Store:** Simple Cosine Similarity")
            st.write("**Similarity Threshold:** 0.5")
            st.write("**Max Results:** 5")

def create_system_settings(analyzer):
    st.header("⚙️ System Settings & Configuration")
    st.info("System administration and configuration panel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics Configuration")
        st.json(analyzer.performance_metrics)
    
    with col2:
        st.subheader("Document Requirements")
        st.json(analyzer.document_requirements)
    
    st.subheader("System Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Database Records", len(analyzer.historical_data))
    with col2:
        st.metric("Unique Institutions", analyzer.historical_data['institution_id'].nunique())
    with col3:
        st.metric("Data Years", f"{analyzer.historical_data['year'].min()}-{analyzer.historical_data['year'].max()}")

def get_available_modules(user_role):
    """Return available modules based on user role"""
    base_modules = []
    
    if user_role == "Institution":
        base_modules = ["🏛️ Institution Portal"]
    elif user_role == "System Admin":
        base_modules = ["📊 Performance Dashboard", "🎓 Accreditation Analytics", "⚙️ System Settings"]
    elif user_role in ["UGC Officer", "AICTE Officer"]:
        base_modules = ["🎓 Accreditation Analytics", "🔄 Approval Workflow", "💾 Data Management", "🔍 RAG Data Management", "📋 Document Analysis"]
    elif user_role == "Review Committee":
        base_modules = ["🎓 Accreditation Analytics", "🤖 AI Reports"]
    
    return base_modules

def main():
    # Safe session state initialization
    if 'institution_user' not in st.session_state:
        st.session_state.institution_user = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'system_user' not in st.session_state:
        st.session_state.system_user = None
    
    # Initialize analytics engine
    try:
        analyzer = InstitutionalAIAnalyzer()
    except Exception as e:
        st.error(f"❌ System initialization error: {str(e)}")
        st.stop()
    
    # Check if user is logged in
    if st.session_state.institution_user is not None:
        create_institution_dashboard(analyzer, st.session_state.institution_user)
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.institution_user = None
            st.session_state.user_role = None
            st.rerun()
        return
    
    if st.session_state.system_user is not None:
        user_role = st.session_state.user_role
        available_modules = get_available_modules(user_role)
        
        st.sidebar.markdown(f'<div class="section-header">🧭 {user_role} Navigation</div>', unsafe_allow_html=True)
        st.sidebar.markdown("---")
        
        if available_modules:
            selected_module = st.sidebar.selectbox("Select Module", available_modules)
            
            # Route to selected module
            if selected_module == "📊 Performance Dashboard":
                create_performance_dashboard(analyzer)
            elif selected_module == "🎓 Accreditation Analytics":
                create_accreditation_analytics_dashboard(analyzer)
            elif selected_module == "⚙️ System Settings":
                create_system_settings(analyzer)
            elif selected_module == "🔄 Approval Workflow":
                create_approval_workflow(analyzer)
            elif selected_module == "💾 Data Management":
                create_data_management_module(analyzer)
            elif selected_module == "🔍 RAG Data Management":
                create_rag_data_management(analyzer)
            elif selected_module == "📋 Document Analysis":
                create_document_analysis_module(analyzer)
            elif selected_module == "🤖 AI Reports":
                create_ai_analysis_reports(analyzer)
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.system_user = None
            st.session_state.user_role = None
            st.rerun()
        return
    
    # Main authentication page with enhanced UI
    st.markdown('<h1 class="main-header">🏛️ AI-Powered Institutional Approval Analytics System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">UGC & AICTE - Institutional Performance Tracking & Decision Support</h3>', unsafe_allow_html=True)
    
    # Authentication tabs
    auth_tabs = st.tabs(["🏛️ Institution Login", "🔐 System Login"])
    
    with auth_tabs[0]:
        create_institution_login(analyzer)
    
    with auth_tabs[1]:
        create_system_login(analyzer)
    
    # System overview with enhanced styling
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>🚀 System Overview</h4>
        <p>This AI-powered platform automates the analysis of institutional historical data, performance metrics, 
        and document compliance for UGC and AICTE approval processes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>🔒 Secure Access</h4>
        <p>Authorized UGC/AICTE personnel and registered institutions only. All activities are logged and monitored.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.success("✅ AI Analytics System Successfully Initialized!")
    
    # Display quick stats with enhanced UI
    st.markdown('<div class="section-header">📈 System Quick Stats</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_institutions = len(analyzer.historical_data['institution_id'].unique())
        st.metric("Total Institutions", total_institutions)
    
    with col2:
        years_data = len(analyzer.historical_data['year'].unique())
        st.metric("Years of Data", years_data)
    
    with col3:
        current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
        if len(current_year_data) > 0:
            avg_performance = current_year_data['performance_score'].mean()
            st.metric("Avg Performance Score", f"{avg_performance:.2f}/10")
    
    with col4:
        if len(current_year_data) > 0:
            approval_ready = (current_year_data['performance_score'] >= 6.0).sum()
            st.metric("Approval Ready", approval_ready)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d;'>
    <p><strong>UGC/AICTE Institutional Analytics Platform</strong> | AI-Powered Decision Support System</p>
    <p>Version 2.0 | For authorized use only | Data last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
