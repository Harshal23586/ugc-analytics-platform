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
        
        # Other parameters (simplified calculation)
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
        
        # Create other tables
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
        # Define categories at the start of the method
        institution_categories = [
            "Multi-disciplinary Education and Research-Intensive",
            "Research-Intensive", 
            "Teaching-Intensive",
            "Specialised Streams",
            "Vocational and Skill-Intensive",
            "Community Engagement & Service",
            "Rural & Remote location"
        ]

        heritage_categories = [
            "Old and Established",
            "New and Upcoming"
        ]
    
        n_institutions = 20
        years_of_data = 10
    
        institutions_data = []
    
        for inst_id in range(1, n_institutions + 1):
            # Assign categories PER INSTITUTION (not per year)
            institution_type = np.random.choice(institution_categories)
            heritage_type = np.random.choice(heritage_categories)
        
            # Set base characteristics based on categories
            if institution_type == "Research-Intensive":
                research_weight = 1.5
                teaching_weight = 0.8
            elif institution_type == "Teaching-Intensive":
                research_weight = 0.7
                teaching_weight = 1.3
            elif institution_type == "Rural & Remote location":
                research_weight = 0.6
                community_weight = 1.4
            else:
                research_weight = 1.0
                teaching_weight = 1.0
            
            if heritage_type == "Old and Established":
                establishment_year = np.random.randint(1950, 1990)
                stability_bonus = 0.2
            else:
                establishment_year = np.random.randint(2000, 2015)
                stability_bonus = 0.0
        
            for year_offset in range(years_of_data):
                year = 2023 - year_offset
            
                institution_data = {
                    # Basic info with categories
                    'institution_id': f'INST_{inst_id:04d}',
                    'institution_name': f'University {inst_id:03d}',
                    'year': year,
                    'institution_type': institution_type,  # ADD THIS
                    'heritage_category': heritage_type,    # ADD THIS
                    'established_year': establishment_year, # ADD THIS
                    'state': np.random.choice(['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Uttar Pradesh']),
                
                # Rest of your metrics with category-based adjustments...
                'research_publications': int(np.random.poisson(15 * research_weight)),
                'teaching_quality_score': np.random.uniform(6, 9) * teaching_weight,
                'community_engagement_score': np.random.uniform(5, 8) * (community_weight if 'community_weight' in locals() else 1.0),
                # ... other metrics
            }
            
                # Generate data according to Appendix 1 framework
                institution_data = {
                    # Basic info
                    'institution_id': f'INST_{inst_id:04d}',
                    'year': year,
                
                    # Appendix 1: Curriculum metrics
                    'curriculum_innovation_score': np.random.uniform(5, 10),
                    'student_feedback_score': np.random.uniform(6, 9),
                    'stakeholder_involvement_score': np.random.uniform(5, 8),
                    'lifelong_learning_initiatives': np.random.randint(1, 10),
                    'multidisciplinary_courses': np.random.randint(5, 20),
                
                    # Appendix 1: Faculty Resources
                    'faculty_selection_transparency': np.random.uniform(6, 9),
                    'faculty_diversity_index': np.random.uniform(5, 8),
                    'continuous_professional_dev': np.random.uniform(4, 9),
                    'social_inclusivity_measures': np.random.uniform(5, 8),
                
                    # Appendix 1: Learning and Teaching
                    'experiential_learning_score': np.random.uniform(5, 9),
                    'digital_technology_adoption': np.random.uniform(4, 9),
                    'research_oriented_teaching': np.random.uniform(5, 8),
                    'critical_thinking_focus': np.random.uniform(5, 9),
                
                    # Appendix 1: Research and Innovation
                    'interdisciplinary_research': np.random.uniform(4, 9),
                    'industry_collaboration_score': np.random.uniform(4, 8),
                    'patents_filed': np.random.randint(0, 15),
                    'research_publications': np.random.randint(5, 50),
                    'translational_research_score': np.random.uniform(3, 8),
                
                    # Appendix 1: Community Engagement
                    'community_projects_count': np.random.randint(2, 25),
                    'social_outreach_score': np.random.uniform(4, 9),
                    'rural_engagement_initiatives': np.random.randint(1, 12),
                
                    # Appendix 1: Green Initiatives
                    'renewable_energy_adoption': np.random.uniform(3, 9),
                    'waste_management_score': np.random.uniform(4, 9),
                    'carbon_footprint_reduction': np.random.uniform(3, 8),
                    'sdg_alignment_score': np.random.uniform(4, 9),
                
                    # Appendix 1: Governance and Administration
                    'egovernance_implementation': np.random.uniform(4, 9),
                    'grievance_redressal_efficiency': np.random.uniform(5, 9),
                    'internationalization_score': np.random.uniform(3, 8),
                    'gender_parity_ratio': np.random.uniform(0.3, 0.8),
                
                    # Appendix 1: Infrastructure Development
                    'digital_infrastructure_score': np.random.uniform(5, 9),
                    'research_lab_quality': np.random.uniform(4, 9),
                    'library_resources_score': np.random.uniform(5, 9),
                    'sports_facilities_score': np.random.uniform(4, 8),
                
                    # Appendix 1: Financial Resources and Management
                    'research_funding_utilization': np.random.uniform(4, 9),
                    'infrastructure_investment': np.random.uniform(3, 8),
                    'financial_sustainability': np.random.uniform(5, 9),
                    'csr_funding_attraction': np.random.uniform(2, 8),
                }
            
                # Calculate composite scores based on Appendix 1 framework
                institution_data['input_score'] = self.calculate_input_score(institution_data)
                institution_data['process_score'] = self.calculate_process_score(institution_data)
                institution_data['outcome_score'] = self.calculate_outcome_score(institution_data)
                institution_data['impact_score'] = self.calculate_impact_score(institution_data)
                institution_data['overall_score'] = self.calculate_overall_score(institution_data)
            
                institutions_data.append(institution_data)
    
        return pd.DataFrame(institutions_data)

    def calculate_input_score(self, data):
        """Calculate input score based on Appendix 1 framework"""
        weights = {
            'faculty_resources': 0.25,
            'infrastructure': 0.25,
            'financial_resources': 0.25,
            'curriculum_inputs': 0.25
        }
    
        score = (
            data['faculty_selection_transparency'] * weights['faculty_resources'] +
            data['digital_infrastructure_score'] * weights['infrastructure'] +
            data['financial_sustainability'] * weights['financial_resources'] +
            data['curriculum_innovation_score'] * weights['curriculum_inputs']
        )
        return score

    def calculate_process_score(self, data):
        """Calculate process score based on Appendix 1 framework"""
        weights = {
            'teaching_processes': 0.3,
            'research_processes': 0.3,
            'governance_processes': 0.2,
            'community_processes': 0.2
        }
    
        score = (
            data['experiential_learning_score'] * weights['teaching_processes'] +
            data['interdisciplinary_research'] * weights['research_processes'] +
            data['egovernance_implementation'] * weights['governance_processes'] +
            data['community_projects_count']/25 * 10 * weights['community_processes']
        )
        return score

    def calculate_outcome_score(self, data):
        """Calculate outcome score based on Appendix 1 framework"""
        weights = {
            'learning_outcomes': 0.4,
            'research_outcomes': 0.3,
            'skill_development': 0.3
        }
    
        score = (
            data['critical_thinking_focus'] * weights['learning_outcomes'] +
            min(data['research_publications']/50 * 10, 10) * weights['research_outcomes'] +
            data['lifelong_learning_initiatives']/10 * 10 * weights['skill_development']
        )
        return score

    def calculate_impact_score(self, data):
        """Calculate impact score based on Appendix 1 framework"""
        weights = {
            'societal_impact': 0.4,
            'environmental_impact': 0.3,
            'economic_impact': 0.3
        }
    
        score = (
            data['social_outreach_score'] * weights['societal_impact'] +
            data['carbon_footprint_reduction'] * weights['environmental_impact'] +
            data['industry_collaboration_score'] * weights['economic_impact']
        )
        return score
        
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
        """Update metrics based on Appendix 1 framework"""
        return {
            "input_parameters": {
                "weight": 0.25,
                "sub_metrics": {
                    "faculty_quality": 0.25,
                    "infrastructure": 0.25,
                    "financial_resources": 0.25,
                    "curriculum_design": 0.25
                }
            },
            "process_parameters": {
                "weight": 0.25,
                "sub_metrics": {
                    "teaching_learning": 0.3,
                    "research_innovation": 0.3,
                    "governance": 0.2,
                    "community_engagement": 0.2
                }
            },
            "outcome_parameters": {
                "weight": 0.25,
                "sub_metrics": {
                    "learning_outcomes": 0.4,
                    "research_outputs": 0.3,
                    "skill_development": 0.3
                }
            },
            "impact_parameters": {
                "weight": 0.25,
                "sub_metrics": {
                    "societal_impact": 0.4,
                    "environmental_impact": 0.3,
                    "economic_impact": 0.3
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

# Enhanced UI Functions
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

     # Navigation for institution users
    institution_tabs = st.tabs([
        "📤 Document Upload", 
        "📝 Data Submission", 
        "📊 My Submissions",
        "📋 Requirements Guide"
    ])
    
    with institution_tabs[0]:
        st.markdown('<div class="section-header">Document Upload Portal</div>', unsafe_allow_html=True)
        st.info("Upload required documents for approval processes")
        # Add document upload functionality here
        
    with institution_tabs[1]:
        st.markdown('<div class="section-header">Data Submission Form</div>', unsafe_allow_html=True)
        st.info("Submit institutional data and performance metrics")
        # Add data submission functionality here
        
    with institution_tabs[2]:
        st.markdown('<div class="section-header">My Submissions & Status</div>', unsafe_allow_html=True)
        # Add submissions view functionality here
        
    with institution_tabs[3]:
        st.markdown('<div class="section-header">Approval Requirements Guide</div>', unsafe_allow_html=True)
        # Add requirements guide functionality here
    
    # Get institution data
    institution_data = analyzer.historical_data[
        analyzer.historical_data['institution_id'] == user['institution_id']
    ].iloc[0] if not analyzer.historical_data[
        analyzer.historical_data['institution_id'] == user['institution_id']
    ].empty else None
    
    if institution_data is not None:
        # Create beautiful accreditation dashboard for the institution
        scores, overall_score, status = analyzer.accreditation_analyzer.create_accreditation_dashboard(
            institution_data, user['institution_name']
        )

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
        st.metric("Approval Eligibility Rate", f"{approval_rate:.1%}")
    
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
                st.info("System Settings Module - To be implemented")
            elif selected_module == "🔄 Approval Workflow":
                st.info("Approval Workflow Module - To be implemented")
            elif selected_module == "💾 Data Management":
                st.info("Data Management Module - To be implemented")
            elif selected_module == "🔍 RAG Data Management":
                st.info("RAG Data Management Module - To be implemented")
            elif selected_module == "📋 Document Analysis":
                st.info("Document Analysis Module - To be implemented")
            elif selected_module == "🤖 AI Reports":
                st.info("AI Reports Module - To be implemented")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.system_user = None
            st.session_state.user_role = None
            st.rerun()
        return
    
    # Main authentication page with enhanced UI
    st.markdown('<h1 class="main-header">🏛️ AI-Powered Institutional Approval Analytics System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">UGC & AICTE - Institutional Performance Tracking & Decision Support</h3>', unsafe_allow_html=True)

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
    # Authentication tabs
    auth_tabs = st.tabs(["🏛️ Institution Login", "🔐 System Login"])
    
    with auth_tabs[0]:
        create_institution_login(analyzer)
    
    with auth_tabs[1]:
        create_system_login(analyzer)
    
    
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
