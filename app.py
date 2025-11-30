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

# Optimized imports
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any, Optional
import hashlib
import sqlite3
import re

# RAG-specific imports
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize session state efficiently
def init_session_state():
    defaults = {
        'session_initialized': True,
        'institution_user': None,
        'user_role': None,
        'rag_analysis': None,
        'selected_institution': None,
        'system_user': None,
        'current_module': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Optimized RAG Classes
class RAGDocument:
    __slots__ = ['page_content', 'metadata']  # Memory optimization
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        # More efficient text splitting
        if not text.strip():
            return []
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Handle overlap
                if self.chunk_overlap > 0:
                    overlap_words = ' '.join(current_chunk).split()[-self.chunk_overlap//10:]
                    current_chunk = overlap_words + [sentence]
                    current_length = len(' '.join(current_chunk))
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

class SimpleVectorStore:
    __slots__ = ['embedding_model', 'documents', 'embeddings']  # Memory optimization
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = np.array([])
    
    def from_embeddings(self, text_embeddings):
        if not text_embeddings:
            return self
            
        texts, embeddings = zip(*text_embeddings)
        self.documents = list(texts)
        self.embeddings = np.array(embeddings)
        return self
    
    def similarity_search_with_score(self, query: str, k: int = 5):
        if not self.embeddings.size:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Use argpartition for better performance with large arrays
        if len(similarities) > k:
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        else:
            top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc = RAGDocument(
                    page_content=self.documents[idx],
                    metadata={"similarity_score": float(similarities[idx])}
                )
                results.append((doc, float(similarities[idx])))
        
        return results

# Page configuration
st.set_page_config(
    page_title="AI-Powered Institutional Approval System - UGC/AICTE",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RAGDataExtractor:
    def __init__(self):
        self.text_splitter = SimpleTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_store = None
        self.documents = []
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            except Exception as e:
                st.error(f"Error initializing embedding model: {e}")
                # Fallback: create a dummy model structure
                class DummyModel:
                    def encode(self, texts):
                        return np.random.randn(len(texts), 384)
                self._embedding_model = DummyModel()
        return self._embedding_model

    def build_vector_store(self, documents: List[RAGDocument]):
        if not documents:
            return None
        
        try:
            texts = [doc.page_content for doc in documents if doc.page_content.strip()]
            if not texts:
                return None
            
            embeddings = self.embedding_model.encode(texts)
            text_embeddings = list(zip(texts, embeddings))
            self.vector_store = SimpleVectorStore(self.embedding_model).from_embeddings(text_embeddings)
            self.documents = documents
            return self.vector_store
        except Exception as e:
            st.error(f"Error building vector store: {e}")
            return None
        
    def extract_text_from_file(self, file) -> str:
        text = ""
        file_extension = file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'pdf':
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or "" + "\n"
                    
            elif file_extension in ['doc', 'docx']:
                doc = docx.Document(file)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                    
            elif file_extension in ['txt']:
                text = str(file.getvalue(), 'utf-8', errors='ignore')
                
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file)
                text = df.to_string()
                
        except Exception as e:
            st.error(f"Error extracting text from {file.name}: {str(e)}")
            
        return text
    
    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text.strip()
    
    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        # Consolidated pattern matching for better performance
        patterns = {
            'curriculum_metrics': {
                'curriculum_innovation_score': r'curriculum.*innovation.*score[:\s]*(\d+(?:\.\d+)?)',
                'student_feedback_score': r'student.*feedback.*score[:\s]*(\d+(?:\.\d+)?)',
                'stakeholder_involvement_score': r'stakeholder.*involvement.*score[:\s]*(\d+(?:\.\d+)?)',
                'multidisciplinary_courses': r'multidisciplinary.*courses[:\s]*(\d+)'
            },
            'faculty_metrics': {
                'faculty_selection_transparency': r'faculty.*selection.*transparency[:\s]*(\d+(?:\.\d+)?)',
                'faculty_diversity_index': r'faculty.*diversity.*index[:\s]*(\d+(?:\.\d+)?)',
                'continuous_professional_dev': r'continuous.*professional.*development[:\s]*(\d+(?:\.\d+)?)'
            },
            'research_innovation_metrics': {
                'research_publications': r'research.*publications[:\s]*(\d+)',
                'patents_filed': r'patents.*filed[:\s]*(\d+)',
                'industry_collaboration_score': r'industry.*collaboration.*score[:\s]*(\d+(?:\.\d+)?)',
                'translational_research_score': r'translational.*research.*score[:\s]*(\d+(?:\.\d+)?)'
            }
        }
        
        data = {category: {} for category in patterns.keys()}
        data.update({
            'community_engagement_metrics': {},
            'green_initiatives_metrics': {},
            'governance_metrics': {},
            'infrastructure_metrics': {},
            'financial_metrics': {}
        })
    
        # Single pass through text for all patterns
        for category, category_patterns in patterns.items():
            for key, pattern in category_patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data[category][key] = match.group(1)
    
        # Extract contextual data
        contextual_patterns = [
            (r'library.*?(\d+(?:,\d+)*)\s*(?:volumes|books)', 'library_volumes', 'infrastructure_metrics'),
            (r'campus.*?(\d+(?:\.\d+)?)\s*(?:acres|hectares)', 'campus_area', 'infrastructure_metrics'),
            (r'financial.*?stability.*?(\d+(?:\.\d+)?)\s*(?:out of|/)', 'financial_stability_score', 'financial_metrics'),
            (r'digital.*?infrastructure.*?(\d+(?:\.\d+)?)\s*(?:out of|/)', 'digital_infrastructure_score', 'infrastructure_metrics'),
            (r'community.*?projects.*?(\d+)', 'community_projects', 'governance_metrics')
        ]
        
        for pattern, key, category in contextual_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data[category][key] = match.group(1)
    
        return data

    def query_documents(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        return self.vector_store.similarity_search_with_score(query, k=k) if self.vector_store else []

    def extract_comprehensive_data(self, uploaded_files: List) -> Dict[str, Any]:
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
            
                # Merge data efficiently
                for category, values in file_data.items():
                    if category in all_structured_data:
                        all_structured_data[category].update(values)
            
                all_structured_data['file_names'].append(file.name)
            
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                continue
    
        if documents:
            self.build_vector_store(documents)
    
        all_structured_data['raw_text'] = all_text
        return all_structured_data

class InstitutionalAIAnalyzer:
    __slots__ = ['conn', '_historical_data', '_performance_metrics', '_document_requirements', 
                'rag_extractor', 'institution_categories', 'heritage_categories']
    
    def __init__(self):
        self.institution_categories = [
            "Multi-disciplinary Education and Research-Intensive",
            "Research-Intensive", 
            "Teaching-Intensive",
            "Specialised Streams",
            "Vocational and Skill-Intensive",
            "Community Engagement & Service",
            "Rural & Remote location"
        ]
        
        self.heritage_categories = [
            "Old and Established",
            "New and Upcoming"
        ]
        
        self.init_database()
        self.rag_extractor = RAGDataExtractor()
        
        # Lazy loading attributes
        self._historical_data = None
        self._performance_metrics = None
        self._document_requirements = None
        
        # Create initial data
        self.create_dummy_data()
    
    @property
    def historical_data(self):
        if self._historical_data is None:
            self._historical_data = self.load_or_generate_data()
        return self._historical_data
    
    @property
    def performance_metrics(self):
        if self._performance_metrics is None:
            self._performance_metrics = self.define_performance_metrics()
        return self._performance_metrics
    
    @property
    def document_requirements(self):
        if self._document_requirements is None:
            self._document_requirements = self.define_document_requirements()
        return self._document_requirements
        
    def init_database(self):
        self.conn = sqlite3.connect('institutions.db', check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        tables = {
            'institutions': '''
                CREATE TABLE IF NOT EXISTS institutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    institution_id TEXT UNIQUE,
                    institution_name TEXT,
                    year INTEGER,
                    institution_type TEXT,
                    heritage_category TEXT,
                    state TEXT,
                    established_year INTEGER,
                    curriculum_innovation_score REAL,
                    student_feedback_score REAL,
                    stakeholder_involvement_score REAL,
                    lifelong_learning_initiatives INTEGER,
                    multidisciplinary_courses INTEGER,
                    faculty_selection_transparency REAL,
                    faculty_diversity_index REAL,
                    continuous_professional_dev REAL,
                    social_inclusivity_measures REAL,
                    experiential_learning_score REAL,
                    digital_technology_adoption REAL,
                    research_oriented_teaching REAL,
                    critical_thinking_focus REAL,
                    interdisciplinary_research REAL,
                    industry_collaboration_score REAL,
                    patents_filed INTEGER,
                    research_publications INTEGER,
                    translational_research_score REAL,
                    community_projects_count INTEGER,
                    social_outreach_score REAL,
                    rural_engagement_initiatives INTEGER,
                    renewable_energy_adoption REAL,
                    waste_management_score REAL,
                    carbon_footprint_reduction REAL,
                    sdg_alignment_score REAL,
                    egovernance_implementation REAL,
                    grievance_redressal_efficiency REAL,
                    internationalization_score REAL,
                    gender_parity_ratio REAL,
                    digital_infrastructure_score REAL,
                    research_lab_quality REAL,
                    library_resources_score REAL,
                    sports_facilities_score REAL,
                    research_funding_utilization REAL,
                    infrastructure_investment REAL,
                    financial_sustainability REAL,
                    csr_funding_attraction REAL,
                    input_score REAL,
                    process_score REAL,
                    outcome_score REAL,
                    impact_score REAL,
                    overall_score REAL,
                    approval_recommendation TEXT,
                    risk_level TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'institution_documents': '''
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
            ''',
            'rag_analysis': '''
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
            ''',
            'institution_submissions': '''
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
            ''',
            'institution_users': '''
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
            ''',
            'system_users': '''
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
            '''
        }
        
        for table_name, table_sql in tables.items():
            cursor.execute(table_sql)
        
        self.conn.commit()
    
    def create_dummy_data(self):
        """Create all dummy data in one method"""
        self.create_dummy_system_users()
        self.create_dummy_institution_users()
    
    def create_dummy_system_users(self):
        system_users = [
            ('ugc_officer', 'ugc123', 'UGC Department Officer', 'ugc.officer@ugc.gov.in', 'UGC Officer', 'UGC Approval Division'),
            ('aicte_officer', 'aicte123', 'AICTE Department Officer', 'aicte.officer@aicte.gov.in', 'AICTE Officer', 'AICTE Approval Division'),
            ('system_admin', 'admin123', 'System Administrator', 'admin@ugc-aicte.gov.in', 'System Admin', 'IT Department'),
            ('review_committee', 'review123', 'Review Committee Member', 'review.committee@ugc-aicte.gov.in', 'Review Committee', 'Review Committee')
        ]
        
        cursor = self.conn.cursor()
        for username, password, full_name, email, role, department in system_users:
            cursor.execute('SELECT 1 FROM system_users WHERE username = ?', (username,))
            if not cursor.fetchone():
                self.create_system_user(username, password, full_name, email, role, department)
    
    def create_dummy_institution_users(self):
        dummy_users = [
            ('HEI_01', 'inst001_admin', 'password123', 'Dr. Rajesh Kumar', 'rajesh.kumar@iitvaranasi.edu.in', '+91-9876543210'),
            ('HEI_07', 'inst007_registrar', 'testpass456', 'Ms. Priya Sharma', 'priya.sharma@himalayanrural.edu.in', '+91-8765432109')
        ]
        
        for institution_id, username, password, contact_person, email, phone in dummy_users:
            cursor = self.conn.cursor()
            cursor.execute('SELECT 1 FROM institution_users WHERE username = ?', (username,))
            if not cursor.fetchone():
                self.create_institution_user(institution_id, username, password, contact_person, email, phone)

    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def create_institution_user(self, institution_id: str, username: str, password: str, 
                          contact_person: str, email: str, phone: str):
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

    def create_system_user(self, username: str, password: str, full_name: str, 
                          email: str, role: str, department: str):
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

    def analyze_documents_with_rag(self, institution_id: str, uploaded_files: List) -> Dict[str, Any]:
        try:
            extracted_data = self.rag_extractor.extract_comprehensive_data(uploaded_files)
        
            if not extracted_data:
                extracted_data = self._get_default_extracted_data(uploaded_files)
        
            # Batch database operation
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO rag_analysis 
                (institution_id, analysis_type, extracted_data, confidence_score)
                VALUES (?, ?, ?, ?)
            ''', (institution_id, 'document_analysis', json.dumps(extracted_data), 0.85))
            self.conn.commit()
        
            ai_insights = self.generate_ai_insights(extracted_data)
        
            return {
                'extracted_data': extracted_data,
                'ai_insights': ai_insights,
                'confidence_score': 0.85,
                'status': 'Analysis Complete'
            }
        
        except Exception as e:
            st.error(f"Error in RAG analysis: {str(e)}")
            return self._get_error_response(uploaded_files)

    def _get_default_extracted_data(self, uploaded_files):
        return {
            'academic_metrics': {}, 'research_metrics': {}, 'infrastructure_metrics': {},
            'governance_metrics': {}, 'student_metrics': {}, 'financial_metrics': {},
            'raw_text': "", 'file_names': [f.name for f in uploaded_files]
        }

    def _get_error_response(self, uploaded_files):
        return {
            'extracted_data': self._get_default_extracted_data(uploaded_files),
            'ai_insights': {
                'strengths': [], 'weaknesses': [], 'recommendations': [],
                'risk_assessment': {'score': 5.0, 'level': 'Medium', 'factors': []},
                'compliance_status': {}
            },
            'confidence_score': 0.0, 'status': 'Analysis Failed'
        }

    def generate_ai_insights(self, extracted_data: Dict) -> Dict[str, Any]:
        insights = {
            'strengths': [], 'weaknesses': [], 'recommendations': [],
            'risk_assessment': {}, 'compliance_status': {}
        }
    
        academic_data = extracted_data.get('academic_metrics', {})
        research_data = extracted_data.get('research_metrics', {})
        financial_data = extracted_data.get('financial_metrics', {})
        
        # Strengths
        if academic_data.get('naac_grade') in ['A++', 'A+', 'A']:
            insights['strengths'].append(f"Strong NAAC accreditation: {academic_data['naac_grade']}")
        if research_data.get('research_publications', 0) > 50:
            insights['strengths'].append("Robust research publication output")
        
        # Weaknesses
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
        score = 5.0
        
        academic_data = extracted_data.get('academic_metrics', {})
        research_data = extracted_data.get('research_metrics', {})
        financial_data = extracted_data.get('financial_metrics', {})
        
        # Adjust score based on data
        adjustments = [
            (academic_data.get('naac_grade') in ['A++', 'A+', 'A'], -1.5),
            (research_data.get('research_publications', 0) > 50, -1.0),
            (financial_data.get('financial_stability_score', 0) > 7, -1.0),
            (academic_data.get('student_faculty_ratio', 0) > 25, 1.5),
            (research_data.get('patents_filed', 0) < 2, 1.0),
            (not academic_data.get('nirf_ranking'), 0.5)
        ]
        
        for condition, adjustment in adjustments:
            if condition:
                score += adjustment
        
        return max(1.0, min(10.0, score))
    
    def identify_risk_factors(self, extracted_data: Dict) -> List[str]:
        risk_factors = []
        academic_data = extracted_data.get('academic_metrics', {})
        research_data = extracted_data.get('research_metrics', {})
        
        conditions = [
            (academic_data.get('student_faculty_ratio', 0) > 25, "High student-faculty ratio affecting quality"),
            (research_data.get('industry_collaborations', 0) < 2, "Limited industry exposure"),
            (not academic_data.get('naac_grade'), "No NAAC accreditation")
        ]
        
        for condition, message in conditions:
            if condition:
                risk_factors.append(message)
        
        return risk_factors
    
    def load_or_generate_data(self):
        try:
            df = pd.read_sql('SELECT * FROM institutions', self.conn)
            if len(df) > 0:
                return df
        except:
            pass
        
        sample_data = self.generate_comprehensive_historical_data()
        sample_data.to_sql('institutions', self.conn, if_exists='replace', index=False)
        return sample_data
    
    def define_performance_metrics(self) -> Dict[str, Dict]:
        return {
            "input_parameters": {
                "weight": 0.25,
                "sub_metrics": {
                    "faculty_quality": 0.25, "infrastructure": 0.25,
                    "financial_resources": 0.25, "curriculum_design": 0.25
                }
            },
            "process_parameters": {
                "weight": 0.25,
                "sub_metrics": {
                    "teaching_learning": 0.3, "research_innovation": 0.3,
                    "governance": 0.2, "community_engagement": 0.2
                }
            },
            "outcome_parameters": {
                "weight": 0.25,
                "sub_metrics": {
                    "learning_outcomes": 0.4, "research_outputs": 0.3, "skill_development": 0.3
                }
            },
            "impact_parameters": {
                "weight": 0.25,
                "sub_metrics": {
                    "societal_impact": 0.4, "environmental_impact": 0.3, "economic_impact": 0.3
                }
            }
        }
    
    def define_document_requirements(self) -> Dict[str, Dict]:
        return {
            "new_approval": {
                "mandatory": [
                    "affidavit_legal_status", "land_documents", "building_plan_approval",
                    "infrastructure_details", "financial_solvency_certificate",
                    "faculty_recruitment_plan", "academic_curriculum", "governance_structure",
                    "curriculum_innovation_plan", "faculty_development_plan"
                ],
                "supporting": [
                    "feasibility_report", "market_demand_analysis", "five_year_development_plan",
                    "industry_partnerships", "research_facilities_plan", "community_engagement_plan",
                    "sustainability_plan", "digital_infrastructure_plan"
                ]
            },
            "renewal_approval": {
                "mandatory": [
                    "previous_approval_letters", "annual_reports", "financial_audit_reports",
                    "faculty_student_data", "infrastructure_utilization", "academic_performance",
                    "research_publications", "community_projects_report", "governance_audit"
                ],
                "supporting": [
                    "naac_accreditation", "nirf_data", "placement_records", 
                    "social_impact_reports", "environmental_sustainability_report",
                    "stakeholder_feedback", "alumni_engagement_report"
                ]
            },
            "expansion_approval": {
                "mandatory": [
                    "current_status_report", "expansion_justification", "additional_infrastructure",
                    "enhanced_faculty_plan", "financial_viability", "market_analysis",
                    "curriculum_expansion_plan", "infrastructure_development_plan"
                ],
                "supporting": [
                    "stakeholder_feedback", "alumni_support", "industry_demand",
                    "government_schemes_participation", "research_collaboration_plan"
                ]
            }
        }
        
    def generate_comprehensive_historical_data(self) -> pd.DataFrame:
        np.random.seed(42)
    
        institutions_list = [
            {'institution_id': 'HEI_01', 'institution_name': 'Indian Institute of Technology, Varanasi', 'institution_type': 'Multi-disciplinary Education and Research-Intensive', 'heritage_category': 'Old and Established', 'state': 'Uttar Pradesh', 'established_year': 1959},
            {'institution_id': 'HEI_02', 'institution_name': 'National Institute of Technology, Srinagar', 'institution_type': 'Research-Intensive', 'heritage_category': 'Old and Established', 'state': 'Jammu & Kashmir', 'established_year': 1960},
            # ... (include all 20 institutions from original code)
        ]

        institutions_data = []
        years_of_data = 10

        for inst_info in institutions_list:
            institution_type = inst_info['institution_type']
            heritage_type = inst_info['heritage_category']
        
            # Set weights based on institution type
            weights = self._get_institution_weights(institution_type, heritage_type)
            research_weight, teaching_weight, community_weight, stability_bonus = weights
        
            for year_offset in range(years_of_data):
                year = 2023 - year_offset
                improvement_factor = 1.0 + (year_offset * 0.02)

                institution_data = self._generate_institution_data(
                    inst_info, year, improvement_factor, 
                    research_weight, teaching_weight, community_weight, stability_bonus
                )
                institutions_data.append(institution_data)
    
        return pd.DataFrame(institutions_data)

    def _get_institution_weights(self, institution_type, heritage_type):
        """Get performance weights based on institution type"""
        weight_map = {
            "Research-Intensive": (1.5, 0.8, 0.7),
            "Teaching-Intensive": (0.7, 1.3, 0.9),
            "Community Engagement & Service": (0.6, 0.9, 1.4),
            "Rural & Remote location": (0.5, 0.8, 1.3),
            "Vocational and Skill-Intensive": (0.4, 1.2, 1.1)
        }
        
        research_weight, teaching_weight, community_weight = weight_map.get(
            institution_type, (1.0, 1.0, 1.0)
        )
        stability_bonus = 0.2 if heritage_type == "Old and Established" else 0.0
        
        return research_weight, teaching_weight, community_weight, stability_bonus

    def _generate_institution_data(self, inst_info, year, improvement_factor, 
                                 research_weight, teaching_weight, community_weight, stability_bonus):
        """Generate data for a single institution year"""
        base_data = {
            'institution_id': inst_info['institution_id'],
            'institution_name': inst_info['institution_name'],
            'year': year,
            'institution_type': inst_info['institution_type'],
            'heritage_category': inst_info['heritage_category'],
            'established_year': inst_info['established_year'],
            'state': inst_info['state'],
        }
        
        # Generate scores with weights and improvements
        scores = self._generate_scores(improvement_factor, research_weight, teaching_weight, community_weight)
        
        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(scores)
        
        # Combine all data
        institution_data = {**base_data, **scores, **composite_scores}
        institution_data['approval_recommendation'] = self.generate_approval_recommendation(institution_data['overall_score'])
        institution_data['risk_level'] = self.assess_risk_level(institution_data['overall_score'])
        
        return institution_data

    def _generate_scores(self, improvement_factor, research_weight, teaching_weight, community_weight):
        """Generate individual performance scores"""
        return {
            # Curriculum parameters
            'curriculum_innovation_score': round(np.random.uniform(5, 9) * improvement_factor, 2),
            'student_feedback_score': round(np.random.uniform(6, 9) * improvement_factor, 2),
            'stakeholder_involvement_score': round(np.random.uniform(5, 8) * improvement_factor, 2),
            'lifelong_learning_initiatives': np.random.randint(1, 10),
            'multidisciplinary_courses': np.random.randint(5, 20),
            
            # Faculty parameters
            'faculty_selection_transparency': round(np.random.uniform(6, 9) * improvement_factor, 2),
            'faculty_diversity_index': round(np.random.uniform(5, 8) * improvement_factor, 2),
            'continuous_professional_dev': round(np.random.uniform(4, 9) * improvement_factor, 2),
            'social_inclusivity_measures': round(np.random.uniform(5, 8) * improvement_factor, 2),
            
            # Learning and Teaching
            'experiential_learning_score': round(np.random.uniform(5, 9) * teaching_weight * improvement_factor, 2),
            'digital_technology_adoption': round(np.random.uniform(4, 9) * improvement_factor, 2),
            'research_oriented_teaching': round(np.random.uniform(5, 8) * research_weight * improvement_factor, 2),
            'critical_thinking_focus': round(np.random.uniform(5, 9) * improvement_factor, 2),
            
            # Research and Innovation
            'interdisciplinary_research': round(np.random.uniform(4, 9) * research_weight * improvement_factor, 2),
            'industry_collaboration_score': round(np.random.uniform(4, 8) * improvement_factor, 2),
            'patents_filed': int(np.random.poisson(3 * research_weight)),
            'research_publications': int(np.random.poisson(15 * research_weight)),
            'translational_research_score': round(np.random.uniform(3, 8) * research_weight * improvement_factor, 2),
            
            # Community Engagement
            'community_projects_count': int(np.random.poisson(8 * community_weight)),
            'social_outreach_score': round(np.random.uniform(4, 9) * community_weight * improvement_factor, 2),
            'rural_engagement_initiatives': np.random.randint(1, 12),
            
            # Green Initiatives
            'renewable_energy_adoption': round(np.random.uniform(3, 9) * improvement_factor, 2),
            'waste_management_score': round(np.random.uniform(4, 9) * improvement_factor, 2),
            'carbon_footprint_reduction': round(np.random.uniform(3, 8) * improvement_factor, 2),
            'sdg_alignment_score': round(np.random.uniform(4, 9) * improvement_factor, 2),
            
            # Governance
            'egovernance_implementation': round(np.random.uniform(4, 9) * improvement_factor, 2),
            'grievance_redressal_efficiency': round(np.random.uniform(5, 9) * improvement_factor, 2),
            'internationalization_score': round(np.random.uniform(3, 8) * improvement_factor, 2),
            'gender_parity_ratio': round(np.random.uniform(0.3, 0.8), 2),
            
            # Infrastructure
            'digital_infrastructure_score': round(np.random.uniform(5, 9) * improvement_factor, 2),
            'research_lab_quality': round(np.random.uniform(4, 9) * research_weight * improvement_factor, 2),
            'library_resources_score': round(np.random.uniform(5, 9) * improvement_factor, 2),
            'sports_facilities_score': round(np.random.uniform(4, 8) * improvement_factor, 2),
            
            # Financial
            'research_funding_utilization': round(np.random.uniform(4, 9) * research_weight * improvement_factor, 2),
            'infrastructure_investment': round(np.random.uniform(3, 8) * improvement_factor, 2),
            'financial_sustainability': round(np.random.uniform(5, 9) * improvement_factor, 2),
            'csr_funding_attraction': round(np.random.uniform(2, 8) * improvement_factor, 2),
        }

    def _calculate_composite_scores(self, data):
        """Calculate composite scores from individual metrics"""
        return {
            'input_score': self.calculate_input_score(data),
            'process_score': self.calculate_process_score(data),
            'outcome_score': self.calculate_outcome_score(data),
            'impact_score': self.calculate_impact_score(data),
            'overall_score': self.calculate_overall_score(data)
        }

    def calculate_input_score(self, data):
        weights = {'faculty_resources': 0.25, 'infrastructure': 0.25, 'financial_resources': 0.25, 'curriculum_inputs': 0.25}
        score = (
            data['faculty_selection_transparency'] * weights['faculty_resources'] +
            data['digital_infrastructure_score'] * weights['infrastructure'] +
            data['financial_sustainability'] * weights['financial_resources'] +
            data['curriculum_innovation_score'] * weights['curriculum_inputs']
        )
        return round(score, 2)

    def calculate_process_score(self, data):
        weights = {'teaching_processes': 0.3, 'research_processes': 0.3, 'governance_processes': 0.2, 'community_processes': 0.2}
        score = (
            data['experiential_learning_score'] * weights['teaching_processes'] +
            data['interdisciplinary_research'] * weights['research_processes'] +
            data['egovernance_implementation'] * weights['governance_processes'] +
            min(data['community_projects_count']/25 * 10, 10) * weights['community_processes']
        )
        return round(score, 2)

    def calculate_outcome_score(self, data):
        weights = {'learning_outcomes': 0.4, 'research_outcomes': 0.3, 'skill_development': 0.3}
        score = (
            data['critical_thinking_focus'] * weights['learning_outcomes'] +
            min(data['research_publications']/50 * 10, 10) * weights['research_outcomes'] +
            min(data['lifelong_learning_initiatives']/10 * 10, 10) * weights['skill_development']
        )
        return round(score, 2)

    def calculate_impact_score(self, data):
        weights = {'societal_impact': 0.4, 'environmental_impact': 0.3, 'economic_impact': 0.3}
        score = (
            data['social_outreach_score'] * weights['societal_impact'] +
            data['carbon_footprint_reduction'] * weights['environmental_impact'] +
            data['industry_collaboration_score'] * weights['economic_impact']
        )
        return round(score, 2)

    def calculate_overall_score(self, data):
        weights = {'input': 0.25, 'process': 0.25, 'outcome': 0.25, 'impact': 0.25}
        score = (
            data['input_score'] * weights['input'] +
            data['process_score'] * weights['process'] +
            data['outcome_score'] * weights['outcome'] +
            data['impact_score'] * weights['impact']
        )
        return round(score, 2)

    def generate_approval_recommendation(self, performance_score: float) -> str:
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

    def assess_risk_level(self, performance_score: float) -> str:
        if performance_score >= 8.0:
            return "Low Risk"
        elif performance_score >= 6.5:
            return "Medium Risk"
        elif performance_score >= 5.0:
            return "High Risk"
        else:
            return "Critical Risk"

    def authenticate_institution_user(self, username: str, password: str) -> Dict:
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
        if user and user['password_hash'] == self.hash_password(password):
            return {
                'institution_id': user['institution_id'],
                'institution_name': user['institution_name'],
                'username': user['username'],
                'role': user.get('role', 'Institution'),
                'contact_person': user.get('contact_person', ''),
                'email': user.get('email', '')
            }
        return None

    def authenticate_system_user(self, username: str, password: str, role: str) -> Dict:
        if not username or not password:
            return None
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM system_users 
            WHERE username = ? AND role = ? AND is_active = 1
        ''', (username, role))
    
        user = cursor.fetchone()
        if user and user['password_hash'] == self.hash_password(password):
            return {
                'username': user['username'],
                'full_name': user['full_name'],
                'role': user['role'],
                'department': user['department'],
                'email': user['email']
            }
        return None

    def save_institution_submission(self, institution_id: str, submission_type: str, submission_data: Dict):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO institution_submissions 
            (institution_id, submission_type, submission_data)
            VALUES (?, ?, ?)
        ''', (institution_id, submission_type, json.dumps(submission_data)))
        self.conn.commit()

    def get_institution_submissions(self, institution_id: str) -> pd.DataFrame:
        return pd.read_sql('SELECT * FROM institution_submissions WHERE institution_id = ? ORDER BY submitted_date DESC', 
                          self.conn, params=(institution_id,))

    def save_uploaded_documents(self, institution_id: str, uploaded_files: List, document_types: List[str]):
        """Optimized batch insert"""
        cursor = self.conn.cursor()
        documents_data = [
            (institution_id, file.name, doc_type, 'Uploaded') 
            for file, doc_type in zip(uploaded_files, document_types)
        ]
        
        cursor.executemany('''
            INSERT INTO institution_documents (institution_id, document_name, document_type, status)
            VALUES (?, ?, ?, ?)
        ''', documents_data)
        self.conn.commit()
    
    def get_institution_documents(self, institution_id: str) -> pd.DataFrame:
        return pd.read_sql('SELECT * FROM institution_documents WHERE institution_id = ? ORDER BY upload_date DESC', 
                          self.conn, params=(institution_id,))
    
    def analyze_document_sufficiency(self, uploaded_docs: List[str], approval_type: str) -> Dict:
        requirements = self.document_requirements[approval_type]
        
        # Use sets for faster lookups
        uploaded_set = {doc.lower() for doc in uploaded_docs}
        mandatory_set = {doc.lower() for doc in requirements['mandatory']}
        supporting_set = {doc.lower() for doc in requirements['supporting']}
        
        mandatory_present = len([doc for doc in mandatory_set if any(doc in uploaded for uploaded in uploaded_set)])
        supporting_present = len([doc for doc in supporting_set if any(doc in uploaded for uploaded in uploaded_set)])
        
        total_mandatory = len(requirements['mandatory'])
        total_supporting = len(requirements['supporting'])
        
        mandatory_sufficiency = (mandatory_present / total_mandatory) * 100 if total_mandatory > 0 else 0
        overall_sufficiency = ((mandatory_present + supporting_present) / 
                             (total_mandatory + total_supporting)) * 100 if (total_mandatory + total_supporting) > 0 else 0
        
        return {
            'mandatory_sufficiency': mandatory_sufficiency,
            'overall_sufficiency': overall_sufficiency,
            'missing_mandatory': [doc for doc in requirements['mandatory'] if not any(doc.lower() in uploaded for uploaded in uploaded_set)],
            'missing_supporting': [doc for doc in requirements['supporting'] if not any(doc.lower() in uploaded for uploaded in uploaded_set)],
            'recommendations': self.generate_document_recommendations(mandatory_sufficiency)
        }
    
    def generate_document_recommendations(self, mandatory_sufficiency: float) -> List[str]:
        if mandatory_sufficiency < 80:
            return ["Critical documents missing - application cannot be processed"]
        elif mandatory_sufficiency < 100:
            return ["Upload all mandatory documents to proceed with approval process"]
        else:
            return ["All mandatory documents present - ready for comprehensive evaluation"]
    
    def generate_comprehensive_report(self, institution_id: str) -> Dict[str, Any]:
        inst_data = self.historical_data[self.historical_data['institution_id'] == institution_id]
        
        if inst_data.empty:
            return {"error": "Institution not found"}
        
        latest_data = inst_data[inst_data['year'] == inst_data['year'].max()].iloc[0]
        historical_trend = inst_data.groupby('year')['overall_score'].mean()
        
        # Determine trend
        if len(historical_trend) > 1:
            trend_analysis = "Improving" if historical_trend.iloc[-1] > historical_trend.iloc[-2] else "Declining" if historical_trend.iloc[-1] < historical_trend.iloc[-2] else "Stable"
        else:
            trend_analysis = "Insufficient Data"
        
        similar_institutions = self.find_similar_institutions(institution_id)
        
        return {
            "institution_info": {
                "name": latest_data['institution_name'],
                "type": latest_data['institution_type'],
                "heritage": latest_data['heritage_category'],
                "state": latest_data['state'],
                "established": latest_data['established_year']
            },
            "performance_analysis": {
                "current_score": latest_data['overall_score'],
                "input_score": latest_data['input_score'],
                "process_score": latest_data['process_score'],
                "outcome_score": latest_data['outcome_score'],
                "impact_score": latest_data['impact_score'],
                "historical_trend": historical_trend.to_dict(),
                "trend_analysis": trend_analysis,
                "approval_recommendation": latest_data['approval_recommendation'],
                "risk_level": latest_data['risk_level']
            },
            "strengths": self.identify_strengths(latest_data),
            "weaknesses": self.identify_weaknesses(latest_data),
            "comparative_analysis": similar_institutions,
            "ai_recommendations": self.generate_ai_recommendations(latest_data)
        }
    
    def find_similar_institutions(self, institution_id: str) -> Dict:
        inst_data = self.historical_data[self.historical_data['institution_id'] == institution_id]
        
        if inst_data.empty:
            return {}
        
        latest_data = inst_data[inst_data['year'] == inst_data['year'].max()].iloc[0]
        
        similar_inst = self.historical_data[
            (self.historical_data['institution_type'] == latest_data['institution_type']) &
            (self.historical_data['year'] == latest_data['year']) &
            (self.historical_data['institution_id'] != institution_id)
        ]
        
        benchmark_data = similar_inst.nlargest(5, 'overall_score')[['institution_name', 'overall_score', 'approval_recommendation']].to_dict('records') if len(similar_inst) > 0 else []
        
        return {
            "benchmark_institutions": benchmark_data,
            "performance_percentile": self.calculate_performance_percentile(latest_data['overall_score'], latest_data['institution_type'])
        }
    
    def calculate_performance_percentile(self, score: float, inst_type: str) -> float:
        type_data = self.historical_data[
            (self.historical_data['institution_type'] == inst_type) &
            (self.historical_data['year'] == 2023)
        ]
        
        return (type_data['overall_score'] < score).mean() * 100 if len(type_data) > 0 else 50.0
    
    def identify_strengths(self, institution_data: pd.Series) -> List[str]:
        strengths = []
        conditions = [
            (institution_data['overall_score'] >= 8.0, f"Excellent Overall Performance: {institution_data['overall_score']:.2f}/10"),
            (institution_data['input_score'] >= 8.0, f"Strong Input Parameters: {institution_data['input_score']:.2f}/10"),
            (institution_data['process_score'] >= 8.0, f"Excellent Process Implementation: {institution_data['process_score']:.2f}/10"),
            (institution_data['research_publications'] > 20, f"Robust Research Output: {institution_data['research_publications']} publications"),
            (institution_data['community_projects_count'] > 10, f"Strong Community Engagement: {institution_data['community_projects_count']} projects"),
            (institution_data['digital_infrastructure_score'] > 8.0, "Advanced Digital Infrastructure")
        ]
        
        for condition, strength in conditions:
            if condition:
                strengths.append(strength)
                
        return strengths
    
    def identify_weaknesses(self, institution_data: pd.Series) -> List[str]:
        weaknesses = []
        conditions = [
            (institution_data['overall_score'] < 6.0, f"Low Overall Performance: {institution_data['overall_score']:.2f}/10"),
            (institution_data['outcome_score'] < 6.0, f"Weak Outcome Parameters: {institution_data['outcome_score']:.2f}/10"),
            (institution_data['research_publications'] < 5, f"Inadequate Research Output: {institution_data['research_publications']} publications"),
            (institution_data['patents_filed'] < 2, "Limited Innovation and Patent Filings"),
            (institution_data['financial_sustainability'] < 6.0, f"Financial Sustainability Concerns: {institution_data['financial_sustainability']:.2f}/10")
        ]
        
        for condition, weakness in conditions:
            if condition:
                weaknesses.append(weakness)
                
        return weaknesses
    
    def generate_ai_recommendations(self, institution_data: pd.Series) -> List[str]:
        recommendations = []
        conditions = [
            (institution_data['research_publications'] < 10, "Establish research promotion policy and faculty development programs"),
            (institution_data['patents_filed'] < 3, "Strengthen innovation ecosystem and IPR culture"),
            (institution_data['financial_sustainability'] < 7.0, "Diversify funding sources and improve financial planning"),
            (institution_data['digital_infrastructure_score'] < 7.0, "Invest in digital infrastructure and e-learning platforms"),
            (institution_data['community_projects_count'] < 5, "Enhance community engagement and social outreach programs")
        ]
        
        for condition, recommendation in conditions:
            if condition:
                recommendations.append(recommendation)
        
        return recommendations

# UI Components (Optimized)
def create_institution_login(analyzer):
    st.header("🏛️ Institution Portal Login")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Existing Institution Users")
        username = st.text_input("Username", key="inst_login_username")
        password = st.text_input("Password", type="password", key="inst_login_password")
        
        if st.button("Login", key="inst_login_button"):
            if user := analyzer.authenticate_institution_user(username, password):
                st.session_state.institution_user = user
                st.session_state.user_role = "Institution"
                st.success(f"Welcome, {user['contact_person']} from {user['institution_name']}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with col2:
        st.subheader("New Institution Registration")
        
        available_institutions = analyzer.historical_data[
            analyzer.historical_data['year'] == 2023
        ][['institution_id', 'institution_name']].drop_duplicates()
        
        selected_institution = st.selectbox(
            "Select Your Institution",
            available_institutions['institution_id'].tolist(),
            format_func=lambda x: available_institutions[available_institutions['institution_id'] == x]['institution_name'].iloc[0],
            key="inst_reg_institution"
        )
        
        new_username = st.text_input("Choose Username", key="inst_reg_username")
        new_password = st.text_input("Choose Password", type="password", key="inst_reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="inst_reg_confirm")
        contact_person = st.text_input("Contact Person Name", key="inst_reg_contact")
        email = st.text_input("Email Address", key="inst_reg_email")
        
        if st.button("Register Institution Account", key="inst_reg_button"):
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            elif not all([new_username, new_password, contact_person, email]):
                st.error("Please fill all required fields!")
            elif analyzer.create_institution_user(selected_institution, new_username, new_password, contact_person, email, ""):
                st.success("Institution account created successfully! You can now login.")
            else:
                st.error("Username already exists. Please choose a different username.")

def create_system_login(analyzer):
    st.header("🔐 System Login")
    
    role = st.selectbox("Select Your Role", ["UGC Officer", "AICTE Officer", "System Admin", "Review Committee"], key="system_login_role")
    username = st.text_input("Username", key="system_login_username")
    password = st.text_input("Password", type="password", key="system_login_password")
    
    if st.button("Login", key="system_login_button"):
        if user := analyzer.authenticate_system_user(username, password, role):
            st.session_state.system_user = user
            st.session_state.user_role = role
            st.success(f"Welcome, {user['full_name']} ({role})!")
            st.rerun()
        else:
            st.error("Invalid credentials for selected role!")

# ... (Other UI components remain similar but can be optimized further)

def get_available_modules(user_role):
    module_map = {
        "Institution": ["🏛️ Institution Portal"],
        "System Admin": ["📊 Performance Dashboard", "⚙️ System Settings"],
        "UGC Officer": ["🔄 Approval Workflow", "💾 Data Management", "🔍 RAG Data Analysis", "📋 Document Analysis", "🤖 AI Reports", "📊 Performance Dashboard"],
        "AICTE Officer": ["🔄 Approval Workflow", "💾 Data Management", "🔍 RAG Data Analysis", "📋 Document Analysis", "🤖 AI Reports", "📊 Performance Dashboard"],
        "Review Committee": ["🤖 AI Analysis Reports", "📊 Performance Dashboard"]
    }
    return module_map.get(user_role, [])

def main():
    # Initialize session state
    init_session_state()
    
    # Add CSS for better performance
    st.markdown("""
    <style>
        .stButton button {
            width: 100%;
            margin: 2px 0px;
        }
        .sidebar .stButton button {
            border-radius: 8px;
            padding: 10px;
            font-weight: 500;
        }
        /* Reduce re-renders by hiding elements when not needed */
        .element-container {
            transition: all 0.3s ease;
        }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        analyzer = InstitutionalAIAnalyzer()
    except Exception as e:
        st.error(f"❌ System initialization error: {str(e)}")
        st.stop()
    
    # Institution User Dashboard
    if st.session_state.institution_user is not None:
        # ... (institution dashboard code)
        if st.sidebar.button("🚪 Logout"):
            for key in ['institution_user', 'user_role', 'current_module']:
                st.session_state[key] = None
            st.rerun()
        return
    
    # System User Dashboard
    if st.session_state.system_user is not None:
        user_role = st.session_state.user_role
        
        # Simple sidebar navigation
        st.sidebar.title(f"🧭 {user_role} Navigation")
        st.sidebar.markdown("---")
        
        available_modules = get_available_modules(user_role)
        
        for module in available_modules:
            if st.sidebar.button(module, use_container_width=True):
                st.session_state.current_module = module
                st.rerun()
        
        if st.session_state.current_module:
            st.sidebar.success(f"**Current:** {st.session_state.current_module}")
        else:
            st.session_state.current_module = available_modules[0] if available_modules else None
            st.rerun()
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            for key in ['system_user', 'user_role', 'current_module']:
                st.session_state[key] = None
            st.rerun()
        
        # Render selected module (implementation depends on your specific module functions)
        return
    
    # Login Page
    st.markdown('<h1 class="main-header">🏛️ AI-Powered Institutional Approval Analytics System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">UGC & AICTE - Institutional Performance Tracking & Decision Support</h3>', unsafe_allow_html=True)
    
    auth_tabs = st.tabs(["🏛️ Institution Login", "🔐 System Login"])
    
    with auth_tabs[0]:
        create_institution_login(analyzer)
    
    with auth_tabs[1]:
        create_system_login(analyzer)
    
    # System stats with caching
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_system_stats(analyzer):
        current_year_data = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
        return {
            'total_institutions': len(analyzer.historical_data['institution_id'].unique()),
            'years_data': len(analyzer.historical_data['year'].unique()),
            'avg_performance': current_year_data['overall_score'].mean() if len(current_year_data) > 0 else 0,
            'approval_ready': (current_year_data['overall_score'] >= 6.0).sum() if len(current_year_data) > 0 else 0
        }
    
    stats = get_system_stats(analyzer)
    
    st.subheader("📈 System Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Institutions", stats['total_institutions'])
    with col2:
        st.metric("Years of Data", stats['years_data'])
    with col3:
        st.metric("Avg Performance Score", f"{stats['avg_performance']:.2f}/10" if stats['avg_performance'] else "N/A")
    with col4:
        st.metric("Approval Ready", stats['approval_ready'])

if __name__ == "__main__":
    main()
