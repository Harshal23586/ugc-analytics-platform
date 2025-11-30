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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import re
from typing import List

# Initialize session state at module level
if 'session_initialized' not in st.session_state:
    st.session_state.session_initialized = True
    st.session_state.institution_user = None
    st.session_state.user_role = None
    st.session_state.rag_analysis = None
    st.session_state.selected_institution = None

class RAGDocument:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
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
        texts, embeddings = zip(*text_embeddings)
        self.documents = list(texts)
        self.embeddings = np.array(embeddings)
        return self
    
    def similarity_search_with_score(self, query: str, k: int = 5):
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

# Page configuration
st.set_page_config(
    page_title="AI-Powered Institutional Approval System - UGC/AICTE",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text.strip()
    
    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        data = {
            'curriculum_metrics': {},
            'faculty_metrics': {},
            'learning_teaching_metrics': {},
            'research_innovation_metrics': {},
            'community_engagement_metrics': {},
            'green_initiatives_metrics': {},
            'governance_metrics': {},
            'infrastructure_metrics': {},
            'financial_metrics': {}
        }
    
        # Curriculum patterns
        curriculum_patterns = {
            'curriculum_innovation_score': r'curriculum.*innovation.*score[:\s]*(\d+(?:\.\d+)?)',
            'student_feedback_score': r'student.*feedback.*score[:\s]*(\d+(?:\.\d+)?)',
            'stakeholder_involvement_score': r'stakeholder.*involvement.*score[:\s]*(\d+(?:\.\d+)?)',
            'multidisciplinary_courses': r'multidisciplinary.*courses[:\s]*(\d+)'
        }
    
        # Faculty patterns
        faculty_patterns = {
            'faculty_selection_transparency': r'faculty.*selection.*transparency[:\s]*(\d+(?:\.\d+)?)',
            'faculty_diversity_index': r'faculty.*diversity.*index[:\s]*(\d+(?:\.\d+)?)',
            'continuous_professional_dev': r'continuous.*professional.*development[:\s]*(\d+(?:\.\d+)?)'
        }
    
        # Research patterns
        research_patterns = {
            'research_publications': r'research.*publications[:\s]*(\d+)',
            'patents_filed': r'patents.*filed[:\s]*(\d+)',
            'industry_collaboration_score': r'industry.*collaboration.*score[:\s]*(\d+(?:\.\d+)?)',
            'translational_research_score': r'translational.*research.*score[:\s]*(\d+(?:\.\d+)?)'
        }
    
        # Extract data for each category
        for key, pattern in curriculum_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['curriculum_metrics'][key] = match.group(1)
    
        for key, pattern in faculty_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['faculty_metrics'][key] = match.group(1)
    
        for key, pattern in research_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['research_innovation_metrics'][key] = match.group(1)
    
        self.extract_contextual_data(text, data)
    
        return data
    
    def extract_contextual_data(self, text: str, data: Dict):
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
        if not self.vector_store:
            return []
            
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

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
        
class InstitutionalAIAnalyzer:
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
        self.historical_data = self.load_or_generate_data()
        self.performance_metrics = self.define_performance_metrics()
        self.document_requirements = self.define_document_requirements()
        self.rag_extractor = RAGDataExtractor()
        self.create_dummy_institution_users()
        self.create_dummy_system_users()

        
    def init_database(self):
        self.conn = sqlite3.connect('institutions.db', check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()

    
    def create_dummy_system_users(self):
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
        
        cursor.execute('''
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
        ''')
        
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

    # Also update the create_dummy_institution_users method to use our new institution IDs
    def create_dummy_institution_users(self):
        dummy_users = [
            {
                'institution_id': 'HEI_01',
                'username': 'inst001_admin',
                'password': 'password123',
                'contact_person': 'Dr. Rajesh Kumar',
                'email': 'rajesh.kumar@iitvaranasi.edu.in',
                'phone': '+91-9876543210'
            },
            {
                'institution_id': 'HEI_07',
                'username': 'inst007_registrar',
                'password': 'testpass456',
                'contact_person': 'Ms. Priya Sharma',
                'email': 'priya.sharma@himalayanrural.edu.in',
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
        return hashlib.sha256(password.encode()).hexdigest()

    def analyze_documents_with_rag(self, institution_id: str, uploaded_files: List) -> Dict[str, Any]:
        try:
            extracted_data = self.rag_extractor.extract_comprehensive_data(uploaded_files)
        
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
        if not extracted_data:
            extracted_data = {}
    
        insights = {
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'risk_assessment': {},
            'compliance_status': {}
        }
    
        academic_data = extracted_data.get('academic_metrics', {})
        research_data = extracted_data.get('research_metrics', {})
        financial_data = extracted_data.get('financial_metrics', {})
        
        if academic_data.get('naac_grade') in ['A++', 'A+', 'A']:
            insights['strengths'].append(f"Strong NAAC accreditation: {academic_data['naac_grade']}")
        
        if research_data.get('research_publications', 0) > 50:
            insights['strengths'].append("Robust research publication output")
        
        if academic_data.get('student_faculty_ratio', 0) > 25:
            insights['weaknesses'].append("High student-faculty ratio needs improvement")
        
        if research_data.get('patents_filed', 0) < 5:
            insights['weaknesses'].append("Limited patent filings - need to strengthen IPR culture")
        
        if not academic_data.get('nirf_ranking'):
            insights['recommendations'].append("Consider participating in NIRF ranking for better visibility")
        
        if research_data.get('industry_collaborations', 0) < 3:
            insights['recommendations'].append("Increase industry collaborations for practical exposure")
        
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
        
        if academic_data.get('naac_grade') in ['A++', 'A+', 'A']:
            score -= 1.5
        if research_data.get('research_publications', 0) > 50:
            score -= 1.0
        if financial_data.get('financial_stability_score', 0) > 7:
            score -= 1.0
        
        if academic_data.get('student_faculty_ratio', 0) > 25:
            score += 1.5
        if research_data.get('patents_filed', 0) < 2:
            score += 1.0
        if not academic_data.get('nirf_ranking'):
            score += 0.5
        
        return max(1.0, min(10.0, score))
    
    def identify_risk_factors(self, extracted_data: Dict) -> List[str]:
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
    
        # Use the 20 institutions from our previous response
        institutions_list = [
            {
                'institution_id': 'HEI_01', 'institution_name': 'Indian Institute of Technology, Varanasi',
                'institution_type': 'Multi-disciplinary Education and Research-Intensive', 'heritage_category': 'Old and Established',
                'state': 'Uttar Pradesh', 'established_year': 1959
            },
                {
                'institution_id': 'HEI_02', 'institution_name': 'National Institute of Technology, Srinagar',
                'institution_type': 'Research-Intensive', 'heritage_category': 'Old and Established',
                'state': 'Jammu & Kashmir', 'established_year': 1960
            },
            {
                'institution_id': 'HEI_03', 'institution_name': 'State University of Bengaluru',
                'institution_type': 'Teaching-Intensive', 'heritage_category': 'Old and Established',
                'state': 'Karnataka', 'established_year': 1964
            },
            {
                'institution_id': 'HEI_04', 'institution_name': 'National Law School, Delhi',
                'institution_type': 'Specialised Streams', 'heritage_category': 'New and Upcoming',
                'state': 'Delhi', 'established_year': 2008
            },
            {
                'institution_id': 'HEI_05', 'institution_name': 'National Skill Development Institute, Pune',
                'institution_type': 'Vocational and Skill-Intensive', 'heritage_category': 'New and Upcoming',
                'state': 'Maharashtra', 'established_year': 2010
            },
            {
                'institution_id': 'HEI_06', 'institution_name': 'Rajiv Gandhi University of Community Health',
                'institution_type': 'Community Engagement & Service', 'heritage_category': 'New and Upcoming',
                'state': 'Telangana', 'established_year': 2012
            },
            {
                'institution_id': 'HEI_07', 'institution_name': 'Himalayan Institute of Rural Studies',
                'institution_type': 'Rural & Remote location', 'heritage_category': 'Old and Established',
                'state': 'Uttarakhand', 'established_year': 1975
            },
            {
                'institution_id': 'HEI_08', 'institution_name': 'Indian Institute of Management, Indore',
                'institution_type': 'Specialised Streams', 'heritage_category': 'Old and Established',
                'state': 'Madhya Pradesh', 'established_year': 1996
            },
            {
                'institution_id': 'HEI_09', 'institution_name': 'Amrita Vishwa Vidyapeetham, Coimbatore',
                'institution_type': 'Multi-disciplinary Education and Research-Intensive', 'heritage_category': 'Old and Established',
                'state': 'Tamil Nadu', 'established_year': 1994
            },
            {
                'institution_id': 'HEI_10', 'institution_name': 'KIIT University, Bhubaneswar',
                'institution_type': 'Teaching-Intensive', 'heritage_category': 'New and Upcoming',
                'state': 'Odisha', 'established_year': 2004
            },
            {
                'institution_id': 'HEI_11', 'institution_name': 'Tata Institute of Social Sciences, Mumbai',
                'institution_type': 'Specialised Streams', 'heritage_category': 'Old and Established',
                'state': 'Maharashtra', 'established_year': 1936
            },
            {
                'institution_id': 'HEI_12', 'institution_name': 'Lovely Professional University, Phagwara',
                'institution_type': 'Teaching-Intensive', 'heritage_category': 'New and Upcoming',
                'state': 'Punjab', 'established_year': 2005
            },
            {
                'institution_id': 'HEI_13', 'institution_name': 'Aligarh Muslim University, Aligarh',
                'institution_type': 'Multi-disciplinary Education and Research-Intensive', 'heritage_category': 'Old and Established',
                'state': 'Uttar Pradesh', 'established_year': 1920
            },
            {
                'institution_id': 'HEI_14', 'institution_name': 'South Indian Institute of Maritime Studies',
                'institution_type': 'Vocational and Skill-Intensive', 'heritage_category': 'Old and Established',
                'state': 'Kerala', 'established_year': 1975
            },
            {
                'institution_id': 'HEI_15', 'institution_name': 'Christ University, Bengaluru',
                'institution_type': 'Teaching-Intensive', 'heritage_category': 'Old and Established',
                'state': 'Karnataka', 'established_year': 1969
            },
            {
                'institution_id': 'HEI_16', 'institution_name': 'Jamia Millia Islamia, New Delhi',
                'institution_type': 'Research-Intensive', 'heritage_category': 'Old and Established',
                'state': 'Delhi', 'established_year': 1920
            },
            {
                'institution_id': 'HEI_17', 'institution_name': 'Symbiosis International University, Pune',
                'institution_type': 'Multi-disciplinary Education and Research-Intensive', 'heritage_category': 'New and Upcoming',
                'state': 'Maharashtra', 'established_year': 2002
            },
            {
                'institution_id': 'HEI_18', 'institution_name': 'National Institute of Fashion Technology, Delhi',
                'institution_type': 'Specialised Streams', 'heritage_category': 'Old and Established',
                'state': 'Delhi', 'established_year': 1986
            },
            {
                'institution_id': 'HEI_19', 'institution_name': 'Birla Institute of Technology, Mesra',
                'institution_type': 'Research-Intensive', 'heritage_category': 'Old and Established',
                'state': 'Jharkhand', 'established_year': 1955
            },
            {
                'institution_id': 'HEI_20', 'institution_name': 'Central Tribal University of Andhra Pradesh',
                'institution_type': 'Rural & Remote location', 'heritage_category': 'New and Upcoming',
                'state': 'Andhra Pradesh', 'established_year': 2019
            }
        ]

        institutions_data = []
        years_of_data = 10

        for inst_info in institutions_list:
            institution_type = inst_info['institution_type']
            heritage_type = inst_info['heritage_category']
        
            # Set weights based on institution type
            if institution_type == "Research-Intensive":
                research_weight = 1.5
                teaching_weight = 0.8
                community_weight = 0.7
            elif institution_type == "Teaching-Intensive":
                research_weight = 0.7
                teaching_weight = 1.3
                community_weight = 0.9
            elif institution_type == "Community Engagement & Service":
                research_weight = 0.6
                teaching_weight = 0.9
                community_weight = 1.4
            elif institution_type == "Rural & Remote location":
                research_weight = 0.5
                teaching_weight = 0.8
                community_weight = 1.3
            elif institution_type == "Vocational and Skill-Intensive":
                research_weight = 0.4
                teaching_weight = 1.2
                community_weight = 1.1
            else:
                research_weight = 1.0
                teaching_weight = 1.0
                community_weight = 1.0
            
            # Heritage bonus
            if heritage_type == "Old and Established":
                stability_bonus = 0.2
            else:
                stability_bonus = 0.0
        
            for year_offset in range(years_of_data):
                year = 2023 - year_offset
                improvement_factor = 1.0 + (year_offset * 0.02)

                # Base institution data
                institution_data = {
                    'institution_id': inst_info['institution_id'],
                    'institution_name': inst_info['institution_name'],
                    'year': year,
                    'institution_type': institution_type,
                    'heritage_category': heritage_type,
                    'established_year': inst_info['established_year'],
                    'state': inst_info['state'],
                
                    # Curriculum parameters (Appendix 1 - i)
                    'curriculum_innovation_score': round(np.random.uniform(5, 9) * improvement_factor, 2),
                    'student_feedback_score': round(np.random.uniform(6, 9) * improvement_factor, 2),
                    'stakeholder_involvement_score': round(np.random.uniform(5, 8) * improvement_factor, 2),
                    'lifelong_learning_initiatives': np.random.randint(1, 10),
                    'multidisciplinary_courses': np.random.randint(5, 20),
                
                    # Faculty Resources parameters (Appendix 1 - ii)
                    'faculty_selection_transparency': round(np.random.uniform(6, 9) * improvement_factor, 2),
                    'faculty_diversity_index': round(np.random.uniform(5, 8) * improvement_factor, 2),
                    'continuous_professional_dev': round(np.random.uniform(4, 9) * improvement_factor, 2),
                    'social_inclusivity_measures': round(np.random.uniform(5, 8) * improvement_factor, 2),
                
                    # Learning and Teaching parameters (Appendix 1 - iii)
                    'experiential_learning_score': round(np.random.uniform(5, 9) * teaching_weight * improvement_factor, 2),
                    'digital_technology_adoption': round(np.random.uniform(4, 9) * improvement_factor, 2),
                    'research_oriented_teaching': round(np.random.uniform(5, 8) * research_weight * improvement_factor, 2),
                    'critical_thinking_focus': round(np.random.uniform(5, 9) * improvement_factor, 2),
                
                    # Research and Innovation parameters (Appendix 1 - iv)
                    'interdisciplinary_research': round(np.random.uniform(4, 9) * research_weight * improvement_factor, 2),
                    'industry_collaboration_score': round(np.random.uniform(4, 8) * improvement_factor, 2),
                    'patents_filed': int(np.random.poisson(3 * research_weight)),
                    'research_publications': int(np.random.poisson(15 * research_weight)),
                    'translational_research_score': round(np.random.uniform(3, 8) * research_weight * improvement_factor, 2),
                
                    # Community Engagement parameters (Appendix 1 - vi)
                    'community_projects_count': int(np.random.poisson(8 * community_weight)),
                    'social_outreach_score': round(np.random.uniform(4, 9) * community_weight * improvement_factor, 2),
                    'rural_engagement_initiatives': np.random.randint(1, 12),
                
                    # Green Initiatives parameters (Appendix 1 - vii)
                    'renewable_energy_adoption': round(np.random.uniform(3, 9) * improvement_factor, 2),
                    'waste_management_score': round(np.random.uniform(4, 9) * improvement_factor, 2),
                    'carbon_footprint_reduction': round(np.random.uniform(3, 8) * improvement_factor, 2),
                    'sdg_alignment_score': round(np.random.uniform(4, 9) * improvement_factor, 2),
                
                    # Governance and Administration parameters (Appendix 1 - viii)
                    'egovernance_implementation': round(np.random.uniform(4, 9) * improvement_factor, 2),
                    'grievance_redressal_efficiency': round(np.random.uniform(5, 9) * improvement_factor, 2),
                    'internationalization_score': round(np.random.uniform(3, 8) * improvement_factor, 2),
                    'gender_parity_ratio': round(np.random.uniform(0.3, 0.8), 2),
                
                    # Infrastructure Development parameters (Appendix 1 - ix)
                    'digital_infrastructure_score': round(np.random.uniform(5, 9) * improvement_factor, 2),
                    'research_lab_quality': round(np.random.uniform(4, 9) * research_weight * improvement_factor, 2),
                    'library_resources_score': round(np.random.uniform(5, 9) * improvement_factor, 2),
                    'sports_facilities_score': round(np.random.uniform(4, 8) * improvement_factor, 2),
                
                    # Financial Resources and Management parameters (Appendix 1 - x)
                    'research_funding_utilization': round(np.random.uniform(4, 9) * research_weight * improvement_factor, 2),
                    'infrastructure_investment': round(np.random.uniform(3, 8) * improvement_factor, 2),
                    'financial_sustainability': round(np.random.uniform(5, 9) * improvement_factor, 2),
                    'csr_funding_attraction': round(np.random.uniform(2, 8) * improvement_factor, 2),
                }
            
                # Calculate scores based on Appendix 1 framework
                institution_data['input_score'] = self.calculate_input_score(institution_data)
                institution_data['process_score'] = self.calculate_process_score(institution_data)
                institution_data['outcome_score'] = self.calculate_outcome_score(institution_data)
                institution_data['impact_score'] = self.calculate_impact_score(institution_data)
                institution_data['overall_score'] = self.calculate_overall_score(institution_data)
            
                institution_data['approval_recommendation'] = self.generate_approval_recommendation(institution_data['overall_score'])
                institution_data['risk_level'] = self.assess_risk_level(institution_data['overall_score'])
            
                institutions_data.append(institution_data)
    
        return pd.DataFrame(institutions_data)

    def calculate_input_score(self, data):
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
        return round(score, 2)

    def calculate_process_score(self, data):
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
            min(data['community_projects_count']/25 * 10, 10) * weights['community_processes']
        )
        return round(score, 2)

    def calculate_outcome_score(self, data):
        weights = {
            'learning_outcomes': 0.4,
            'research_outcomes': 0.3,
            'skill_development': 0.3
        }
        
        score = (
            data['critical_thinking_focus'] * weights['learning_outcomes'] +
            min(data['research_publications']/50 * 10, 10) * weights['research_outcomes'] +
            min(data['lifelong_learning_initiatives']/10 * 10, 10) * weights['skill_development']
        )
        return round(score, 2)

    def calculate_impact_score(self, data):
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
        return round(score, 2)

    def calculate_overall_score(self, data):
        weights = {
            'input': 0.25,
            'process': 0.25,
            'outcome': 0.25,
            'impact': 0.25
        }
        
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

    def save_institution_submission(self, institution_id: str, submission_type: str, 
                                  submission_data: Dict):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO institution_submissions 
            (institution_id, submission_type, submission_data)
            VALUES (?, ?, ?)
        ''', (institution_id, submission_type, json.dumps(submission_data)))
        self.conn.commit()

    def get_institution_submissions(self, institution_id: str) -> pd.DataFrame:
        return pd.read_sql('''
            SELECT * FROM institution_submissions 
            WHERE institution_id = ? 
            ORDER BY submitted_date DESC
        ''', self.conn, params=(institution_id,))

    def save_uploaded_documents(self, institution_id: str, uploaded_files: List, document_types: List[str]):
        cursor = self.conn.cursor()
        for i, uploaded_file in enumerate(uploaded_files):
            cursor.execute('''
                INSERT INTO institution_documents (institution_id, document_name, document_type, status)
                VALUES (?, ?, ?, ?)
            ''', (institution_id, uploaded_file.name, document_types[i], 'Uploaded'))
        self.conn.commit()
    
    def get_institution_documents(self, institution_id: str) -> pd.DataFrame:
        return pd.read_sql('''
            SELECT * FROM institution_documents 
            WHERE institution_id = ? 
            ORDER BY upload_date DESC
        ''', self.conn, params=(institution_id,))
    
    def analyze_document_sufficiency(self, uploaded_docs: List[str], approval_type: str) -> Dict:
        requirements = self.document_requirements[approval_type]
        
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
        recommendations = []
        
        if mandatory_sufficiency < 100:
            recommendations.append("Upload all mandatory documents to proceed with approval process")
        
        if mandatory_sufficiency < 80:
            recommendations.append("Critical documents missing - application cannot be processed")
        
        if mandatory_sufficiency >= 100:
            recommendations.append("All mandatory documents present - ready for comprehensive evaluation")
        
        return recommendations
    
    def generate_comprehensive_report(self, institution_id: str) -> Dict[str, Any]:
        inst_data = self.historical_data[
            self.historical_data['institution_id'] == institution_id
        ]
        
        if inst_data.empty:
            return {"error": "Institution not found"}
        
        latest_data = inst_data[inst_data['year'] == inst_data['year'].max()].iloc[0]
        historical_trend = inst_data.groupby('year')['overall_score'].mean()
        
        if len(historical_trend) > 1:
            if historical_trend.iloc[-1] > historical_trend.iloc[-2]:
                trend_analysis = "Improving"
            elif historical_trend.iloc[-1] == historical_trend.iloc[-2]:
                trend_analysis = "Stable"
            else:
                trend_analysis = "Declining"
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
        inst_data = self.historical_data[
            self.historical_data['institution_id'] == institution_id
        ]
        
        if inst_data.empty:
            return {}
        
        latest_data = inst_data[inst_data['year'] == inst_data['year'].max()].iloc[0]
        
        similar_inst = self.historical_data[
            (self.historical_data['institution_type'] == latest_data['institution_type']) &
            (self.historical_data['year'] == latest_data['year']) &
            (self.historical_data['institution_id'] != institution_id)
        ]
        
        if len(similar_inst) > 0:
            similar_inst = similar_inst.nlargest(5, 'overall_score')
            benchmark_data = similar_inst[['institution_name', 'overall_score', 'approval_recommendation']].to_dict('records')
        else:
            benchmark_data = []
        
        return {
            "benchmark_institutions": benchmark_data,
            "performance_percentile": self.calculate_performance_percentile(latest_data['overall_score'], latest_data['institution_type'])
        }
    
    def calculate_performance_percentile(self, score: float, inst_type: str) -> float:
        type_data = self.historical_data[
            (self.historical_data['institution_type'] == inst_type) &
            (self.historical_data['year'] == 2023)
        ]
        
        if len(type_data) == 0:
            return 50.0
        
        return (type_data['overall_score'] < score).mean() * 100
    
    def identify_strengths(self, institution_data: pd.Series) -> List[str]:
        strengths = []
        
        if institution_data['overall_score'] >= 8.0:
            strengths.append(f"Excellent Overall Performance: {institution_data['overall_score']:.2f}/10")
        
        if institution_data['input_score'] >= 8.0:
            strengths.append(f"Strong Input Parameters: {institution_data['input_score']:.2f}/10")
        
        if institution_data['process_score'] >= 8.0:
            strengths.append(f"Excellent Process Implementation: {institution_data['process_score']:.2f}/10")
        
        if institution_data['research_publications'] > 20:
            strengths.append(f"Robust Research Output: {institution_data['research_publications']} publications")
        
        if institution_data['community_projects_count'] > 10:
            strengths.append(f"Strong Community Engagement: {institution_data['community_projects_count']} projects")
        
        if institution_data['digital_infrastructure_score'] > 8.0:
            strengths.append("Advanced Digital Infrastructure")
            
        return strengths
    
    def identify_weaknesses(self, institution_data: pd.Series) -> List[str]:
        weaknesses = []
        
        if institution_data['overall_score'] < 6.0:
            weaknesses.append(f"Low Overall Performance: {institution_data['overall_score']:.2f}/10")
        
        if institution_data['outcome_score'] < 6.0:
            weaknesses.append(f"Weak Outcome Parameters: {institution_data['outcome_score']:.2f}/10")
        
        if institution_data['research_publications'] < 5:
            weaknesses.append(f"Inadequate Research Output: {institution_data['research_publications']} publications")
        
        if institution_data['patents_filed'] < 2:
            weaknesses.append("Limited Innovation and Patent Filings")
        
        if institution_data['financial_sustainability'] < 6.0:
            weaknesses.append(f"Financial Sustainability Concerns: {institution_data['financial_sustainability']:.2f}/10")
            
        return weaknesses
    
    def generate_ai_recommendations(self, institution_data: pd.Series) -> List[str]:
        recommendations = []
        
        if institution_data['research_publications'] < 10:
            recommendations.append("Establish research promotion policy and faculty development programs")
        
        if institution_data['patents_filed'] < 3:
            recommendations.append("Strengthen innovation ecosystem and IPR culture")
        
        if institution_data['financial_sustainability'] < 7.0:
            recommendations.append("Diversify funding sources and improve financial planning")
        
        if institution_data['digital_infrastructure_score'] < 7.0:
            recommendations.append("Invest in digital infrastructure and e-learning platforms")
        
        if institution_data['community_projects_count'] < 5:
            recommendations.append("Enhance community engagement and social outreach programs")
        
        return recommendations

# Institution-specific modules
def create_institution_login(analyzer):
    st.header("🏛️ Institution Portal Login")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Existing Institution Users")
        username = st.text_input("Username", key="inst_login_username")
        password = st.text_input("Password", type="password", key="inst_login_password")
        
        if st.button("Login", key="inst_login_button"):
            user = analyzer.authenticate_institution_user(username, password)
            if user:
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
        
        if st.button("Register Institution Account", key="inst_reg_button"):
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
    st.header("🔐 System Login")
    
    role = st.selectbox(
        "Select Your Role",
        ["UGC Officer", "AICTE Officer", "System Admin", "Review Committee"],
        key="system_login_role"
    )

    username = st.text_input("Username", key="system_login_username")
    password = st.text_input("Password", type="password", key="system_login_password")
    
    if st.button("Login", key="system_login_button"):
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
        
    st.header(f"🏛️ Institution Dashboard - {user.get('institution_name', 'Unknown')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Institution ID", user.get('institution_id', 'N/A'))
    with col2:
        st.metric("Contact Person", user.get('contact_person', 'N/A'))
    with col3:
        st.metric("Email", user.get('email', 'N/A'))
    with col4:
        st.metric("Role", user.get('role', 'N/A'))
    
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
            analyzer.save_uploaded_documents(user['institution_id'], uploaded_files, document_types)
            st.success("✅ Documents uploaded successfully!")
            
            file_names = [file.name for file in uploaded_files]
            analysis_result = analyzer.analyze_document_sufficiency(file_names, approval_type)
            
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
            
            if analysis_result['missing_mandatory']:
                st.error("**❌ Missing Mandatory Documents:**")
                for doc in analysis_result['missing_mandatory']:
                    st.write(f"• {doc.replace('_', ' ').title()}")
            
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

# Performance Dashboard
def create_performance_dashboard(analyzer):
    st.header("📊 Institutional Performance Analytics Dashboard")
    
    df = analyzer.historical_data
    current_year_data = df[df['year'] == 2023]
    
    if len(current_year_data) == 0:
        st.warning("No data available for the current year.")
        return
    
    st.subheader("🏆 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_performance = current_year_data['overall_score'].mean()
        st.metric("Average Performance Score", f"{avg_performance:.2f}/10")
    
    with col2:
        approval_rate = (current_year_data['overall_score'] >= 6.0).mean()
        st.metric("Approval Eligibility Rate", f"{approval_rate:.1%}")
    
    with col3:
        high_risk_count = (current_year_data['risk_level'] == 'High Risk').sum() + (
            current_year_data['risk_level'] == 'Critical Risk').sum()
        st.metric("High/Critical Risk Institutions", high_risk_count)
    
    with col4:
        avg_research = current_year_data['research_publications'].mean()
        st.metric("Avg Research Publications", f"{avg_research:.1f}")
    
    with col5:
        avg_community = current_year_data['community_projects_count'].mean()
        st.metric("Avg Community Projects", f"{avg_community:.1f}")
    
    st.subheader("📈 Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not current_year_data['overall_score'].empty:
            fig1 = px.histogram(
                current_year_data, 
                x='overall_score',
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
            filtered_data = current_year_data.dropna(subset=['institution_type', 'overall_score'])
            if not filtered_data.empty:
                fig2 = px.box(
                    filtered_data,
                    x='institution_type',
                    y='overall_score',
                    title="Performance Score by Institution Type",
                    color='institution_type'
                )
                fig2.update_layout(
                    xaxis_title="Institution Type",
                    yaxis_title="Performance Score",
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("📅 Historical Performance Trends")
    
    trend_data = df.groupby(['year', 'institution_type'])['overall_score'].mean().reset_index()
    
    if len(trend_data) > 1 and not trend_data.empty:
        fig3 = px.line(
            trend_data,
            x='year',
            y='overall_score',
            color='institution_type',
            title="Average Performance Score Trend (2014-2023)",
            markers=True
        )
        fig3.update_layout(
            xaxis_title="Year", 
            yaxis_title="Average Performance Score",
            legend_title="Institution Type"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("⚠️ Institutional Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_distribution = current_year_data['risk_level'].value_counts()
        if not risk_distribution.empty:
            fig4 = px.pie(
                values=risk_distribution.values,
                names=risk_distribution.index,
                title="Institutional Risk Level Distribution",
                color=risk_distribution.index,
                color_discrete_map={
                    'Low Risk': '#2ecc71',
                    'Medium Risk': '#f39c12',
                    'High Risk': '#e74c3c',
                    'Critical Risk': '#c0392b'
                }
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        scatter_data = current_year_data.dropna(subset=['research_publications', 'community_projects_count', 'risk_level'])
        if not scatter_data.empty:
            fig5 = px.scatter(
                scatter_data,
                x='research_publications',
                y='community_projects_count',
                color='risk_level',
                size='overall_score',
                hover_data=['institution_name'],
                title="Research Output vs Community Engagement",
                color_discrete_map={
                    'Low Risk': '#2ecc71',
                    'Medium Risk': '#f39c12',
                    'High Risk': '#e74c3c',
                    'Critical Risk': '#c0392b'
                }
            )
            fig5.update_layout(
                xaxis_title="Research Publications",
                yaxis_title="Community Projects Count"
            )
            st.plotly_chart(fig5, use_container_width=True)

# Document Analysis Module
def create_document_analysis_module(analyzer):
    st.header("📋 AI-Powered Document Sufficiency Analysis")
    
    st.info("Analyze document completeness and generate sufficiency reports for approval processes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Document Upload & Analysis")
        
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
            st.subheader("📝 Document Type Assignment")
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
            
            if st.button("💾 Save Documents & Analyze"):
                analyzer.save_uploaded_documents(selected_institution, uploaded_files, document_types)
                st.success("✅ Documents saved successfully!")
                
                file_names = [file.name for file in uploaded_files]
                analysis_result = analyzer.analyze_document_sufficiency(file_names, approval_type)
                
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
                
                if analysis_result['missing_mandatory']:
                    st.error("**❌ Missing Mandatory Documents:**")
                    for doc in analysis_result['missing_mandatory']:
                        st.write(f"• {doc.replace('_', ' ').title()}")
                
                if analysis_result['missing_supporting']:
                    st.warning("**📝 Missing Supporting Documents:**")
                    for doc in analysis_result['missing_supporting']:
                        st.write(f"• {doc.replace('_', ' ').title()}")
                
                st.info("**💡 AI Recommendations:**")
                for recommendation in analysis_result['recommendations']:
                    st.write(f"• {recommendation}")
    
    with col2:
        st.subheader("Document Requirements Guide")
        
        requirements = analyzer.document_requirements
        
        for approval_type, docs in requirements.items():
            with st.expander(f"{approval_type.replace('_', ' ').title()} Requirements"):
                st.write("**Mandatory Documents:**")
                for doc in docs['mandatory']:
                    st.write(f"• {doc.replace('_', ' ').title()}")
                
                st.write("**Supporting Documents:**")
                for doc in docs['supporting']:
                    st.write(f"• {doc.replace('_', ' ').title()}")
        
        if selected_institution:
            st.subheader("📁 Previously Uploaded Documents")
            existing_docs = analyzer.get_institution_documents(selected_institution)
            if len(existing_docs) > 0:
                st.dataframe(existing_docs[['document_name', 'document_type', 'upload_date', 'status']])
            else:
                st.info("No documents uploaded yet for this institution.")

# AI Analysis Reports
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
            report = analyzer.generate_comprehensive_report(selected_institution)
            
            if "error" not in report:
                st.subheader(f"🏛️ AI Analysis Report: {report['institution_info']['name']}")
                
                st.info("**Institution Overview**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Type", report['institution_info']['type'])
                with col2:
                    st.metric("Heritage", report['institution_info']['heritage'])
                with col3:
                    st.metric("State", report['institution_info']['state'])
                with col4:
                    st.metric("Established", report['institution_info']['established'])
                
                st.subheader("📊 Performance Scores")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Overall", f"{report['performance_analysis']['current_score']:.2f}/10")
                with col2:
                    st.metric("Input", f"{report['performance_analysis']['input_score']:.2f}/10")
                with col3:
                    st.metric("Process", f"{report['performance_analysis']['process_score']:.2f}/10")
                with col4:
                    st.metric("Outcome", f"{report['performance_analysis']['outcome_score']:.2f}/10")
                with col5:
                    st.metric("Impact", f"{report['performance_analysis']['impact_score']:.2f}/10")
                
                recommendation = report['performance_analysis']['approval_recommendation']
                if "Full Approval" in recommendation:
                    st.success(f"**✅ {recommendation}**")
                elif "Provisional" in recommendation:
                    st.warning(f"**🟡 {recommendation}**")
                elif "Conditional" in recommendation or "Monitoring" in recommendation:
                    st.error(f"**🟠 {recommendation}**")
                else:
                    st.error(f"**🔴 {recommendation}**")
                
                risk_level = report['performance_analysis']['risk_level']
                if risk_level == "Low Risk":
                    st.success(f"**Risk Level: {risk_level}**")
                elif risk_level == "Medium Risk":
                    st.warning(f"**Risk Level: {risk_level}**")
                else:
                    st.error(f"**Risk Level: {risk_level}**")
                
                st.metric(
                    "Performance Trend", 
                    report['performance_analysis']['trend_analysis'],
                    delta=report['performance_analysis']['trend_analysis'],
                    delta_color="normal" if report['performance_analysis']['trend_analysis'] == "Improving" else "off" if report['performance_analysis']['trend_analysis'] == "Stable" else "inverse"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if report['strengths']:
                        st.success("**✅ Institutional Strengths**")
                        for strength in report['strengths']:
                            st.write(f"• {strength}")
                    else:
                        st.info("No significant strengths identified")
                
                with col2:
                    if report['weaknesses']:
                        st.error("**⚠️ Areas for Improvement**")
                        for weakness in report['weaknesses']:
                            st.write(f"• {weakness}")
                    else:
                        st.success("No major weaknesses identified")
                
                if report['ai_recommendations']:
                    st.warning("**🎯 AI Improvement Recommendations**")
                    for recommendation in report['ai_recommendations']:
                        st.write(f"• {recommendation}")
                else:
                    st.success("Institution is performing well across all parameters")
                
                st.info("**📊 Comparative Analysis**")
                if report['comparative_analysis']:
                    st.write(f"**Performance Percentile:** {report['comparative_analysis']['performance_percentile']:.1f}%")
                    if report['comparative_analysis']['benchmark_institutions']:
                        st.write("**Benchmark Institutions:**")
                        for bench in report['comparative_analysis']['benchmark_institutions']:
                            st.write(f"• **{bench['institution_name']}**: {bench['overall_score']:.2f} - {bench['approval_recommendation']}")
                    else:
                        st.info("No similar institutions found for comparison")
                
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
        
        top_performers = df[df['year'] == 2023].nlargest(5, 'overall_score')[
            ['institution_name', 'overall_score', 'approval_recommendation']
        ]
        
        st.write("**🏆 Top Performing Institutions**")
        for _, inst in top_performers.iterrows():
            st.write(f"• **{inst['institution_name']}** ({inst['overall_score']:.2f})")
            st.write(f"  _{inst['approval_recommendation']}_")
        
        st.markdown("---")
        
        high_risk = df[
            (df['year'] == 2023) & 
            (df['risk_level'].isin(['High Risk', 'Critical Risk']))
        ].head(5)
        
        if not high_risk.empty:
            st.write("**🚨 High Risk Institutions**")
            for _, inst in high_risk.iterrows():
                st.write(f"• **{inst['institution_name']}** - {inst['risk_level']}")
        
        st.markdown("---")
        st.write("**📊 Quick Statistics**")
        total_inst = len(df[df['year'] == 2023])
        approved = len(df[(df['year'] == 2023) & (df['overall_score'] >= 7.0)])
        st.write(f"• Total Institutions: {total_inst}")
        st.write(f"• High Performing: {approved}")
        st.write(f"• Approval Rate: {(approved/total_inst*100):.1f}%")

# Data Management Module
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
                
                st.subheader("Data Preview")
                st.dataframe(new_data.head())
                
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
                            analyzer.historical_data = analyzer.load_or_generate_data()
                        except Exception as e:
                            st.error(f"❌ Error saving to database: {str(e)}")
                            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with tab2:
        st.subheader("Current Database Contents")
        
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
        
        st.subheader("Data Preview")
        st.dataframe(analyzer.historical_data.head(10))
        
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

# Approval Workflow
def create_approval_workflow(analyzer):
    st.header("🔄 AI-Enhanced Approval Workflow")
    
    st.info("Streamlined approval process with AI-powered decision support")
    
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
            "description": "AI analyzes 10 years of institutional performance data",
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

# RAG Data Management
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
            
            with st.expander("📋 Document Preview"):
                for i, file in enumerate(uploaded_files):
                    st.write(f"**{i+1}. {file.name}** ({file.size} bytes)")
            
            if st.button("🚀 Start RAG Analysis", type="primary"):
                with st.spinner("🤖 AI is analyzing documents and extracting data..."):
                    analysis_result = analyzer.analyze_documents_with_rag(
                        selected_institution, 
                        uploaded_files
                    )
                    
                    if analysis_result and analysis_result.get('status') == 'Analysis Complete':
                        st.success("✅ RAG Analysis Completed Successfully!")
                        
                        st.session_state.rag_analysis = analysis_result
                        st.session_state.selected_institution = selected_institution
                        
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
                        if analysis_result:
                            st.session_state.rag_analysis = analysis_result
    
    with tab2:
        st.subheader("Extracted Data View")
        
        if 'rag_analysis' in st.session_state and st.session_state.rag_analysis:
            analysis_result = st.session_state.rag_analysis
            extracted_data = analysis_result.get('extracted_data', {})
            
            if not extracted_data:
                st.warning("No extracted data available. Please run RAG analysis first.")
                return
            
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
        st.info("Configure RAG system parameters and settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.slider("Text Chunk Size", 500, 2000, 1000, help="Size of text chunks for processing")
            chunk_overlap = st.slider("Chunk Overlap", 100, 500, 200, help="Overlap between text chunks")
        
        with col2:
            similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.7, help="Minimum similarity score for matches")
            max_results = st.slider("Max Results", 1, 20, 5, help="Maximum number of results to return")
        
        if st.button("Apply Settings"):
            st.success("RAG settings updated successfully!")

# System Settings
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
    if user_role == "Institution":
        return ["🏛️ Institution Portal"]
    elif user_role == "System Admin":
        return ["📊 Performance Dashboard", "⚙️ System Settings"]
    elif user_role in ["UGC Officer", "AICTE Officer"]:
        return ["🔄 Approval Workflow", "💾 Data Management", "🔍 RAG Data Management", "📋 Document Analysis"]
    elif user_role == "Review Committee":
        return ["🤖 AI Reports"]
    else:
        return []

def main():     
    if 'institution_user' not in st.session_state:
        st.session_state.institution_user = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'system_user' not in st.session_state:
        st.session_state.system_user = None
    
    try:
        analyzer = InstitutionalAIAnalyzer()
    except Exception as e:
        st.error(f"❌ System initialization error: {str(e)}")
        st.stop()
    
    if st.session_state.institution_user is not None:
        create_institution_dashboard(analyzer, st.session_state.institution_user)
        if st.sidebar.button("🚪 Logout"):
            st.session_state.institution_user = None
            st.session_state.user_role = None
            st.rerun()
        return
    
    if st.session_state.system_user is not None:
        user_role = st.session_state.user_role
        available_modules = get_available_modules(user_role)
        
        st.sidebar.title(f"🧭 {user_role} Navigation")
        st.sidebar.markdown("---")
        
        if available_modules:
            selected_module = st.sidebar.selectbox("Select Module", available_modules)
            
            if selected_module == "📊 Performance Dashboard" and user_role == "System Admin":
                create_performance_dashboard(analyzer)
            elif selected_module == "⚙️ System Settings" and user_role == "System Admin":
                create_system_settings(analyzer)
            elif selected_module == "🔄 Approval Workflow" and user_role in ["UGC Officer", "AICTE Officer"]:
                create_approval_workflow(analyzer)
            elif selected_module == "💾 Data Management" and user_role in ["UGC Officer", "AICTE Officer"]:
                create_data_management_module(analyzer)
            elif selected_module == "🔍 RAG Data Management" and user_role in ["UGC Officer", "AICTE Officer"]:
                create_rag_data_management(analyzer)
            elif selected_module == "📋 Document Analysis" and user_role in ["UGC Officer", "AICTE Officer"]:
                create_document_analysis_module(analyzer)
            elif selected_module == "🤖 AI Reports" and user_role == "Review Committee":
                create_ai_analysis_reports(analyzer)
        
        if st.sidebar.button("🚪 Logout"):
            st.session_state.system_user = None
            st.session_state.user_role = None
            st.rerun()
        return
    
    st.markdown('<h1 class="main-header">🏛️ AI-Powered Institutional Approval Analytics System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">UGC & AICTE - Institutional Performance Tracking & Decision Support</h3>', unsafe_allow_html=True)
    
    auth_tabs = st.tabs(["🏛️ Institution Login", "🔐 System Login"])
    
    with auth_tabs[0]:
        create_institution_login(analyzer)
    
    with auth_tabs[1]:
        create_system_login(analyzer)
    
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
    
    st.subheader("📈 System Quick Stats")
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
            avg_performance = current_year_data['overall_score'].mean()
            st.metric("Avg Performance Score", f"{avg_performance:.2f}/10")
        else:
            st.metric("Avg Performance Score", "N/A")
    
    with col4:
        if len(current_year_data) > 0:
            approval_ready = (current_year_data['overall_score'] >= 6.0).sum()
            st.metric("Approval Ready", approval_ready)
        else:
            st.metric("Approval Ready", "N/A")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d;'>
    <p><strong>UGC/AICTE Institutional Analytics Platform</strong> | AI-Powered Decision Support System</p>
    <p>Version 2.0 | For authorized use only | Data last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
