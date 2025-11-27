import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add missing imports
from typing import List, Dict, Tuple

# Fix PyTorch conflict with Streamlit
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# AI/ML imports with error handling
try:
    import torch
    # Fix for torch.classes path issue
    if hasattr(torch, 'classes') and hasattr(torch.classes, '__path__'):
        torch.classes.__path__ = []
except ImportError:
    st.warning("PyTorch not available - some advanced features disabled")

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    st.warning("RAG components not available - using simplified version")

# Page configuration
st.set_page_config(
    page_title="AI-Powered UGC/AICTE Analytics Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedUGC_AICTE_Analytics:
    def __init__(self):
        self.sample_data = self.generate_comprehensive_data()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.rag_initialized = False
        self.setup_advanced_rag_system()
        
    def setup_advanced_rag_system(self):
        """Initialize RAG components with robust error handling"""
        try:
            if RAG_AVAILABLE:
                # Initialize sentence transformer for embeddings
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Initialize ChromaDB with new client configuration
                self.chroma_client = chromadb.PersistentClient(path="./.chroma_db")
                
                # Create or get collection
                self.collection = self.chroma_client.get_or_create_collection(
                    name="ugc_aicte_guidelines",
                    metadata={"description": "UGC/AICTE guidelines and regulations"}
                )
                
                # Preload with comprehensive guidelines
                self.initialize_comprehensive_knowledge_base()
                self.rag_initialized = True
            else:
                self.initialize_fallback_knowledge_base()
                
        except Exception as e:
            st.warning(f"Advanced RAG system initialization: {e}")
            self.initialize_fallback_knowledge_base()
    
    def initialize_fallback_knowledge_base(self):
        """Fallback knowledge base without vector DB"""
        self.guidelines_db = {
            "approval_criteria": [
                "Minimum 1:20 faculty-student ratio for undergraduate programs",
                "Infrastructure: Digital classrooms, library with 5000+ books, adequate laboratory facilities",
                "Placement rate must exceed 60% for technical institutions",
                "Research publications and industry collaborations enhance approval chances significantly"
            ],
            "technical_standards": [
                "Technical institutions must have industry collaborations with MoUs",
                "Curriculum must be updated annually to match industry requirements",
                "Laboratory equipment should be less than 5 years old",
                "Faculty must have minimum 3 years of industry experience"
            ],
            "documentation": [
                "Required documents: Affiliation certificates, Land documents, Building approval plans",
                "Faculty qualifications: PhDs minimum 30%, Masters 70%",
                "Financial statements for last 3 years audited by CA",
                "Infrastructure details with photographs and floor plans"
            ],
            "compliance": [
                "Must meet all statutory requirements including AICTE/UGC norms",
                "Financial stability with positive net worth for last 3 years",
                "Student feedback mechanism with 70%+ satisfaction rate",
                "Governance structure with proper academic and administrative bodies"
            ]
        }
        self.rag_initialized = False
    
    def initialize_comprehensive_knowledge_base(self):
        """Initialize with comprehensive UGC/AICTE guidelines"""
        guidelines = [
            {
                "document": "UGC Approval Guidelines 2024",
                "content": "Institutions must maintain minimum 1:20 faculty-student ratio for approval. Infrastructure should include digital classrooms, library with 5000+ books, and adequate laboratory facilities. Research output minimum 2 publications per faculty annually.",
                "category": "approval_criteria"
            },
            {
                "document": "AICTE Technical Standards 2024",
                "content": "Technical institutions must have industry collaborations, minimum 70% placement rate, and updated curriculum matching industry requirements. Research publications are mandatory for autonomous status. Laboratories must have modern equipment.",
                "category": "technical_standards"
            },
            {
                "document": "NAAC Accreditation Framework",
                "content": "Grading criteria: Curricular Aspects (150), Teaching-Learning (200), Research (150), Infrastructure (100), Student Support (100), Governance (100), Innovations (100). Minimum score 2.5 for accreditation, 3.0 for autonomous status.",
                "category": "accreditation"
            },
            {
                "document": "Document Requirements Checklist",
                "content": "Required documents: Affiliation certificates, Land documents, Building approval plans, Faculty qualifications, Financial statements, Infrastructure details, Academic calendar, Research publications, Industry MoUs, Student placement records.",
                "category": "documentation"
            }
        ]
        
        # Check if collection is empty before adding documents
        try:
            existing_count = self.collection.count()
            if existing_count == 0:
                # Add to vector database
                for i, guideline in enumerate(guidelines):
                    self.collection.add(
                        documents=[guideline["content"]],
                        metadatas=[{"category": guideline["category"], "document": guideline["document"]}],
                        ids=[f"guideline_{i}"]
                    )
        except Exception as e:
            st.warning(f"Knowledge base initialization: {e}")
    
    def query_rag_system(self, query: str, n_results: int = 5) -> Dict:
        """Query the RAG system for relevant guidelines"""
        try:
            if not self.rag_initialized:
                # Fallback to simple search
                results = {"documents": [], "metadatas": []}
                for category, guidelines in self.guidelines_db.items():
                    for guideline in guidelines:
                        if query.lower() in guideline.lower():
                            results["documents"].append([guideline])
                            results["metadatas"].append([{"category": category, "document": "Fallback Guideline"}])
                return results
            
            # Search similar documents using new ChromaDB API
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            return results
        except Exception as e:
            st.error(f"RAG query error: {e}")
            return {"documents": [[]], "metadatas": [[]]}
    
    def generate_ai_recommendations(self, institution_data: Dict) -> Dict:
        """Generate comprehensive AI-powered recommendations"""
        recommendations = {
            "immediate_actions": [],
            "strategic_improvements": [],
            "compliance_suggestions": [],
            "rag_insights": [],
            "risk_factors": [],
            "success_indicators": []
        }
        
        # Analyze institutional data
        placement_rate = institution_data.get('placement_rate', 0)
        infrastructure_score = institution_data.get('infrastructure_score', 0)
        research_publications = institution_data.get('research_publications', 0)
        compliance_score = institution_data.get('compliance_score', 0)
        document_sufficiency = institution_data.get('document_sufficiency', 0)
        naac_grade = institution_data.get('naac_grade', 'B+')
        
        # Generate RAG-powered insights
        rag_query = f"""
        Institution with NAAC {naac_grade}, placement rate {placement_rate}%,
        infrastructure score {infrastructure_score}, research publications {research_publications},
        compliance score {compliance_score}, document sufficiency {document_sufficiency}%
        """
        
        rag_results = self.query_rag_system(rag_query)
        
        # Add RAG insights
        if rag_results and 'documents' in rag_results and rag_results['documents']:
            for doc_list in rag_results['documents']:
                for doc in doc_list:
                    recommendations["rag_insights"].append(doc)
        
        # Generate specific recommendations
        if placement_rate < 75:
            recommendations["immediate_actions"].append(
                "Establish dedicated placement cell with industry partnerships"
            )
            recommendations["risk_factors"].append(f"Low placement rate: {placement_rate}% (Target: 75%+)")
        else:
            recommendations["success_indicators"].append(f"Strong placement rate: {placement_rate}%")
        
        if infrastructure_score < 7:
            recommendations["immediate_actions"].append(
                "Upgrade laboratory equipment and digital infrastructure facilities"
            )
            recommendations["risk_factors"].append(f"Infrastructure score low: {infrastructure_score}/10")
        else:
            recommendations["success_indicators"].append(f"Good infrastructure: {infrastructure_score}/10")
        
        if research_publications < 50:
            recommendations["strategic_improvements"].append(
                "Establish research centers and provide faculty research grants. Target 2 publications per faculty annually"
            )
            recommendations["risk_factors"].append(f"Low research output: {research_publications} publications")
        else:
            recommendations["success_indicators"].append(f"Strong research: {research_publications} publications")
        
        if compliance_score < 8:
            recommendations["compliance_suggestions"].append(
                "Conduct comprehensive compliance audit and address all statutory requirements"
            )
            recommendations["risk_factors"].append(f"Compliance issues: {compliance_score}/10")
        else:
            recommendations["success_indicators"].append(f"Good compliance: {compliance_score}/10")
        
        if document_sufficiency < 85:
            recommendations["immediate_actions"].append(
                f"Complete pending documentation. Current: {document_sufficiency}% (Target: 85%+)"
            )
        
        # NAAC grade based recommendations
        if naac_grade in ['A++', 'A+']:
            recommendations["success_indicators"].append(f"Excellent NAAC accreditation: {naac_grade}")
        elif naac_grade in ['B+', 'B']:
            recommendations["strategic_improvements"].append(f"Focus on improving NAAC grade from {naac_grade} to A+")
        
        return recommendations
    
    def generate_comprehensive_data(self):
        """Generate comprehensive institutional data"""
        np.random.seed(42)
        n_institutions = 150  # Reduced for better performance
        
        data = {
            'institution_id': range(1, n_institutions + 1),
            'institution_name': [f'Institute_{i:03d}' for i in range(1, n_institutions + 1)],
            'established_year': np.random.randint(1950, 2020, n_institutions),
            'institution_type': np.random.choice(['University', 'College', 'Technical Institute'], n_institutions),
            'ownership': np.random.choice(['Government', 'Private', 'Deemed'], n_institutions),
            'state': np.random.choice(['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi'], n_institutions),
            'naac_grade': np.random.choice(['A++', 'A+', 'A', 'B++', 'B+'], n_institutions, p=[0.05, 0.1, 0.15, 0.3, 0.4]),
            'nirf_ranking': np.random.choice(range(1, 201), n_institutions),
            'total_faculty': np.random.randint(50, 500, n_institutions),
            'student_strength': np.random.randint(1000, 15000, n_institutions),
            'research_publications': np.random.randint(0, 500, n_institutions),
            'placement_rate': np.random.uniform(60, 95, n_institutions),
            'infrastructure_score': np.random.uniform(5, 10, n_institutions),
            'compliance_score': np.random.uniform(6, 10, n_institutions),
            'documents_submitted': np.random.randint(70, 100, n_institutions),
            'required_documents': 100,
        }
        
        df = pd.DataFrame(data)
        
        # Calculate derived metrics
        df['document_sufficiency'] = (df['documents_submitted'] / df['required_documents']) * 100
        df['faculty_student_ratio'] = df['total_faculty'] / df['student_strength']
        
        # Generate approval status with sophisticated logic
        approval_score = (
            df['naac_grade'].map({'A++': 10, 'A+': 9, 'A': 8, 'B++': 7, 'B+': 6}) * 0.3 +
            (1 - (df['nirf_ranking'] / 200)) * 0.2 +
            (df['placement_rate'] / 100) * 0.25 +
            (df['infrastructure_score'] / 10) * 0.15 +
            (df['compliance_score'] / 10) * 0.1
        )
        
        df['approval_probability'] = approval_score
        df['approval_status'] = np.where(
            approval_score > 0.7, 'Approved',
            np.where(approval_score > 0.5, 'Pending', 'Rejected')
        )
        
        # Add risk level
        df['risk_level'] = np.where(
            approval_score > 0.7, 'Low',
            np.where(approval_score > 0.5, 'Medium', 'High')
        )
        
        return df
    
    def train_ai_models(self, df):
        """Train multiple AI models for different tasks"""
        # Approval prediction model
        feature_columns = [
            'naac_grade', 'nirf_ranking', 'placement_rate', 'infrastructure_score',
            'compliance_score', 'document_sufficiency', 'research_publications'
        ]
        
        X = pd.get_dummies(df[feature_columns], columns=['naac_grade'])
        y_approval = df['approval_status']
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_approval)
        
        return self.model, self.scaler, feature_columns
    
    def predict_with_explanation(self, institution_data):
        """Predict with detailed explanation of factors"""
        # Create a simple prediction without complex ML for stability
        placement_rate = institution_data.get('placement_rate', 0)
        infrastructure_score = institution_data.get('infrastructure_score', 0)
        compliance_score = institution_data.get('compliance_score', 0)
        document_sufficiency = institution_data.get('document_sufficiency', 0)
        naac_grade = institution_data.get('naac_grade', 'B+')
        
        # Simple scoring logic
        score = (
            {'A++': 1.0, 'A+': 0.9, 'A': 0.8, 'B++': 0.7, 'B+': 0.6}.get(naac_grade, 0.5) * 0.3 +
            (placement_rate / 100) * 0.25 +
            (infrastructure_score / 10) * 0.2 +
            (compliance_score / 10) * 0.15 +
            (document_sufficiency / 100) * 0.1
        )
        
        prediction = "Approved" if score > 0.7 else "Pending" if score > 0.5 else "Rejected"
        probability = min(score, 0.95)  # Cap at 95%
        risk_level = "Low" if score > 0.7 else "Medium" if score > 0.5 else "High"
        
        explanation = {
            'prediction': prediction,
            'probability': probability,
            'risk_level': risk_level,
            'key_factors': [
                (f"NAAC Grade: {naac_grade}", 0.3),
                (f"Placement Rate: {placement_rate}%", 0.25),
                (f"Infrastructure: {infrastructure_score}/10", 0.2)
            ],
            'strengths': [],
            'weaknesses': []
        }
        
        # Analyze strengths and weaknesses
        if naac_grade in ['A++', 'A+', 'A']:
            explanation['strengths'].append(f"Strong NAAC accreditation: {naac_grade}")
        if placement_rate > 80:
            explanation['strengths'].append(f"Excellent placement record: {placement_rate:.1f}%")
        if document_sufficiency > 90:
            explanation['strengths'].append(f"Complete documentation: {document_sufficiency:.1f}%")
            
        if infrastructure_score < 6:
            explanation['weaknesses'].append(f"Inadequate infrastructure: {infrastructure_score}/10")
        if compliance_score < 7:
            explanation['weaknesses'].append(f"Compliance issues: {compliance_score}/10")
        if placement_rate < 70:
            explanation['weaknesses'].append(f"Placement concerns: {placement_rate:.1f}%")
        
        return explanation

def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🚀 AI-Powered UGC/AICTE Institutional Analytics</h1>', unsafe_allow_html=True)
    
    # Initialize advanced analytics engine
    try:
        analytics = AdvancedUGC_AICTE_Analytics()
        df = analytics.sample_data
        
        # Train AI models
        model, scaler, features = analytics.train_ai_models(df)
        
        st.success("✅ AI Analytics Platform Successfully Initialized!")
    except Exception as e:
        st.error(f"Error initializing platform: {e}")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🤖 AI Navigation Panel")
    app_mode = st.sidebar.selectbox("Choose AI Module", 
        ["🏠 Smart Dashboard", "💡 AI Recommendation Engine", "🔍 RAG Query System", 
         "🔮 Predictive Analytics"])
    
    if app_mode == "🏠 Smart Dashboard":
        st.header("🎯 Smart Institutional Intelligence Dashboard")
        
        # AI-powered metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            approval_rate = (df['approval_status'] == 'Approved').mean() * 100
            st.metric("AI Approval Rate", f"{approval_rate:.1f}%")
        
        with col2:
            high_risk = len(df[df['risk_level'] == 'High'])
            st.metric("High Risk Institutions", high_risk)
        
        with col3:
            avg_score = df['approval_probability'].mean() * 100
            st.metric("Avg Approval Score", f"{avg_score:.1f}%")
        
        with col4:
            total_institutions = len(df)
            st.metric("Total Institutions", total_institutions)
        
        # AI Insights Section
        st.subheader("🔍 AI-Generated Institutional Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top performing institutions
            top_performers = df.nlargest(5, 'approval_probability')[['institution_name', 'approval_probability', 'naac_grade', 'risk_level']]
            st.write("**🏆 Top Performing Institutions**")
            for _, inst in top_performers.iterrows():
                risk_color = "risk-low" if inst['risk_level'] == 'Low' else "risk-medium" if inst['risk_level'] == 'Medium' else "risk-high"
                st.write(f"• **{inst['institution_name']}** - Score: {inst['approval_probability']:.1%} - "
                        f"NAAC: {inst['naac_grade']} - "
                        f"<span class='{risk_color}'>Risk: {inst['risk_level']}</span>", unsafe_allow_html=True)
        
        with col2:
            # Institutions needing improvement
            need_improvement = df.nsmallest(5, 'approval_probability')[['institution_name', 'approval_probability', 'approval_status', 'risk_level']]
            st.write("**⚠️ Institutions Needing Attention**")
            for _, inst in need_improvement.iterrows():
                risk_color = "risk-low" if inst['risk_level'] == 'Low' else "risk-medium" if inst['risk_level'] == 'Medium' else "risk-high"
                st.write(f"• **{inst['institution_name']}** - Score: {inst['approval_probability']:.1%} - "
                        f"Status: {inst['approval_status']} - "
                        f"<span class='{risk_color}'>Risk: {inst['risk_level']}</span>", unsafe_allow_html=True)
        
        # AI Trend Analysis
        st.subheader("📈 AI Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance by establishment year
            df['establishment_decade'] = (df['established_year'] // 10) * 10
            decade_performance = df.groupby('establishment_decade')['approval_probability'].mean().reset_index()
            fig = px.line(decade_performance, x='establishment_decade', y='approval_probability',
                         title="Performance by Establishment Decade",
                         markers=True)
            st.plotly_chart(fig)
        
        with col2:
            # Risk distribution
            risk_dist = df['risk_level'].value_counts()
            fig = px.pie(values=risk_dist.values, names=risk_dist.index,
                        title="Risk Assessment Distribution")
            st.plotly_chart(fig)
    
    elif app_mode == "💡 AI Recommendation Engine":
        st.header("💡 AI-Powered Recommendation Engine")
        
        st.info("This AI system analyzes institutional data and provides personalized improvement recommendations.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Institution selector for recommendations
            selected_institution = st.selectbox("Select Institution for AI Analysis", 
                                              df['institution_name'].tolist())
            
            if selected_institution:
                institution_data = df[df['institution_name'] == selected_institution].iloc[0].to_dict()
                
                # Get AI recommendations
                recommendations = analytics.generate_ai_recommendations(institution_data)
                
                # Display recommendations
                st.subheader(f"🎯 AI Recommendations for {selected_institution}")
                
                if recommendations["success_indicators"]:
                    st.success("**✅ Strengths:**")
                    for strength in recommendations["success_indicators"]:
                        st.write(f"• {strength}")
                
                if recommendations["risk_factors"]:
                    st.error("**🚨 Risk Factors:**")
                    for risk in recommendations["risk_factors"]:
                        st.write(f"• {risk}")
                
                if recommendations["immediate_actions"]:
                    st.warning("**🚀 Immediate Actions:**")
                    for action in recommendations["immediate_actions"]:
                        st.write(f"• {action}")
                
                if recommendations["strategic_improvements"]:
                    st.info("**🎯 Strategic Improvements:**")
                    for improvement in recommendations["strategic_improvements"]:
                        st.write(f"• {improvement}")
        
        with col2:
            # Quick AI assessment
            st.subheader("⚡ Quick Assessment")
            if selected_institution:
                explanation = analytics.predict_with_explanation(institution_data)
                
                # Display metrics - FIXED: No unsafe_allow_html in st.metric
                st.metric("AI Approval Probability", f"{explanation['probability']:.1%}")
                st.metric("Predicted Status", explanation['prediction'])
                
                # Display risk level with color using markdown
                risk_level = explanation['risk_level']
                risk_color = "🟢" if risk_level == 'Low' else "🟡" if risk_level == 'Medium' else "🔴"
                st.metric("Risk Level", f"{risk_color} {risk_level}")
                
                st.write("**🎯 Key Factors:**")
                for factor, importance in explanation['key_factors'][:3]:
                    st.write(f"• {factor}")
    
    elif app_mode == "🔍 RAG Query System":
        st.header("🔍 Guidelines Query System")
        
        st.success("Query UGC/AICTE guidelines and regulations.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # RAG query interface
            query = st.text_input("Search guidelines:", 
                                placeholder="e.g., faculty qualification requirements")
            
            if st.button("🔍 Search Guidelines") and query:
                results = analytics.query_rag_system(query)
                
                if results and 'documents' in results and results['documents']:
                    st.subheader("📚 Relevant Guidelines:")
                    
                    for i, doc_list in enumerate(results['documents']):
                        for j, doc in enumerate(doc_list):
                            with st.expander(f"Guideline {i+1}"):
                                st.write(doc)
                else:
                    st.warning("No relevant guidelines found.")
        
        with col2:
            st.subheader("💡 Sample Queries")
            sample_queries = [
                "Faculty student ratio",
                "Infrastructure standards", 
                "NAAC accreditation",
                "Document requirements"
            ]
            
            for sample in sample_queries:
                if st.button(sample):
                    st.session_state.last_query = sample
    
    elif app_mode == "🔮 Predictive Analytics":
        st.header("🔮 Predictive Analytics")
        
        st.info("Predict approval probabilities for institutions.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Approval Prediction")
            
            with st.form("prediction_form"):
                naac_grade = st.selectbox("NAAC Grade", ['A++', 'A+', 'A', 'B++', 'B+'])
                placement_rate = st.slider("Placement Rate (%)", 60.0, 100.0, 80.0)
                infrastructure_score = st.slider("Infrastructure Score", 5.0, 10.0, 7.0)
                compliance_score = st.slider("Compliance Score", 6.0, 10.0, 8.0)
                document_sufficiency = st.slider("Document Sufficiency (%)", 70.0, 100.0, 85.0)
                
                submitted = st.form_submit_button("🚀 Predict Approval")
                
                if submitted:
                    institution_data = {
                        'naac_grade': naac_grade,
                        'placement_rate': placement_rate,
                        'infrastructure_score': infrastructure_score,
                        'compliance_score': compliance_score,
                        'document_sufficiency': document_sufficiency
                    }
                    
                    explanation = analytics.predict_with_explanation(institution_data)
                    
                    # Display results
                    st.success(f"**Prediction:** {explanation['prediction']}")
                    st.info(f"**Confidence:** {explanation['probability']:.1%}")
                    
                    if explanation['strengths']:
                        st.success("**✅ Strengths:**")
                        for strength in explanation['strengths']:
                            st.write(f"• {strength}")
                    
                    if explanation['weaknesses']:
                        st.error("**⚠️ Improvements Needed:**")
                        for weakness in explanation['weaknesses']:
                            st.write(f"• {weakness}")
        
        with col2:
            st.subheader("Model Insights")
            
            # Simple feature importance
            features_importance = {
                "NAAC Grade": 30,
                "Placement Rate": 25, 
                "Infrastructure": 20,
                "Compliance": 15,
                "Documentation": 10
            }
            
            fig = px.bar(x=list(features_importance.values()), 
                        y=list(features_importance.keys()),
                        orientation='h',
                        title="Feature Importance")
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
