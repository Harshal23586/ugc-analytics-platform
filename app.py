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
from typing import Dict, List, Tuple, Any
import hashlib
import sqlite3
import io
import base64

# Page configuration
st.set_page_config(
    page_title="AI-Powered Institutional Approval System - UGC/AICTE",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InstitutionalAIAnalyzer:
    def __init__(self):
        self.init_database()
        self.historical_data = self.load_or_generate_data()
        self.performance_metrics = self.define_performance_metrics()
        self.document_requirements = self.define_document_requirements()
        
    def init_database(self):
        """Initialize SQLite database for storing institutional data"""
        self.conn = sqlite3.connect('institutions.db', check_same_thread=False)
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
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS institution_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                institution_id TEXT,
                document_name TEXT,
                document_type TEXT,
                file_path TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'Pending',
                FOREIGN KEY (institution_id) REFERENCES institutions (institution_id)
            )
        ''')
        
        self.conn.commit()
    
    def load_or_generate_data(self):
        """Load data from database or generate sample data"""
        try:
            # Try to load from database
            df = pd.read_sql('SELECT * FROM institutions', self.conn)
            if len(df) > 0:
                return df
        except:
            pass
        
        # Generate sample data if database is empty
        sample_data = self.generate_comprehensive_historical_data()
        sample_data.to_sql('institutions', self.conn, if_exists='replace', index=False)
        return sample_data
    
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
            },
            "governance_administration": {
                "weight": 0.15,
                "sub_metrics": {
                    "financial_stability": 0.30,
                    "administrative_efficiency": 0.25,
                    "compliance_record": 0.25,
                    "grievance_redressal": 0.20
                }
            },
            "student_development": {
                "weight": 0.15,
                "sub_metrics": {
                    "placement_rate": 0.35,
                    "higher_education_rate": 0.20,
                    "entrepreneurship_cell": 0.15,
                    "extracurricular_activities": 0.15,
                    "alumni_network": 0.15
                }
            },
            "social_impact": {
                "weight": 0.10,
                "sub_metrics": {
                    "community_engagement": 0.30,
                    "rural_outreach": 0.25,
                    "inclusive_education": 0.25,
                    "environmental_initiatives": 0.20
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
            },
            "renewal_approval": {
                "mandatory": [
                    "previous_approval_letters", "annual_reports", "financial_audit_reports",
                    "faculty_student_data", "infrastructure_utilization", "academic_performance"
                ],
                "supporting": [
                    "naac_accreditation", "nirf_data", "research_publications",
                    "placement_records", "social_impact_reports"
                ]
            },
            "expansion_approval": {
                "mandatory": [
                    "current_status_report", "expansion_justification", "additional_infrastructure",
                    "enhanced_faculty_plan", "financial_viability", "market_analysis"
                ],
                "supporting": [
                    "stakeholder_feedback", "alumni_support", "industry_demand",
                    "government_schemes_participation"
                ]
            }
        }
    
    def generate_comprehensive_historical_data(self) -> pd.DataFrame:
        """Generate comprehensive historical data for institutions"""
        np.random.seed(42)
        n_institutions = 200
        years_of_data = 5
        
        institutions_data = []
        
        for inst_id in range(1, n_institutions + 1):
            base_quality = np.random.uniform(0.3, 0.9)
            
            for year_offset in range(years_of_data):
                year = 2023 - year_offset
                inst_trend = base_quality + (year_offset * 0.02)
                
                # Generate realistic data with proper distributions
                naac_grades = ['A++', 'A+', 'A', 'B++', 'B+', 'B', 'C']
                naac_probs = [0.05, 0.10, 0.15, 0.25, 0.25, 0.15, 0.05]
                naac_grade = np.random.choice(naac_grades, p=naac_probs)
                
                nirf_rank = np.random.choice(list(range(1, 201)) + [None]*50, p=[0.005]*200 + [0.5])
                student_faculty_ratio = max(10, np.random.normal(20, 5))
                phd_faculty_ratio = np.random.beta(2, 2) * 0.6 + 0.3  # Beta distribution for ratios
                
                # Research metrics with realistic distributions
                publications = max(0, int(np.random.poisson(inst_trend * 30)))
                research_grants = max(0, int(np.random.exponential(inst_trend * 500000)))
                patents = np.random.poisson(inst_trend * 3)
                
                # Infrastructure scores
                digital_infrastructure_score = max(1, min(10, np.random.normal(7, 1.5)))
                library_volumes = max(1000, int(np.random.normal(20000, 10000)))
                
                # Governance scores
                financial_stability = max(1, min(10, np.random.normal(7.5, 1.2)))
                compliance_score = max(1, min(10, np.random.normal(8, 1)))
                
                # Student development
                placement_rate = max(40, min(98, np.random.normal(75, 10)))
                higher_education_rate = max(5, min(50, np.random.normal(20, 8)))
                
                # Social impact
                community_projects = np.random.poisson(inst_trend * 8)
                
                # Calculate performance score
                performance_score = self.calculate_performance_score({
                    'naac_grade': naac_grade,
                    'nirf_ranking': nirf_rank,
                    'student_faculty_ratio': student_faculty_ratio,
                    'phd_faculty_ratio': phd_faculty_ratio,
                    'publications_per_faculty': publications / max(1, np.random.randint(30, 150)),
                    'research_grants': research_grants,
                    'digital_infrastructure': digital_infrastructure_score,
                    'financial_stability': financial_stability,
                    'placement_rate': placement_rate,
                    'community_engagement': community_projects
                })
                
                institution_data = {
                    'institution_id': f'INST_{inst_id:04d}',
                    'institution_name': f'University/College {inst_id:03d}',
                    'year': year,
                    'institution_type': np.random.choice(['State University', 'Deemed University', 'Private University', 'Autonomous College'], p=[0.3, 0.2, 0.3, 0.2]),
                    'state': np.random.choice(['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Uttar Pradesh', 'Kerala', 'Gujarat'], p=[0.2, 0.15, 0.15, 0.1, 0.2, 0.1, 0.1]),
                    'established_year': np.random.randint(1950, 2015),
                    
                    # Academic Metrics
                    'naac_grade': naac_grade,
                    'nirf_ranking': nirf_rank,
                    'student_faculty_ratio': round(student_faculty_ratio, 1),
                    'phd_faculty_ratio': round(phd_faculty_ratio, 3),
                    
                    # Research Metrics
                    'research_publications': publications,
                    'research_grants_amount': research_grants,
                    'patents_filed': patents,
                    'industry_collaborations': np.random.poisson(inst_trend * 6),
                    
                    # Infrastructure Metrics
                    'digital_infrastructure_score': round(digital_infrastructure_score, 1),
                    'library_volumes': library_volumes,
                    'laboratory_equipment_score': round(max(1, min(10, np.random.normal(7, 1.3))), 1),
                    
                    # Governance Metrics
                    'financial_stability_score': round(financial_stability, 1),
                    'compliance_score': round(compliance_score, 1),
                    'administrative_efficiency': round(max(1, min(10, np.random.normal(7.2, 1.1))), 1),
                    
                    # Student Development Metrics
                    'placement_rate': round(placement_rate, 1),
                    'higher_education_rate': round(higher_education_rate, 1),
                    'entrepreneurship_cell_score': round(max(1, min(10, np.random.normal(6.5, 1.5))), 1),
                    
                    # Social Impact Metrics
                    'community_projects': community_projects,
                    'rural_outreach_score': round(max(1, min(10, np.random.normal(6.8, 1.4))), 1),
                    'inclusive_education_index': round(max(1, min(10, np.random.normal(7.5, 1.2))), 1),
                    
                    # Government Schemes Participation
                    'rusa_participation': np.random.choice([0, 1], p=[0.4, 0.6]),
                    'nmeict_participation': np.random.choice([0, 1], p=[0.5, 0.5]),
                    'fist_participation': np.random.choice([0, 1], p=[0.6, 0.4]),
                    'dst_participation': np.random.choice([0, 1], p=[0.7, 0.3]),
                    
                    # Overall Performance
                    'performance_score': round(performance_score, 2),
                    'approval_recommendation': self.generate_approval_recommendation(performance_score),
                    'risk_level': self.assess_risk_level(performance_score)
                }
                
                institutions_data.append(institution_data)
        
        return pd.DataFrame(institutions_data)
    
    def calculate_performance_score(self, metrics: Dict) -> float:
        """Calculate overall performance score based on weighted metrics"""
        score = 0
        
        # NAAC Grade scoring
        naac_scores = {'A++': 10, 'A+': 9, 'A': 8, 'B++': 7, 'B+': 6, 'B': 5, 'C': 4}
        naac_score = naac_scores.get(metrics['naac_grade'], 5)
        score += naac_score * 0.15
        
        # NIRF Ranking scoring (inverse)
        nirf_score = 0
        if metrics['nirf_ranking'] and metrics['nirf_ranking'] <= 200:
            nirf_score = (201 - metrics['nirf_ranking']) / 200 * 10
        score += nirf_score * 0.10
        
        # Student-Faculty Ratio (lower is better)
        sf_ratio_score = max(0, 10 - max(0, metrics['student_faculty_ratio'] - 15) / 3)
        score += sf_ratio_score * 0.10
        
        # PhD Faculty Ratio
        phd_score = metrics['phd_faculty_ratio'] * 10
        score += phd_score * 0.10
        
        # Research Publications
        pub_score = min(10, metrics['publications_per_faculty'] * 3)
        score += pub_score * 0.10
        
        # Research Grants (log scale)
        grant_score = min(10, np.log1p(metrics['research_grants'] / 100000) * 2.5)
        score += grant_score * 0.10
        
        # Infrastructure
        infra_score = metrics['digital_infrastructure']
        score += infra_score * 0.10
        
        # Financial Stability
        financial_score = metrics['financial_stability']
        score += financial_score * 0.10
        
        # Placement Rate
        placement_score = metrics['placement_rate'] / 10
        score += placement_score * 0.10
        
        # Community Engagement
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
    
    def assess_risk_level(self, performance_score: float) -> str:
        """Assess institutional risk level"""
        if performance_score >= 8.0:
            return "Low Risk"
        elif performance_score >= 6.5:
            return "Medium Risk"
        elif performance_score >= 5.0:
            return "High Risk"
        else:
            return "Critical Risk"
    
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
    
    def generate_comprehensive_report(self, institution_id: str) -> Dict[str, Any]:
        """Generate comprehensive AI analysis report for an institution"""
        inst_data = self.historical_data[
            self.historical_data['institution_id'] == institution_id
        ]
        
        if inst_data.empty:
            return {"error": "Institution not found"}
        
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
        
        # Comparative analysis
        similar_institutions = self.find_similar_institutions(institution_id)
        
        return {
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
            },
            "strengths": self.identify_strengths(latest_data),
            "weaknesses": self.identify_weaknesses(latest_data),
            "comparative_analysis": similar_institutions,
            "ai_recommendations": self.generate_ai_recommendations(latest_data)
        }
    
    def find_similar_institutions(self, institution_id: str) -> Dict:
        """Find similar institutions for comparative analysis"""
        inst_data = self.historical_data[
            self.historical_data['institution_id'] == institution_id
        ]
        
        if inst_data.empty:
            return {}
        
        latest_data = inst_data[inst_data['year'] == inst_data['year'].max()].iloc[0]
        
        # Find similar institutions based on type and performance
        similar_inst = self.historical_data[
            (self.historical_data['institution_type'] == latest_data['institution_type']) &
            (self.historical_data['year'] == latest_data['year']) &
            (self.historical_data['institution_id'] != institution_id)
        ]
        
        if len(similar_inst) > 0:
            similar_inst = similar_inst.nlargest(5, 'performance_score')
            benchmark_data = similar_inst[['institution_name', 'performance_score', 'approval_recommendation']].to_dict('records')
        else:
            benchmark_data = []
        
        return {
            "benchmark_institutions": benchmark_data,
            "performance_percentile": self.calculate_performance_percentile(latest_data['performance_score'], latest_data['institution_type'])
        }
    
    def calculate_performance_percentile(self, score: float, inst_type: str) -> float:
        """Calculate performance percentile within institution type"""
        type_data = self.historical_data[
            (self.historical_data['institution_type'] == inst_type) &
            (self.historical_data['year'] == 2023)
        ]
        
        if len(type_data) == 0:
            return 50.0
        
        return (type_data['performance_score'] < score).mean() * 100
    
    def identify_strengths(self, institution_data: pd.Series) -> List[str]:
        """Identify institutional strengths"""
        strengths = []
        
        if institution_data['naac_grade'] in ['A++', 'A+', 'A']:
            strengths.append(f"Excellent NAAC Accreditation: {institution_data['naac_grade']}")
        
        if institution_data['placement_rate'] > 80:
            strengths.append(f"Strong Placement Record: {institution_data['placement_rate']:.1f}%")
        
        if institution_data['research_publications'] > 100:
            strengths.append(f"Robust Research Output: {institution_data['research_publications']} publications")
        
        if institution_data['financial_stability_score'] > 8.5:
            strengths.append("Excellent Financial Stability")
        
        if institution_data['phd_faculty_ratio'] > 0.7:
            strengths.append(f"Highly Qualified Faculty: {institution_data['phd_faculty_ratio']:.1%} PhDs")
        
        if institution_data['digital_infrastructure_score'] > 8.5:
            strengths.append("Advanced Digital Infrastructure")
            
        return strengths
    
    def identify_weaknesses(self, institution_data: pd.Series) -> List[str]:
        """Identify institutional weaknesses"""
        weaknesses = []
        
        if institution_data['student_faculty_ratio'] > 25:
            weaknesses.append(f"High Student-Faculty Ratio: {institution_data['student_faculty_ratio']:.1f}")
        
        if institution_data['placement_rate'] < 65:
            weaknesses.append(f"Low Placement Rate: {institution_data['placement_rate']:.1f}%")
        
        if institution_data['research_publications'] < 20:
            weaknesses.append(f"Inadequate Research Output: {institution_data['research_publications']} publications")
        
        if institution_data['digital_infrastructure_score'] < 7:
            weaknesses.append(f"Weak Digital Infrastructure: {institution_data['digital_infrastructure_score']:.1f}/10")
        
        if institution_data['community_projects'] < 5:
            weaknesses.append("Limited Community Engagement")
            
        return weaknesses
    
    def generate_ai_recommendations(self, institution_data: pd.Series) -> List[str]:
        """Generate AI-powered improvement recommendations"""
        recommendations = []
        
        if institution_data['student_faculty_ratio'] > 25:
            recommendations.append("Recruit additional faculty members to improve student-faculty ratio")
        
        if institution_data['placement_rate'] < 70:
            recommendations.append("Strengthen industry partnerships and career development programs")
        
        if institution_data['research_publications'] < 50:
            recommendations.append("Establish research promotion policy and faculty development programs")
        
        if institution_data['digital_infrastructure_score'] < 7:
            recommendations.append("Invest in digital infrastructure and e-learning platforms")
        
        if institution_data['community_projects'] < 5:
            recommendations.append("Enhance community engagement and social outreach programs")
        
        return recommendations

def create_performance_dashboard(analyzer):
    st.header("📊 Institutional Performance Analytics Dashboard")
    
    df = analyzer.historical_data
    current_year_data = df[df['year'] == 2023]
    
    # Ensure we have data
    if len(current_year_data) == 0:
        st.warning("No data available for the current year. Please check data generation.")
        return
    
    # Key Performance Indicators
    st.subheader("🏆 Key Performance Indicators")
    
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
        research_intensity = current_year_data['research_publications'].sum() / current_year_data['research_publications'].count()
        st.metric("Avg Research Publications", f"{research_intensity:.1f}")
    
    # Performance Analysis
    st.subheader("📈 Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance Distribution
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
        # Performance by Institution Type
        fig2 = px.box(
            current_year_data,
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
    
    # Trend Analysis
    st.subheader("📅 Historical Performance Trends")
    
    trend_data = df.groupby(['year', 'institution_type'])['performance_score'].mean().reset_index()
    
    fig3 = px.line(
        trend_data,
        x='year',
        y='performance_score',
        color='institution_type',
        title="Average Performance Score Trend (2019-2023)",
        markers=True
    )
    fig3.update_layout(
        xaxis_title="Year", 
        yaxis_title="Average Performance Score",
        legend_title="Institution Type"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Risk Analysis
    st.subheader("⚠️ Institutional Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_distribution = current_year_data['risk_level'].value_counts()
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
        # Placement vs Research Analysis
        fig5 = px.scatter(
            current_year_data,
            x='research_publications',
            y='placement_rate',
            color='risk_level',
            size='performance_score',
            hover_data=['institution_name'],
            title="Research Output vs Placement Rate",
            color_discrete_map={
                'Low Risk': '#2ecc71',
                'Medium Risk': '#f39c12',
                'High Risk': '#e74c3c',
                'Critical Risk': '#c0392b'
            }
        )
        fig5.update_layout(
            xaxis_title="Research Publications",
            yaxis_title="Placement Rate (%)"
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    # Additional Visualizations
    st.subheader("🎯 Additional Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # State-wise Performance
        state_performance = current_year_data.groupby('state')['performance_score'].mean().sort_values(ascending=False).head(10)
        fig6 = px.bar(
            x=state_performance.index,
            y=state_performance.values,
            title="Top 10 States by Average Performance Score",
            color=state_performance.values,
            color_continuous_scale='Viridis'
        )
        fig6.update_layout(
            xaxis_title="State",
            yaxis_title="Average Performance Score",
            showlegend=False
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        # NAAC Grade Distribution
        naac_dist = current_year_data['naac_grade'].value_counts()
        fig7 = px.bar(
            x=naac_dist.index,
            y=naac_dist.values,
            title="NAAC Grade Distribution",
            color=naac_dist.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig7.update_layout(
            xaxis_title="NAAC Grade",
            yaxis_title="Number of Institutions",
            showlegend=False
        )
        st.plotly_chart(fig7, use_container_width=True)

def create_document_analysis_module(analyzer):
    st.header("📋 AI-Powered Document Sufficiency Analysis")
    
    st.info("Analyze document completeness and generate sufficiency reports for approval processes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Document Upload & Analysis")
        
        # Institution selection
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
                    key=f"doc_type_{i}"
                )
                document_types.append(doc_type)
            
            if st.button("💾 Save Documents & Analyze"):
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
            # Generate comprehensive report
            report = analyzer.generate_comprehensive_report(selected_institution)
            
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
                
                # Strengths and Weaknesses
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
                
                # AI Recommendations
                if report['ai_recommendations']:
                    st.warning("**🎯 AI Improvement Recommendations**")
                    for recommendation in report['ai_recommendations']:
                        st.write(f"• {recommendation}")
                else:
                    st.success("Institution is performing well across all parameters")
                
                # Comparative Analysis
                st.info("**📊 Comparative Analysis**")
                if report['comparative_analysis']:
                    st.write(f"**Performance Percentile:** {report['comparative_analysis']['performance_percentile']:.1f}%")
                    if report['comparative_analysis']['benchmark_institutions']:
                        st.write("**Benchmark Institutions:**")
                        for bench in report['comparative_analysis']['benchmark_institutions']:
                            st.write(f"• **{bench['institution_name']}**: {bench['performance_score']:.2f} - {bench['approval_recommendation']}")
                    else:
                        st.info("No similar institutions found for comparison")
                
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

def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🏛️ AI-Powered Institutional Approval Analytics System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">UGC & AICTE - Institutional Performance Tracking & Decision Support</h3>', unsafe_allow_html=True)
    
    # System overview
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
        <p>Authorized UGC/AICTE personnel only. All activities are logged and monitored.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize analytics engine
    try:
        analyzer = InstitutionalAIAnalyzer()
        st.success("✅ AI Analytics System Successfully Initialized!")
        
        # Display quick stats
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
                avg_performance = current_year_data['performance_score'].mean()
                st.metric("Avg Performance Score", f"{avg_performance:.2f}/10")
            else:
                st.metric("Avg Performance Score", "N/A")
        
        with col4:
            if len(current_year_data) > 0:
                approval_ready = (current_year_data['performance_score'] >= 6.0).sum()
                st.metric("Approval Ready", approval_ready)
            else:
                st.metric("Approval Ready", "N/A")
            
    except Exception as e:
        st.error(f"❌ System initialization error: {str(e)}")
        st.stop()
    
    # Navigation
    st.sidebar.title("🧭 Navigation Panel")
    st.sidebar.markdown("### AI Modules")
    
    app_mode = st.sidebar.selectbox(
        "Select Analysis Module",
        [
            "📊 Performance Dashboard",
            "📋 Document Analysis", 
            "🤖 AI Reports",
            "💾 Data Management",
            "🔄 Approval Workflow",
            "⚙️ System Settings"
        ]
    )
    
    # Authentication (simplified for demo)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Authentication")
    user_role = st.sidebar.selectbox(
        "User Role",
        ["UGC Officer", "AICTE Officer", "System Admin", "Review Committee"]
    )
    
    # Route to selected module
    if app_mode == "📊 Performance Dashboard":
        create_performance_dashboard(analyzer)
    
    elif app_mode == "📋 Document Analysis":
        create_document_analysis_module(analyzer)
    
    elif app_mode == "🤖 AI Reports":
        create_ai_analysis_reports(analyzer)
    
    elif app_mode == "💾 Data Management":
        create_data_management_module(analyzer)
    
    elif app_mode == "🔄 Approval Workflow":
        create_approval_workflow(analyzer)
    
    elif app_mode == "⚙️ System Settings":
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
