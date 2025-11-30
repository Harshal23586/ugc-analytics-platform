import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Higher Education Accreditation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

class AccreditationAnalyzer:
    def __init__(self):
        self.parameters = [
            "Curriculum", "Faculty Resources", "Learning and Teaching", 
            "Research and Innovation", "Extracurricular Activities",
            "Community Engagement", "Green Initiatives", 
            "Governance and Administration", "Infrastructure Development",
            "Financial Resources"
        ]
        
    def generate_sample_data(self):
        """Generate sample institutional data for demonstration"""
        np.random.seed(42)
        
        # Generate 10-year trend data
        years = list(range(2014, 2024))
        
        data = {
            'year': years,
            'student_strength': np.random.randint(1000, 5000, len(years)),
            'pass_percentage': np.random.uniform(75, 95, len(years)),
            'dropout_rate': np.random.uniform(2, 8, len(years)),
            'faculty_student_ratio': np.random.uniform(0.05, 0.15, len(years)),
            'phd_faculty_percentage': np.random.uniform(60, 90, len(years)),
            'research_publications': np.random.randint(50, 200, len(years)),
            'placement_rate': np.random.uniform(70, 95, len(years)),
            'infrastructure_investment': np.random.randint(1000000, 5000000, len(years)),
            'research_funding': np.random.randint(500000, 2000000, len(years))
        }
        
        return pd.DataFrame(data)
    
    def calculate_parameter_scores(self, data):
        """Calculate scores for each parameter based on data"""
        scores = {}
        
        # Curriculum Score
        scores['Curriculum'] = (
            data['pass_percentage'].mean() * 0.4 +
            (100 - data['dropout_rate'].mean()) * 0.3 +
            data['placement_rate'].mean() * 0.3
        )
        
        # Faculty Resources Score
        scores['Faculty Resources'] = (
            min(data['faculty_student_ratio'].mean() * 1000, 100) * 0.4 +
            data['phd_faculty_percentage'].mean() * 0.4 +
            (data['research_publications'].mean() / 2) * 0.2
        )
        
        # Learning and Teaching Score
        scores['Learning and Teaching'] = (
            data['pass_percentage'].mean() * 0.5 +
            (100 - data['dropout_rate'].mean()) * 0.3 +
            data['student_strength'].mean() / 50 * 0.2
        )
        
        # Research and Innovation Score
        scores['Research and Innovation'] = (
            min(data['research_publications'].mean() * 0.5, 100) * 0.6 +
            min(data['research_funding'].mean() / 20000, 100) * 0.4
        )
        
        # Other parameters (simplified calculation)
        for param in self.parameters[4:]:
            base_score = np.random.uniform(65, 85)
            trend_factor = data['pass_percentage'].pct_change().mean() * 100
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

def main():
    st.markdown('<div class="main-header">🎓 Higher Education Accreditation Analysis System</div>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = AccreditationAnalyzer()
    
    # Sidebar
    st.sidebar.title("Institution Details")
    institution_name = st.sidebar.text_input("Institution Name", "Sample University")
    institution_type = st.sidebar.selectbox(
        "Institution Type",
        ["Technical (AICTE + NBA)", "Non-Technical (UGC + NAAC)"]
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "📈 Performance Analysis", 
        "📋 Document Verification",
        "🎯 AI Recommendations",
        "📄 Report Generator"
    ])
    
    with tab1:
        st.markdown('<div class="section-header">Institutional Performance Overview</div>', unsafe_allow_html=True)
        
        # Generate sample data
        data = analyzer.generate_sample_data()
        
        # Calculate scores
        scores = analyzer.calculate_parameter_scores(data)
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        overall_score, status, status_class = analyzer.predict_accreditation_status(scores)
        maturity_level, maturity_desc = analyzer.assess_maturity_level(scores)
        
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
    
    with tab2:
        st.markdown('<div class="section-header">10-Year Trend Analysis</div>', unsafe_allow_html=True)
        
        # Trend charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Academic Performance Trends
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=data['year'], y=data['pass_percentage'], 
                                     mode='lines+markers', name='Pass Percentage'))
            fig1.add_trace(go.Scatter(x=data['year'], y=data['placement_rate'], 
                                     mode='lines+markers', name='Placement Rate'))
            fig1.update_layout(title='Academic Performance Trends', height=300)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Faculty Metrics
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data['year'], y=data['phd_faculty_percentage'], 
                                     mode='lines+markers', name='PhD Faculty %'))
            fig2.add_trace(go.Bar(x=data['year'], y=data['research_publications'], 
                                 name='Research Publications'))
            fig2.update_layout(title='Faculty and Research Metrics', height=300)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Student Metrics
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=data['year'], y=data['student_strength'], 
                                     mode='lines+markers', name='Student Strength'))
            fig3.add_trace(go.Scatter(x=data['year'], y=data['dropout_rate'], 
                                     mode='lines+markers', name='Dropout Rate'))
            fig3.update_layout(title='Student Metrics', height=300)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Financial Metrics
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=data['year'], y=data['infrastructure_investment'], 
                                 name='Infrastructure Investment'))
            fig4.add_trace(go.Bar(x=data['year'], y=data['research_funding'], 
                                 name='Research Funding'))
            fig4.update_layout(title='Financial Investments', height=300)
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-header">Document Verification Checklist</div>', unsafe_allow_html=True)
        
        # Document categories
        categories = {
            "A. Institutional Basic Documents": [
                "Certificate of Incorporation/Establishment",
                "University/Statutory Approval Letters",
                "Organizational Structure Chart",
                "Strategic Plan/Institutional Development Plan",
                "Academic Calendar (last 10 years)"
            ],
            "B. Curriculum Related": [
                "Program Curriculum and Syllabi",
                "Curriculum Review Committee Minutes",
                "Industry-Academia Interaction Records",
                "Student Feedback Reports on Curriculum",
                "Course Completion and Pass Percentage Data"
            ],
            "C. Faculty Resources": [
                "Faculty Database with Qualifications",
                "Faculty Recruitment Policy and Records",
                "Faculty Development Program Records",
                "Research Publications List (10 years)",
                "Faculty Performance Appraisal Records"
            ]
        }
        
        document_status = {}
        
        for category, documents in categories.items():
            st.subheader(category)
            for doc in documents:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"• {doc}")
                with col2:
                    status = st.selectbox(
                        f"Status_{doc}",
                        ["Not Uploaded", "Uploaded", "Verified", "Rejected"],
                        key=f"status_{doc}"
                    )
                with col3:
                    if status == "Uploaded":
                        st.button("Verify", key=f"verify_{doc}")
                
                document_status[doc] = status
        
        # Document completeness score
        total_docs = len(document_status)
        uploaded_docs = sum(1 for status in document_status.values() if status in ["Uploaded", "Verified"])
        verified_docs = sum(1 for status in document_status.values() if status == "Verified")
        
        completeness_score = (uploaded_docs / total_docs) * 100
        verification_score = (verified_docs / total_docs) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Document Completeness", f"{completeness_score:.1f}%")
        with col2:
            st.metric("Verification Status", f"{verification_score:.1f}%")
    
    with tab4:
        st.markdown('<div class="section-header">AI Analysis & Recommendations</div>', unsafe_allow_html=True)
        
        # Strengths and Weaknesses
        scores = analyzer.calculate_parameter_scores(analyzer.generate_sample_data())
        
        # Identify top 3 strengths and weaknesses
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        strengths = sorted_scores[:3]
        weaknesses = sorted_scores[-3:]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Key Strengths")
            for param, score in strengths:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{param}</strong><br>
                    Score: {score:.1f}/100
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("⚠️ Areas for Improvement")
            for param, score in weaknesses:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{param}</strong><br>
                    Score: {score:.1f}/100
                </div>
                """, unsafe_allow_html=True)
        
        # Risk Assessment
        st.subheader("🔍 Risk Assessment")
        
        risks = [
            ("Curriculum Relevance", "Medium", "Regular industry interaction needed"),
            ("Faculty Retention", "Low", "Good retention rates observed"),
            ("Research Funding", "High", "Need to diversify funding sources"),
            ("Infrastructure Maintenance", "Medium", "Regular upgrades required")
        ]
        
        for risk, level, description in risks:
            risk_color = {"High": "red", "Medium": "orange", "Low": "green"}[level]
            st.markdown(f"""
            <div style="border-left: 4px solid {risk_color}; padding: 10px; margin: 5px 0;">
                <strong>{risk}</strong> | <span style="color: {risk_color}">{level} Risk</span><br>
                {description}
            </div>
            """, unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="section-header">Accreditation Report Generator</div>', unsafe_allow_html=True)
        
        # Generate comprehensive report
        if st.button("Generate Full Accreditation Report"):
            
            # Create report data
            data = analyzer.generate_sample_data()
            scores = analyzer.calculate_parameter_scores(data)
            overall_score, status, _ = analyzer.predict_accreditation_status(scores)
            maturity_level, maturity_desc = analyzer.assess_maturity_level(scores)
            
            # Display report
            st.markdown(f"""
            # Accreditation Analysis Report
            **Institution:** {institution_name}  
            **Type:** {institution_type}  
            **Report Date:** {datetime.now().strftime("%Y-%m-%d")}  
            
            ## Executive Summary
            - **Overall Score:** {overall_score:.1f}/100
            - **Accreditation Status:** {status}
            - **Maturity Level:** {maturity_level} ({maturity_desc})
            
            ## Detailed Analysis
            
            ### Parameter-wise Performance
            """)
            
            # Parameter scores table
            score_df = pd.DataFrame(list(scores.items()), columns=['Parameter', 'Score'])
            st.dataframe(score_df, use_container_width=True)
            
            # Recommendations
            st.markdown("""
            ## Recommendations
            
            ### Immediate Actions (0-6 months)
            1. Enhance research publication quality and quantity
            2. Strengthen industry-academia collaboration
            3. Improve digital learning infrastructure
            
            ### Medium-term Goals (6-18 months)
            1. Develop interdisciplinary programs
            2. Enhance international collaborations
            3. Implement sustainable campus initiatives
            
            ### Long-term Vision (18+ months)
            1. Achieve global recognition in specific domains
            2. Establish innovation and incubation center
            3. Develop comprehensive alumni engagement program
            """)
            
            # Download report
            report_text = f"""
            ACCREDITATION ANALYSIS REPORT
            Institution: {institution_name}
            Type: {institution_type}
            Date: {datetime.now().strftime("%Y-%m-%d")}
            
            EXECUTIVE SUMMARY
            Overall Score: {overall_score:.1f}/100
            Accreditation Status: {status}
            Maturity Level: {maturity_level} ({maturity_desc})
            
            PARAMETER SCORES:
            """
            for param, score in scores.items():
                report_text += f"{param}: {score:.1f}\n"
            
            # Create download button
            st.download_button(
                label="Download Report as Text",
                data=report_text,
                file_name=f"accreditation_report_{institution_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
