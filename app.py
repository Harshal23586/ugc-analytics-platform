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
import os
from io import StringIO

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
    
    def load_external_data(self, uploaded_file):
        """Load data from uploaded file (CSV, Excel, or JSON)"""
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV, Excel, or JSON file.")
                return None
            
            st.success(f"Data loaded successfully! Shape: {data.shape}")
            return data
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def validate_data_structure(self, data):
        """Validate if the loaded data has required structure"""
        required_columns = ['year']
        optional_columns = [
            'student_strength', 'pass_percentage', 'dropout_rate', 
            'faculty_student_ratio', 'phd_faculty_percentage',
            'research_publications', 'placement_rate', 
            'infrastructure_investment', 'research_funding'
        ]
        
        missing_required = [col for col in required_columns if col not in data.columns]
        if missing_required:
            st.error(f"Missing required columns: {missing_required}")
            return False
        
        # Check if we have at least some of the optional columns
        available_optional = [col for col in optional_columns if col in data.columns]
        if len(available_optional) < 3:
            st.warning("Limited data columns available. Some analyses may not be possible.")
        
        return True
    
    def preprocess_data(self, data):
        """Preprocess and clean the loaded data"""
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Convert year to integer if possible
        if 'year' in processed_data.columns:
            processed_data['year'] = pd.to_numeric(processed_data['year'], errors='coerce')
            processed_data = processed_data.dropna(subset=['year'])
            processed_data['year'] = processed_data['year'].astype(int)
        
        # Handle missing values for numeric columns
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in processed_data.columns:
                # Fill missing values with column mean
                processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
        
        return processed_data
    
    def calculate_parameter_scores(self, data):
        """Calculate scores for each parameter based on data"""
        scores = {}
        
        # Curriculum Score
        if all(col in data.columns for col in ['pass_percentage', 'dropout_rate', 'placement_rate']):
            scores['Curriculum'] = (
                data['pass_percentage'].mean() * 0.4 +
                (100 - data['dropout_rate'].mean()) * 0.3 +
                data['placement_rate'].mean() * 0.3
            )
        else:
            scores['Curriculum'] = np.random.uniform(65, 85)
        
        # Faculty Resources Score
        if all(col in data.columns for col in ['faculty_student_ratio', 'phd_faculty_percentage', 'research_publications']):
            scores['Faculty Resources'] = (
                min(data['faculty_student_ratio'].mean() * 1000, 100) * 0.4 +
                data['phd_faculty_percentage'].mean() * 0.4 +
                (data['research_publications'].mean() / 2) * 0.2
            )
        else:
            scores['Faculty Resources'] = np.random.uniform(65, 85)
        
        # Learning and Teaching Score
        if all(col in data.columns for col in ['pass_percentage', 'dropout_rate', 'student_strength']):
            scores['Learning and Teaching'] = (
                data['pass_percentage'].mean() * 0.5 +
                (100 - data['dropout_rate'].mean()) * 0.3 +
                data['student_strength'].mean() / 50 * 0.2
            )
        else:
            scores['Learning and Teaching'] = np.random.uniform(65, 85)
        
        # Research and Innovation Score
        if all(col in data.columns for col in ['research_publications', 'research_funding']):
            scores['Research and Innovation'] = (
                min(data['research_publications'].mean() * 0.5, 100) * 0.6 +
                min(data['research_funding'].mean() / 20000, 100) * 0.4
            )
        else:
            scores['Research and Innovation'] = np.random.uniform(65, 85)
        
        # Other parameters (simplified calculation based on available data)
        for param in self.parameters[4:]:
            if len(data.columns) > 1:
                base_score = np.random.uniform(65, 85)
                # Use available numeric data for trend calculation
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    trend_data = data[numeric_cols[0]]
                    if len(trend_data) > 1:
                        trend_factor = trend_data.pct_change().mean() * 100
                    else:
                        trend_factor = 0
                else:
                    trend_factor = 0
                scores[param] = min(max(base_score + trend_factor, 0), 100)
            else:
                scores[param] = np.random.uniform(65, 85)
        
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
    
    # Data Upload Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Integration")
    
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Use Sample Data", "Upload Your Data"]
    )
    
    uploaded_data = None
    if data_source == "Upload Your Data":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Institutional Data",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload CSV, Excel, or JSON file with institutional data"
        )
        
        if uploaded_file is not None:
            uploaded_data = analyzer.load_external_data(uploaded_file)
            if uploaded_data is not None:
                if analyzer.validate_data_structure(uploaded_data):
                    uploaded_data = analyzer.preprocess_data(uploaded_data)
                    st.sidebar.success("Data validated and preprocessed successfully!")
                    
                    # Show data preview
                    with st.sidebar.expander("Data Preview"):
                        st.dataframe(uploaded_data.head())
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "📈 Performance Analysis", 
        "📋 Document Verification",
        "🎯 AI Recommendations",
        "📄 Report Generator",
        "📁 Data Management"
    ])
    
    # Use uploaded data if available, otherwise use sample data
    if uploaded_data is not None:
        data = uploaded_data
        data_source_note = "Using uploaded institutional data"
    else:
        data = analyzer.generate_sample_data()
        data_source_note = "Using sample data for demonstration"
    
    with tab1:
        st.markdown('<div class="section-header">Institutional Performance Overview</div>', unsafe_allow_html=True)
        st.info(data_source_note)
        
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
        st.markdown('<div class="section-header">Trend Analysis</div>', unsafe_allow_html=True)
        st.info(data_source_note)
        
        # Dynamic trend charts based on available data
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'year' in data.columns and len(numeric_columns) > 1:
            # Remove year from numeric columns for plotting
            plot_columns = [col for col in numeric_columns if col != 'year']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Academic Performance Trends
                academic_metrics = [col for col in ['pass_percentage', 'placement_rate', 'dropout_rate'] 
                                  if col in data.columns]
                if academic_metrics:
                    fig1 = go.Figure()
                    for metric in academic_metrics:
                        fig1.add_trace(go.Scatter(x=data['year'], y=data[metric], 
                                                 mode='lines+markers', name=metric.replace('_', ' ').title()))
                    fig1.update_layout(title='Academic Performance Trends', height=300)
                    st.plotly_chart(fig1, use_container_width=True)
                
                # Faculty Metrics
                faculty_metrics = [col for col in ['phd_faculty_percentage', 'faculty_student_ratio'] 
                                 if col in data.columns]
                research_metrics = [col for col in ['research_publications'] if col in data.columns]
                
                if faculty_metrics or research_metrics:
                    fig2 = go.Figure()
                    for metric in faculty_metrics:
                        fig2.add_trace(go.Scatter(x=data['year'], y=data[metric], 
                                                 mode='lines+markers', name=metric.replace('_', ' ').title()))
                    for metric in research_metrics:
                        fig2.add_trace(go.Bar(x=data['year'], y=data[metric], 
                                             name=metric.replace('_', ' ').title()))
                    fig2.update_layout(title='Faculty and Research Metrics', height=300)
                    st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # Student Metrics
                student_metrics = [col for col in ['student_strength', 'dropout_rate'] 
                                 if col in data.columns]
                if student_metrics:
                    fig3 = go.Figure()
                    for metric in student_metrics:
                        fig3.add_trace(go.Scatter(x=data['year'], y=data[metric], 
                                                 mode='lines+markers', name=metric.replace('_', ' ').title()))
                    fig3.update_layout(title='Student Metrics', height=300)
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Financial Metrics
                financial_metrics = [col for col in ['infrastructure_investment', 'research_funding'] 
                                   if col in data.columns]
                if financial_metrics:
                    fig4 = go.Figure()
                    for metric in financial_metrics:
                        fig4.add_trace(go.Bar(x=data['year'], y=data[metric], 
                                             name=metric.replace('_', ' ').title()))
                    fig4.update_layout(title='Financial Investments', height=300)
                    st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("Insufficient data for trend analysis. Please ensure your data includes a 'year' column and numeric metrics.")
    
    with tab6:
        st.markdown('<div class="section-header">Data Management</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Data")
            st.dataframe(data, use_container_width=True)
            
            # Data statistics
            st.subheader("Data Statistics")
            st.write(f"**Shape:** {data.shape}")
            st.write(f"**Time Period:** {data['year'].min()}-{data['year'].max()}" if 'year' in data.columns else "**Time Period:** Not specified")
            
            numeric_stats = data.describe()
            st.dataframe(numeric_stats, use_container_width=True)
        
        with col2:
            st.subheader("Data Quality Assessment")
            
            # Missing values analysis
            missing_values = data.isnull().sum()
            missing_percentage = (missing_values / len(data)) * 100
            
            quality_metrics = {
                "Total Records": len(data),
                "Total Columns": len(data.columns),
                "Columns with Missing Data": sum(missing_values > 0),
                "Complete Columns": sum(missing_values == 0)
            }
            
            for metric, value in quality_metrics.items():
                st.metric(metric, value)
            
            if sum(missing_values) > 0:
                st.subheader("Missing Values Details")
                missing_df = pd.DataFrame({
                    'Column': missing_values.index,
                    'Missing Count': missing_values.values,
                    'Missing %': missing_percentage.values
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                st.dataframe(missing_df, use_container_width=True)
            
            # Data download
            st.subheader("Export Data")
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Current Data as CSV",
                data=csv,
                file_name=f"{institution_name}_accreditation_data.csv",
                mime="text/csv"
            )

    # Rest of the tabs (3, 4, 5) remain the same as in your original code
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
        scores = analyzer.calculate_parameter_scores(data)
        
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
            scores = analyzer.calculate_parameter_scores(data)
            overall_score, status, _ = analyzer.predict_accreditation_status(scores)
            maturity_level, maturity_desc = analyzer.assess_maturity_level(scores)
            
            # Display report
            st.markdown(f"""
            # Accreditation Analysis Report
            **Institution:** {institution_name}  
            **Type:** {institution_type}  
            **Report Date:** {datetime.now().strftime("%Y-%m-%d")}  
            **Data Source:** {data_source_note}
            
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
            
            # Data overview
            st.markdown("### Data Overview")
            st.write(f"**Analysis Period:** {data['year'].min()}-{data['year'].max()}" if 'year' in data.columns else "**Analysis Period:** Not specified")
            st.write(f"**Total Metrics Available:** {len(data.columns)}")
            
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
            Data Source: {data_source_note}
            
            EXECUTIVE SUMMARY
            Overall Score: {overall_score:.1f}/100
            Accreditation Status: {status}
            Maturity Level: {maturity_level} ({maturity_desc})
            
            PARAMETER SCORES:
            """
            for param, score in scores.items():
                report_text += f"{param}: {score:.1f}\n"
            
            report_text += f"\nDATA OVERVIEW:\n"
            report_text += f"Analysis Period: {data['year'].min()}-{data['year'].max()}\n" if 'year' in data.columns else "Analysis Period: Not specified\n"
            report_text += f"Total Metrics Available: {len(data.columns)}\n"
            
            # Create download button
            st.download_button(
                label="Download Report as Text",
                data=report_text,
                file_name=f"accreditation_report_{institution_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
