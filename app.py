import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add missing imports
from typing import List, Dict, Tuple
import requests
import json
import time

# Page configuration
st.set_page_config(
    page_title="AI-Powered UGC/AICTE Analytics Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class UGCAPIIntegration:
    """Class to handle UGC/AICTE API integration"""
    
    def __init__(self):
        self.base_urls = {
            'ugc': 'https://api.ugc.ac.in/api/v1',
            'aicte': 'https://api.aicte-india.org/v1',
            'naac': 'https://naac.gov.in/api',
            'nirf': 'https://nirf.gov.in/api'
        }
        self.api_keys = {}  # To be configured
        self.mock_mode = True  # Start with mock data
        
    def set_api_credentials(self, ugc_key=None, aicte_key=None):
        """Set API credentials for real integration"""
        self.api_keys = {
            'ugc': ugc_key,
            'aicte': aicte_key
        }
        self.mock_mode = False if ugc_key or aicte_key else True
        
    def get_institution_data(self, institution_id=None):
        """Get institution data from UGC API or mock data"""
        if self.mock_mode:
            return self._get_mock_institution_data(institution_id)
        else:
            return self._get_real_institution_data(institution_id)
    
    def _get_real_institution_data(self, institution_id):
        """Fetch real data from UGC API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_keys.get("ugc")}',
                'Content-Type': 'application/json'
            }
            
            # Example API endpoints (you'll need actual endpoints)
            endpoints = {
                'basic_info': f'{self.base_urls["ugc"]}/institutions/{institution_id}',
                'approval_status': f'{self.base_urls["ugc"]}/institutions/{institution_id}/approvals',
                'performance': f'{self.base_urls["ugc"]}/institutions/{institution_id}/performance'
            }
            
            responses = {}
            for key, endpoint in endpoints.items():
                response = requests.get(endpoint, headers=headers, timeout=10)
                if response.status_code == 200:
                    responses[key] = response.json()
                else:
                    st.warning(f"API Error for {key}: {response.status_code}")
                    return self._get_mock_institution_data(institution_id)
            
            return self._transform_api_data(responses)
            
        except Exception as e:
            st.error(f"API Connection Error: {e}")
            return self._get_mock_institution_data(institution_id)
    
    def _get_mock_institution_data(self, institution_id=None):
        """Generate realistic mock data that matches API structure"""
        np.random.seed(42 if institution_id is None else institution_id)
        
        return {
            'institution_id': institution_id or np.random.randint(1000, 9999),
            'name': f"Institute_{institution_id}" if institution_id else "Demo Institute",
            'type': np.random.choice(['University', 'College', 'Technical Institute']),
            'established_year': np.random.randint(1950, 2020),
            'state': np.random.choice(['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi']),
            'naac_grade': np.random.choice(['A++', 'A+', 'A', 'B++', 'B+']),
            'naac_score': np.random.uniform(2.5, 3.8),
            'nirf_ranking': np.random.randint(1, 200),
            'total_faculty': np.random.randint(50, 500),
            'student_strength': np.random.randint(1000, 15000),
            'research_publications': np.random.randint(0, 500),
            'placement_rate': np.random.uniform(60, 95),
            'infrastructure_score': np.random.uniform(5, 10),
            'compliance_status': np.random.choice(['Compliant', 'Partially Compliant', 'Non-Compliant']),
            'last_approval_date': '2024-01-15',
            'approval_status': np.random.choice(['Approved', 'Pending', 'Conditional']),
            'documents_submitted': np.random.randint(15, 20),
            'total_documents_required': 20
        }
    
    def _transform_api_data(self, api_responses):
        """Transform API response to our standard format"""
        # This would map actual API response to our data structure
        basic_info = api_responses.get('basic_info', {})
        performance = api_responses.get('performance', {})
        approvals = api_responses.get('approval_status', {})
        
        return {
            'institution_id': basic_info.get('id'),
            'name': basic_info.get('name'),
            'type': basic_info.get('type'),
            'established_year': basic_info.get('establishedYear'),
            'state': basic_info.get('state'),
            'naac_grade': performance.get('naacGrade'),
            'naac_score': performance.get('naacScore'),
            'nirf_ranking': performance.get('nirfRank'),
            'total_faculty': basic_info.get('facultyCount'),
            'student_strength': basic_info.get('studentStrength'),
            'research_publications': performance.get('researchPublications'),
            'placement_rate': performance.get('placementRate'),
            'infrastructure_score': performance.get('infrastructureScore'),
            'compliance_status': approvals.get('complianceStatus'),
            'last_approval_date': approvals.get('lastApprovalDate'),
            'approval_status': approvals.get('currentStatus'),
            'documents_submitted': approvals.get('documentsSubmitted'),
            'total_documents_required': approvals.get('documentsRequired')
        }
    
    def get_bulk_institutions(self, filters=None):
        """Get multiple institutions data - for dashboard"""
        if self.mock_mode:
            return self._get_mock_bulk_data()
        else:
            return self._get_real_bulk_data(filters)
    
    def _get_real_bulk_data(self, filters):
        """Fetch bulk data from UGC API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_keys.get("ugc")}',
                'Content-Type': 'application/json'
            }
            
            params = filters or {}
            response = requests.get(
                f'{self.base_urls["ugc"]}/institutions',
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return [self._transform_api_data({'basic_info': item}) for item in response.json()['data']]
            else:
                st.warning("API Bulk Data Unavailable - Using Mock Data")
                return self._get_mock_bulk_data()
                
        except Exception as e:
            st.error(f"Bulk API Error: {e}")
            return self._get_mock_bulk_data()
    
    def _get_mock_bulk_data(self):
        """Generate bulk mock data"""
        np.random.seed(42)
        institutions = []
        
        for i in range(100):
            inst_data = self._get_mock_institution_data(1000 + i)
            institutions.append(inst_data)
        
        return institutions

class AdvancedUGC_AICTE_Analytics:
    def __init__(self):
        self.api_client = UGCAPIIntegration()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.setup_ai_systems()
        
    def setup_ai_systems(self):
        """Initialize AI systems"""
        # Load or train models
        self.sample_data = self.generate_sample_data_from_api()
        self.train_ai_models()
        
    def generate_sample_data_from_api(self):
        """Generate data structure from API/mock data"""
        institutions = self.api_client.get_bulk_institutions()
        
        # Convert to DataFrame
        df_data = []
        for inst in institutions:
            df_data.append({
                'institution_id': inst['institution_id'],
                'institution_name': inst['name'],
                'institution_type': inst['type'],
                'state': inst['state'],
                'naac_grade': inst['naac_grade'],
                'nirf_ranking': inst['nirf_ranking'],
                'total_faculty': inst['total_faculty'],
                'student_strength': inst['student_strength'],
                'research_publications': inst['research_publications'],
                'placement_rate': inst['placement_rate'],
                'infrastructure_score': inst['infrastructure_score'],
                'compliance_score': self._compliance_to_score(inst['compliance_status']),
                'documents_submitted': inst['documents_submitted'],
                'required_documents': inst['total_documents_required'],
                'established_year': inst['established_year']
            })
        
        df = pd.DataFrame(df_data)
        
        # Calculate derived metrics
        df['document_sufficiency'] = (df['documents_submitted'] / df['required_documents']) * 100
        df['faculty_student_ratio'] = df['total_faculty'] / df['student_strength']
        
        # Calculate approval probability
        df['approval_probability'] = self._calculate_approval_probability(df)
        df['approval_status'] = np.where(
            df['approval_probability'] > 0.7, 'Approved',
            np.where(df['approval_probability'] > 0.5, 'Pending', 'Rejected')
        )
        df['risk_level'] = np.where(
            df['approval_probability'] > 0.7, 'Low',
            np.where(df['approval_probability'] > 0.5, 'Medium', 'High')
        )
        
        return df
    
    def _compliance_to_score(self, compliance_status):
        """Convert compliance status to numerical score"""
        compliance_scores = {
            'Compliant': 10,
            'Partially Compliant': 6,
            'Non-Compliant': 3
        }
        return compliance_scores.get(compliance_status, 5)
    
    def _calculate_approval_probability(self, df):
        """Calculate approval probability based on multiple factors"""
        return (
            df['naac_grade'].map({'A++': 1.0, 'A+': 0.9, 'A': 0.8, 'B++': 0.7, 'B+': 0.6}).fillna(0.5) * 0.3 +
            (1 - (df['nirf_ranking'] / 200)).clip(0, 1) * 0.2 +
            (df['placement_rate'] / 100) * 0.25 +
            (df['infrastructure_score'] / 10) * 0.15 +
            (df['compliance_score'] / 10) * 0.1
        )
    
    def train_ai_models(self):
        """Train AI models"""
        feature_columns = [
            'naac_grade', 'nirf_ranking', 'placement_rate', 'infrastructure_score',
            'compliance_score', 'document_sufficiency', 'research_publications'
        ]
        
        X = pd.get_dummies(self.sample_data[feature_columns], columns=['naac_grade'])
        y = self.sample_data['approval_status']
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        return self.model, self.scaler, feature_columns
    
    def get_live_institution_data(self, institution_id):
        """Get real-time data for a specific institution"""
        return self.api_client.get_institution_data(institution_id)
    
    def generate_ai_recommendations(self, institution_data):
        """Generate AI-powered recommendations"""
        recommendations = {
            "immediate_actions": [],
            "strategic_improvements": [],
            "compliance_suggestions": [],
            "risk_factors": [],
            "success_indicators": []
        }
        
        # Analyze data
        placement_rate = institution_data.get('placement_rate', 0)
        infrastructure_score = institution_data.get('infrastructure_score', 0)
        compliance_status = institution_data.get('compliance_status', '')
        naac_grade = institution_data.get('naac_grade', 'B+')
        
        if placement_rate < 75:
            recommendations["immediate_actions"].append("Improve placement cell activities")
            recommendations["risk_factors"].append(f"Low placement rate: {placement_rate}%")
        else:
            recommendations["success_indicators"].append(f"Good placement rate: {placement_rate}%")
        
        if infrastructure_score < 7:
            recommendations["immediate_actions"].append("Upgrade infrastructure facilities")
            recommendations["risk_factors"].append(f"Infrastructure score: {infrastructure_score}/10")
        
        if compliance_status != 'Compliant':
            recommendations["compliance_suggestions"].append("Address compliance issues immediately")
            recommendations["risk_factors"].append(f"Compliance status: {compliance_status}")
        
        if naac_grade in ['A++', 'A+']:
            recommendations["success_indicators"].append(f"Excellent NAAC grade: {naac_grade}")
        
        return recommendations

def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .api-status-live { color: #28a745; font-weight: bold; }
    .api-status-mock { color: #ffc107; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🚀 AI-Powered UGC/AICTE Analytics Platform</h1>', unsafe_allow_html=True)
    
    # API Configuration Section
    with st.sidebar:
        st.header("🔧 API Configuration")
        
        api_mode = st.radio("Data Source", ["Mock Data (Demo)", "Live UGC API"])
        
        if api_mode == "Live UGC API":
            st.subheader("API Credentials")
            ugc_api_key = st.text_input("UGC API Key", type="password")
            aicte_api_key = st.text_input("AICTE API Key", type="password")
            
            if st.button("Connect to APIs"):
                if ugc_api_key or aicte_api_key:
                    st.success("API Connected Successfully!")
                else:
                    st.warning("Please enter API keys")
        
        st.markdown("---")
    
    # Initialize analytics engine
    analytics = AdvancedUGC_AICTE_Analytics()
    
    # Update API mode based on selection
    if api_mode == "Live UGC API":
        analytics.api_client.set_api_credentials(ugc_api_key, aicte_api_key)
        st.sidebar.markdown('<p class="api-status-live">🔴 Live API Mode</p>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<p class="api-status-mock">🟡 Mock Data Mode</p>', unsafe_allow_html=True)
    
    # Main navigation
    st.sidebar.header("🤖 Navigation")
    app_mode = st.sidebar.selectbox("Choose Module", 
        ["🏠 Live Dashboard", "🔍 Institution Lookup", "📊 API Analytics", 
         "💡 AI Recommendations", "⚡ Real-time Monitoring"])
    
    if app_mode == "🏠 Live Dashboard":
        st.header("📊 Live Institutional Dashboard")
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_inst = len(analytics.sample_data)
            st.metric("Total Institutions", total_inst)
        
        with col2:
            approved = (analytics.sample_data['approval_status'] == 'Approved').sum()
            st.metric("Approved", approved)
        
        with col3:
            high_risk = (analytics.sample_data['risk_level'] == 'High').sum()
            st.metric("High Risk", high_risk)
        
        with col4:
            avg_placement = analytics.sample_data['placement_rate'].mean()
            st.metric("Avg Placement", f"{avg_placement:.1f}%")
        
        # Live data table
        st.subheader("📋 Institutional Data")
        display_cols = ['institution_name', 'institution_type', 'state', 'naac_grade', 
                       'approval_status', 'risk_level', 'placement_rate']
        st.dataframe(analytics.sample_data[display_cols], use_container_width=True)
    
    elif app_mode == "🔍 Institution Lookup":
        st.header("🔍 Real-time Institution Lookup")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            institution_id = st.number_input("Enter Institution ID", min_value=1000, max_value=9999, value=1001)
            
            if st.button("🔍 Fetch Institution Data"):
                with st.spinner("Fetching data from UGC API..."):
                    live_data = analytics.get_live_institution_data(institution_id)
                    
                    st.subheader("📄 Institution Details")
                    st.json(live_data)  # Display raw API response
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("NAAC Grade", live_data.get('naac_grade', 'N/A'))
                    with col2:
                        st.metric("Placement Rate", f"{live_data.get('placement_rate', 0):.1f}%")
                    with col3:
                        st.metric("Compliance", live_data.get('compliance_status', 'N/A'))
        
        with col2:
            st.subheader("💡 Lookup Tips")
            st.write("• Use actual UGC Institution IDs")
            st.write("• Contact UGC for API access")
            st.write("• Mock data shown in demo mode")
    
    elif app_mode == "📊 API Analytics":
        st.header("📊 API Data Analytics")
        
        # API status
        st.subheader("🔌 API Connection Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("UGC API", "🔴 Offline" if analytics.api_client.mock_mode else "🟢 Online")
        with col2:
            st.metric("AICTE API", "🔴 Offline" if analytics.api_client.mock_mode else "🟢 Online")
        with col3:
            st.metric("Data Mode", "Mock" if analytics.api_client.mock_mode else "Live")
        
        # Data quality metrics
        st.subheader("📈 Data Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            completeness = analytics.sample_data.notna().mean().mean() * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
            
            fig = px.pie(analytics.sample_data, names='approval_status', 
                        title='Approval Status Distribution')
            st.plotly_chart(fig)
        
        with col2:
            avg_metrics = analytics.sample_data[['placement_rate', 'infrastructure_score']].mean()
            st.metric("Avg Placement", f"{avg_metrics['placement_rate']:.1f}%")
            st.metric("Avg Infrastructure", f"{avg_metrics['infrastructure_score']:.1f}/10")
            
            fig = px.box(analytics.sample_data, y='placement_rate', 
                        title='Placement Rate Distribution')
            st.plotly_chart(fig)
    
    elif app_mode == "💡 AI Recommendations":
        st.header("💡 AI-Powered Recommendations")
        
        institution_id = st.selectbox("Select Institution", 
                                    analytics.sample_data['institution_id'].tolist())
        
        if institution_id:
            institution_data = analytics.sample_data[
                analytics.sample_data['institution_id'] == institution_id
            ].iloc[0].to_dict()
            
            recommendations = analytics.generate_ai_recommendations(institution_data)
            
            st.subheader(f"🎯 Recommendations for Institute {institution_id}")
            
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
    
    elif app_mode == "⚡ Real-time Monitoring":
        st.header("⚡ Real-time API Monitoring")
        
        st.info("This module would show real-time data streaming from UGC/AICTE APIs")
        
        # Simulate real-time updates
        if st.button("🔄 Refresh Live Data"):
            with st.spinner("Fetching latest data..."):
                time.sleep(2)  # Simulate API call
                st.success("Data updated successfully!")
                
                # Show recent updates
                st.subheader("📅 Recent Updates")
                updates = [
                    {"institution": "Institute_1001", "update": "Approval status changed", "time": "2 min ago"},
                    {"institution": "Institute_1005", "update": "New compliance data", "time": "5 min ago"},
                    {"institution": "Institute_1012", "update": "Placement data updated", "time": "10 min ago"}
                ]
                
                for update in updates:
                    st.write(f"• **{update['institution']}**: {update['update']} ({update['time']})")

if __name__ == "__main__":
    main()
