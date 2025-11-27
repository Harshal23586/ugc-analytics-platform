import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add missing imports
from typing import Dict, List, Tuple
import json

# Fix PyTorch conflict with Streamlit
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# Page configuration
st.set_page_config(
    page_title="AI-Powered UGC/AICTE Analytics Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DocumentComplianceAnalyzer:
    def __init__(self):
        self.mandatory_documents = self.get_mandatory_document_list()
        self.compliance_thresholds = self.get_compliance_thresholds()
    
    def get_mandatory_document_list(self) -> List[Dict]:
        """Define mandatory documents for new technical institutions"""
        return [
            {"id": "DOC-001", "name": "Affidavit 2", "category": "Legal", "critical": True},
            {"id": "DOC-002", "name": "Approved Building Plan", "category": "Infrastructure", "critical": True},
            {"id": "DOC-003", "name": "Land Ownership/Lease Deed", "category": "Infrastructure", "critical": True},
            {"id": "DOC-004", "name": "Occupancy/Completion Certificate", "category": "Infrastructure", "critical": True},
            {"id": "DOC-005", "name": "Fire Safety Certificate", "category": "Safety", "critical": True},
            {"id": "DOC-006", "name": "Financial Certificate (Bank)", "category": "Financial", "critical": True},
            {"id": "DOC-007", "name": "Trust/Society/Company Registration", "category": "Legal", "critical": True},
            {"id": "DOC-008", "name": "Resolution in Format 3", "category": "Administrative", "critical": True},
            {"id": "DOC-009", "name": "Faculty Roster/List", "category": "Academic", "critical": False},
            {"id": "DOC-010", "name": "Building Area Statement", "category": "Infrastructure", "critical": False},
            {"id": "DOC-011", "name": "Structural Stability Certificate", "category": "Infrastructure", "critical": False},
            {"id": "DOC-012", "name": "Library Inventory", "category": "Academic", "critical": False},
        ]
    
    def get_compliance_thresholds(self) -> Dict:
        """Define compliance thresholds from AICTE handbook"""
        return {
            "faculty_student_ratio_ug_engineering": 20,
            "faculty_student_ratio_ug_management": 25,
            "admin_area_min": 750,
            "amenities_area_min": 500,
            "internet_bandwidth_0_300": 100,
            "min_lease_period": 30,
            "placement_rate_min": 60,
            "document_sufficiency_min": 85,
            "infrastructure_score_min": 7,
            "compliance_score_min": 8
        }
    
    def check_document_presence(self, uploaded_files: List[str]) -> Tuple[pd.DataFrame, bool]:
        """Check which mandatory documents are present"""
        document_status = []
        all_critical_present = True
        
        for doc in self.mandatory_documents:
            status = "Missing"
            note = "Document not found"
            
            # Simulate document checking (in real implementation, this would use file parsing)
            for uploaded_file in uploaded_files:
                if doc["name"].lower() in uploaded_file.lower():
                    status = "Present"
                    note = "Document verified"
                    break
            
            if doc["critical"] and status == "Missing":
                all_critical_present = False
            
            document_status.append({
                "Document ID": doc["id"],
                "Document Name": doc["name"],
                "Category": doc["category"],
                "Critical": "Yes" if doc["critical"] else "No",
                "Status": status,
                "Note": note
            })
        
        return pd.DataFrame(document_status), all_critical_present
    
    def extract_parameter_values(self, document_data: Dict) -> Dict:
        """Extract actual parameter values from documents (simulated RAG)"""
        # In real implementation, this would use actual document parsing and RAG
        extracted_values = {}
        
        # Simulate extracted values based on document presence
        if document_data.get("Faculty Roster", False):
            extracted_values["faculty_count"] = np.random.randint(30, 100)
            extracted_values["student_intake"] = np.random.randint(500, 2000)
            extracted_values["faculty_student_ratio"] = extracted_values["faculty_count"] / extracted_values["student_intake"]
        
        if document_data.get("Building Area Statement", False):
            extracted_values["admin_area"] = np.random.randint(600, 900)
            extracted_values["amenities_area"] = np.random.randint(400, 700)
        
        if document_data.get("Financial Certificate", False):
            extracted_values["working_capital"] = np.random.randint(5000000, 20000000)
        
        return extracted_values
    
    def analyze_compliance(self, extracted_values: Dict) -> pd.DataFrame:
        """Compare extracted values with thresholds"""
        compliance_results = []
        
        parameters = [
            ("faculty_student_ratio", "Faculty-Student Ratio", "faculty_student_ratio_ug_engineering", "students_per_faculty", "Lower"),
            ("admin_area", "Administrative Area", "admin_area_min", "sq_m", "Higher"),
            ("amenities_area", "Amenities Area", "amenities_area_min", "sq_m", "Higher"),
            ("placement_rate", "Placement Rate", "placement_rate_min", "percentage", "Higher")
        ]
        
        for param_key, param_name, threshold_key, unit, better_direction in parameters:
            if param_key in extracted_values:
                actual = extracted_values[param_key]
                threshold = self.compliance_thresholds.get(threshold_key, 0)
                
                if better_direction == "Higher":
                    is_compliant = actual >= threshold
                    gap = threshold - actual if actual < threshold else 0
                else:  # Lower is better
                    is_compliant = actual <= threshold
                    gap = actual - threshold if actual > threshold else 0
                
                compliance_results.append({
                    "Parameter": param_name,
                    "Actual Value": f"{actual:.2f}",
                    "Threshold": f"{threshold}",
                    "Status": "Compliant" if is_compliant else "Non-Compliant",
                    "Gap": f"{gap:.2f}",
                    "Unit": unit
                })
        
        return pd.DataFrame(compliance_results)

class AdvancedUGC_AICTE_Analytics:
    def __init__(self):
        self.sample_data = self.generate_comprehensive_data()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.compliance_analyzer = DocumentComplianceAnalyzer()
        
    def generate_comprehensive_data(self):
        """Generate comprehensive institutional data"""
        np.random.seed(42)
        n_institutions = 150
        
        data = {
            'institution_id': range(1, n_institutions + 1),
            'institution_name': [f'Institute_{i:03d}' for i in range(1, n_institutions + 1)],
            'established_year': np.random.randint(1950, 2020, n_institutions),
            'institution_type': np.random.choice(['University', 'College', 'Technical Institute', 'Research Center'], n_institutions),
            'ownership': np.random.choice(['Government', 'Private', 'Deemed', 'Autonomous'], n_institutions),
            'state': np.random.choice([
                'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Uttar Pradesh', 
                'Kerala', 'Gujarat', 'Rajasthan', 'West Bengal', 'Andhra Pradesh'
            ], n_institutions),
            'naac_grade': np.random.choice(['A++', 'A+', 'A', 'B++', 'B+', 'B'], n_institutions, p=[0.05, 0.1, 0.15, 0.3, 0.25, 0.15]),
            'nirf_ranking': np.random.choice(range(1, 201), n_institutions),
            'total_faculty': np.random.randint(50, 500, n_institutions),
            'student_strength': np.random.randint(1000, 15000, n_institutions),
            'research_publications': np.random.randint(0, 500, n_institutions),
            'patents_filed': np.random.randint(0, 50, n_institutions),
            'placement_rate': np.random.uniform(60, 95, n_institutions),
            'infrastructure_score': np.random.uniform(5, 10, n_institutions),
            'financial_stability': np.random.uniform(5, 10, n_institutions),
            'compliance_score': np.random.uniform(6, 10, n_institutions),
            'documents_submitted': np.random.randint(70, 100, n_institutions),
            'required_documents': 100,
            'industry_collaborations': np.random.randint(0, 20, n_institutions),
            'international_students': np.random.randint(0, 200, n_institutions),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate derived metrics
        df['document_sufficiency'] = (df['documents_submitted'] / df['required_documents']) * 100
        df['faculty_student_ratio'] = df['total_faculty'] / df['student_strength']
        df['research_intensity'] = df['research_publications'] / np.maximum(df['total_faculty'], 1)
        
        # Generate approval status with sophisticated logic
        approval_score = (
            df['naac_grade'].map({'A++': 10, 'A+': 9, 'A': 8, 'B++': 7, 'B+': 6, 'B': 5}) * 0.25 +
            (1 - (df['nirf_ranking'] / 200)) * 0.15 +
            (df['placement_rate'] / 100) * 0.20 +
            (df['infrastructure_score'] / 10) * 0.15 +
            (df['compliance_score'] / 10) * 0.15 +
            (df['document_sufficiency'] / 100) * 0.10
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
        """Train AI models for different tasks"""
        feature_columns = [
            'naac_grade', 'nirf_ranking', 'placement_rate', 'infrastructure_score',
            'compliance_score', 'document_sufficiency', 'research_publications'
        ]
        
        X = pd.get_dummies(df[feature_columns], columns=['naac_grade'])
        y_approval = df['approval_status']
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_approval)
        
        return self.model, self.scaler, feature_columns
    
    def generate_ai_recommendations(self, institution_data: Dict) -> Dict:
        """Generate AI-powered recommendations"""
        recommendations = {
            "immediate_actions": [],
            "strategic_improvements": [],
            "compliance_suggestions": [],
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
                "Establish research centers and provide faculty research grants"
            )
            recommendations["risk_factors"].append(f"Low research output: {research_publications} publications")
        else:
            recommendations["success_indicators"].append(f"Strong research: {research_publications} publications")
        
        if compliance_score < 8:
            recommendations["compliance_suggestions"].append(
                "Conduct comprehensive compliance audit and address statutory requirements"
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
    
    def predict_with_explanation(self, institution_data):
        """Predict with detailed explanation of factors"""
        # Simple scoring logic
        placement_rate = institution_data.get('placement_rate', 0)
        infrastructure_score = institution_data.get('infrastructure_score', 0)
        compliance_score = institution_data.get('compliance_score', 0)
        document_sufficiency = institution_data.get('document_sufficiency', 0)
        naac_grade = institution_data.get('naac_grade', 'B+')
        
        score = (
            {'A++': 1.0, 'A+': 0.9, 'A': 0.8, 'B++': 0.7, 'B+': 0.6, 'B': 0.5}.get(naac_grade, 0.5) * 0.3 +
            (placement_rate / 100) * 0.25 +
            (infrastructure_score / 10) * 0.2 +
            (compliance_score / 10) * 0.15 +
            (document_sufficiency / 100) * 0.1
        )
        
        prediction = "Approved" if score > 0.7 else "Pending" if score > 0.5 else "Rejected"
        probability = min(score, 0.95)
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

def create_document_compliance_module(analytics):
    st.header("📋 Document Compliance Analyzer")
    st.info("Two-stage RAG system for document verification and compliance analysis")
    
    # Stage 1: Document Upload and Presence Check
    st.subheader("📤 Stage 1: Document Upload & Verification")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload Institutional Documents",
            type=['pdf', 'doc', 'docx', 'xlsx', 'jpg', 'png'],
            accept_multiple_files=True,
            help="Upload all required documents for compliance checking"
        )
        
        if uploaded_files:
            file_names = [file.name for file in uploaded_files]
            st.success(f"✅ {len(uploaded_files)} files uploaded successfully")
            
            # Check document presence
            document_status_df, all_critical_present = analytics.compliance_analyzer.check_document_presence(file_names)
            
            # Display document status
            st.subheader("📊 Document Status Dashboard")
            
            # Create status summary
            status_summary = document_status_df['Status'].value_counts()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_docs = len(document_status_df)
                st.metric("Total Documents", total_docs)
            
            with col2:
                present_docs = status_summary.get('Present', 0)
                st.metric("Documents Present", f"{present_docs}/{total_docs}")
            
            with col3:
                critical_status = "✅ Complete" if all_critical_present else "❌ Incomplete"
                st.metric("Critical Documents", critical_status)
            
            # Display detailed document status
            st.dataframe(
                document_status_df.style.apply(
                    lambda x: ['background-color: #ff6b6b' if v == 'Missing' and x['Critical'] == 'Yes' 
                              else 'background-color: #d4edda' if v == 'Present' 
                              else '' for v in x],
                    axis=1
                ),
                use_container_width=True
            )
    
    with col2:
        st.subheader("ℹ️ Compliance Guidelines")
        st.write("**Critical Documents:**")
        st.write("• Affidavit 2")
        st.write("• Approved Building Plan")
        st.write("• Land Ownership/Lease Deed")
        st.write("• Fire Safety Certificate")
        st.write("• Financial Certificate")
        
        st.write("**Important Notes:**")
        st.write("• All critical documents must be present")
        st.write("• Lease period: Minimum 30 years")
        st.write("• Documents must be duly attested")
    
    # Stage 2: Compliance Analysis
    if uploaded_files and all_critical_present:
        st.subheader("🔍 Stage 2: Compliance Parameter Analysis")
        
        # Simulate parameter extraction (in real implementation, this would use RAG)
        document_data = {doc["name"]: doc["Status"] == "Present" for doc in analytics.compliance_analyzer.mandatory_documents}
        extracted_values = analytics.compliance_analyzer.extract_parameter_values(document_data)
        
        if extracted_values:
            # Analyze compliance
            compliance_results = analytics.compliance_analyzer.analyze_compliance(extracted_values)
            
            # Display compliance results
            st.subheader("📈 Compliance Analysis Results")
            
            # Compliance metrics
            compliant_count = len(compliance_results[compliance_results['Status'] == 'Compliant'])
            total_params = len(compliance_results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Compliance Score", f"{compliant_count}/{total_params}")
            with col2:
                compliance_rate = (compliant_count / total_params) * 100
                st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
            with col3:
                non_compliant = total_params - compliant_count
                st.metric("Issues Found", non_compliant)
            
            # Display compliance table
            st.dataframe(
                compliance_results.style.apply(
                    lambda x: ['background-color: #d4edda' if v == 'Compliant' else 'background-color: #f8d7da' for v in x],
                    subset=['Status']
                ),
                use_container_width=True
            )
            
            # Generate recommendations
            st.subheader("💡 AI Recommendations")
            
            if non_compliant > 0:
                st.warning("**🚨 Immediate Actions Required:**")
                
                non_compliant_params = compliance_results[compliance_results['Status'] == 'Non-Compliant']
                for _, row in non_compliant_params.iterrows():
                    st.write(f"• **{row['Parameter']}**: Gap of {row['Gap']} {row['Unit']} below threshold")
                
                st.info("**📋 Suggested Improvements:**")
                st.write("• Review and update faculty recruitment strategy")
                st.write("• Enhance infrastructure facilities as per AICTE norms")
                st.write("• Strengthen industry partnerships for placements")
            else:
                st.success("✅ All parameters are compliant with AICTE standards!")
                
        else:
            st.warning("Unable to extract sufficient data from documents for compliance analysis.")
    
    elif uploaded_files and not all_critical_present:
        st.error("❌ Critical documents are missing. Please upload all required critical documents to proceed with compliance analysis.")

def create_enhanced_dashboard(df):
    st.header("🏢 Institutional Intelligence Dashboard")
    
    # Dashboard Overview Metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_inst = len(df)
        st.metric("Total Institutions", total_inst)
    
    with col2:
        approved = len(df[df['approval_status'] == 'Approved'])
        st.metric("Approved", f"{approved} ({approved/total_inst*100:.1f}%)")
    
    with col3:
        pending = len(df[df['approval_status'] == 'Pending'])
        st.metric("Pending", f"{pending} ({pending/total_inst*100:.1f}%)")
    
    with col4:
        rejected = len(df[df['approval_status'] == 'Rejected'])
        st.metric("Rejected", f"{rejected} ({rejected/total_inst*100:.1f}%)")
    
    with col5:
        avg_placement = df['placement_rate'].mean()
        st.metric("Avg Placement", f"{avg_placement:.1f}%")
    
    with col6:
        avg_compliance = df['compliance_score'].mean()
        st.metric("Avg Compliance", f"{avg_compliance:.1f}/10")
    
    # Filters Section
    st.subheader("🔍 Dashboard Filters")
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        institution_types = st.multiselect(
            "Institution Type",
            options=df['institution_type'].unique(),
            default=df['institution_type'].unique()
        )
    
    with filter_col2:
        ownership_types = st.multiselect(
            "Ownership",
            options=df['ownership'].unique(),
            default=df['ownership'].unique()
        )
    
    with filter_col3:
        states = st.multiselect(
            "State",
            options=df['state'].unique(),
            default=df['state'].unique()
        )
    
    with filter_col4:
        naac_grades = st.multiselect(
            "NAAC Grade",
            options=sorted(df['naac_grade'].unique()),
            default=sorted(df['naac_grade'].unique())
        )
    
    # Apply filters
    filtered_df = df[
        (df['institution_type'].isin(institution_types)) &
        (df['ownership'].isin(ownership_types)) &
        (df['state'].isin(states)) &
        (df['naac_grade'].isin(naac_grades))
    ]
    
    st.info(f"📊 Showing {len(filtered_df)} out of {len(df)} institutions")
    
    if len(filtered_df) == 0:
        st.warning("No institutions match the selected filters. Please adjust your criteria.")
        return
    
    # Main Visualization Grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Approval Status Distribution
        status_counts = filtered_df['approval_status'].value_counts()
        fig1 = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Approval Status Distribution",
            color=status_counts.index,
            color_discrete_map={'Approved': '#2ecc71', 'Pending': '#f39c12', 'Rejected': '#e74c3c'}
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Risk Level Distribution
        risk_counts = filtered_df['risk_level'].value_counts()
        fig2 = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Risk Level Distribution",
            color=risk_counts.index,
            color_discrete_map={'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'},
            labels={'x': 'Risk Level', 'y': 'Count'}
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        # Placement Rate vs Infrastructure Score
        fig3 = px.scatter(
            filtered_df,
            x='infrastructure_score',
            y='placement_rate',
            color='approval_status',
            size='student_strength',
            hover_data=['institution_name', 'naac_grade'],
            title="Placement Rate vs Infrastructure Score",
            color_discrete_map={'Approved': '#2ecc71', 'Pending': '#f39c12', 'Rejected': '#e74c3c'}
        )
        st.plotly_chart(fig3, use_container_width=True)

def create_recommendation_engine(df, analytics):
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
            
            # Display metrics
            st.metric("AI Approval Probability", f"{explanation['probability']:.1%}")
            st.metric("Predicted Status", explanation['prediction'])
            
            # Display risk level with color using emojis
            risk_level = explanation['risk_level']
            risk_color = "🟢" if risk_level == 'Low' else "🟡" if risk_level == 'Medium' else "🔴"
            st.metric("Risk Level", f"{risk_color} {risk_level}")

def create_predictive_analytics(analytics):
    st.header("🔮 Predictive Analytics")
    
    st.info("Predict approval probabilities for institutions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Approval Prediction")
        
        with st.form("prediction_form"):
            naac_grade = st.selectbox("NAAC Grade", ['A++', 'A+', 'A', 'B++', 'B+', 'B'])
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
    
    # Initialize analytics engine
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
        ["📋 Document Compliance", "🏢 Institutional Dashboard", "💡 AI Recommendation Engine", 
         "🔮 Predictive Analytics"])
    
    if app_mode == "📋 Document Compliance":
        create_document_compliance_module(analytics)
    
    elif app_mode == "🏢 Institutional Dashboard":
        create_enhanced_dashboard(df)
    
    elif app_mode == "💡 AI Recommendation Engine":
        create_recommendation_engine(df, analytics)
    
    elif app_mode == "🔮 Predictive Analytics":
        create_predictive_analytics(analytics)

if __name__ == "__main__":
    main()
