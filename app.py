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
from typing import Dict

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

class AdvancedUGC_AICTE_Analytics:
    def __init__(self):
        self.sample_data = self.generate_comprehensive_data()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        
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
    
    # Additional filters
    filter_col5, filter_col6, filter_col7, filter_col8 = st.columns(4)
    
    with filter_col5:
        min_placement = st.slider(
            "Min Placement Rate (%)",
            min_value=60, max_value=95, value=60
        )
    
    with filter_col6:
        min_infra = st.slider(
            "Min Infrastructure Score",
            min_value=3.0, max_value=10.0, value=3.0, step=0.5
        )
    
    with filter_col7:
        risk_levels = st.multiselect(
            "Risk Level",
            options=df['risk_level'].unique(),
            default=df['risk_level'].unique()
        )
    
    with filter_col8:
        approval_statuses = st.multiselect(
            "Approval Status",
            options=df['approval_status'].unique(),
            default=df['approval_status'].unique()
        )
    
    # Apply filters
    filtered_df = df[
        (df['institution_type'].isin(institution_types)) &
        (df['ownership'].isin(ownership_types)) &
        (df['state'].isin(states)) &
        (df['naac_grade'].isin(naac_grades)) &
        (df['placement_rate'] >= min_placement) &
        (df['infrastructure_score'] >= min_infra) &
        (df['risk_level'].isin(risk_levels)) &
        (df['approval_status'].isin(approval_statuses))
    ]
    
    st.info(f"📊 Showing {len(filtered_df)} out of {len(df)} institutions")
    
    if len(filtered_df) == 0:
        st.warning("No institutions match the selected filters. Please adjust your criteria.")
        return
    
    # Main Visualization Grid
    st.subheader("📈 Comprehensive Institutional Analytics")
    
    # Row 1: Key Performance Indicators
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
        # NAAC Grade Distribution
        naac_counts = filtered_df['naac_grade'].value_counts()
        fig3 = px.bar(
            x=naac_counts.index,
            y=naac_counts.values,
            title="NAAC Grade Distribution",
            color=naac_counts.values,
            color_continuous_scale='Viridis'
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    # Row 2: Performance Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Placement Rate vs Infrastructure Score
        fig4 = px.scatter(
            filtered_df,
            x='infrastructure_score',
            y='placement_rate',
            color='approval_status',
            size='student_strength',
            hover_data=['institution_name', 'naac_grade'],
            title="Placement Rate vs Infrastructure Score",
            color_discrete_map={'Approved': '#2ecc71', 'Pending': '#f39c12', 'Rejected': '#e74c3c'}
        )
        fig4.update_layout(
            xaxis_title="Infrastructure Score (/10)",
            yaxis_title="Placement Rate (%)"
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        # Compliance Score Distribution by Institution Type
        fig5 = px.box(
            filtered_df,
            x='institution_type',
            y='compliance_score',
            color='ownership',
            title="Compliance Score by Institution Type & Ownership",
            points="all"
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    # Row 3: Geographic and Temporal Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance by State
        state_performance = filtered_df.groupby('state').agg({
            'approval_probability': 'mean',
            'placement_rate': 'mean',
            'institution_id': 'count'
        }).reset_index()
        
        fig6 = px.bar(
            state_performance.nlargest(10, 'institution_id'),
            x='state',
            y='approval_probability',
            color='placement_rate',
            title="Top 10 States by Approval Probability",
            hover_data=['institution_id'],
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        # Performance Trend by Establishment Year
        filtered_df['establishment_decade'] = (filtered_df['established_year'] // 10) * 10
        decade_stats = filtered_df.groupby('establishment_decade').agg({
            'approval_probability': 'mean',
            'placement_rate': 'mean',
            'infrastructure_score': 'mean',
            'institution_id': 'count'
        }).reset_index()
        
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(
            x=decade_stats['establishment_decade'],
            y=decade_stats['approval_probability'],
            mode='lines+markers',
            name='Approval Probability',
            line=dict(color='#3498db', width=3)
        ))
        fig7.add_trace(go.Scatter(
            x=decade_stats['establishment_decade'],
            y=decade_stats['placement_rate']/100,
            mode='lines+markers',
            name='Placement Rate (scaled)',
            line=dict(color='#e74c3c', width=3)
        ))
        fig7.update_layout(
            title="Performance Trends by Establishment Decade",
            xaxis_title="Establishment Decade",
            yaxis_title="Score (Normalized)",
            showlegend=True
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    # Row 4: Detailed Analysis
    st.subheader("🔬 Detailed Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation Heatmap
        numeric_cols = ['approval_probability', 'placement_rate', 'infrastructure_score', 
                       'compliance_score', 'research_publications', 'document_sufficiency']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig8 = px.imshow(
            corr_matrix,
            title="Performance Metrics Correlation Matrix",
            aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig8, use_container_width=True)
    
    with col2:
        # Faculty-Student Ratio Analysis
        filtered_df['faculty_category'] = pd.cut(
            filtered_df['faculty_student_ratio'],
            bins=[0, 0.02, 0.05, 0.1, 1],
            labels=['Very Low (<0.02)', 'Low (0.02-0.05)', 'Good (0.05-0.1)', 'Excellent (>0.1)']
        )
        
        faculty_stats = filtered_df.groupby('faculty_category').agg({
            'approval_probability': 'mean',
            'placement_rate': 'mean',
            'institution_id': 'count'
        }).reset_index()
        
        fig9 = px.bar(
            faculty_stats,
            x='faculty_category',
            y='approval_probability',
            color='placement_rate',
            title="Approval Probability by Faculty-Student Ratio",
            hover_data=['institution_id'],
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    # Row 5: Institutional Comparison
    st.subheader("🏆 Institutional Leaderboard")
    
    # Top and Bottom Performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🥇 Top 10 Performing Institutions**")
        top_performers = filtered_df.nlargest(10, 'approval_probability')[
            ['institution_name', 'institution_type', 'naac_grade', 'approval_probability', 'placement_rate']
        ]
        top_performers['approval_probability'] = top_performers['approval_probability'].apply(lambda x: f"{x:.1%}")
        top_performers['placement_rate'] = top_performers['placement_rate'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(top_performers, use_container_width=True)
    
    with col2:
        st.write("**📉 Institutions Needing Improvement**")
        bottom_performers = filtered_df.nsmallest(10, 'approval_probability')[
            ['institution_name', 'institution_type', 'naac_grade', 'approval_probability', 'risk_level']
        ]
        bottom_performers['approval_probability'] = bottom_performers['approval_probability'].apply(lambda x: f"{x:.1%}")
        st.dataframe(bottom_performers, use_container_width=True)
    
    # Row 6: Export and Detailed View
    st.subheader("📤 Export & Detailed Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Detailed Report"):
            st.success("Detailed report generated! (This would typically export to PDF/Excel)")
    
    with col2:
        # Quick statistics
        st.write("**📈 Quick Stats:**")
        st.write(f"• Highest Placement: {filtered_df['placement_rate'].max():.1f}%")
        st.write(f"• Best Infrastructure: {filtered_df['infrastructure_score'].max():.1f}/10")
        st.write(f"• Top NAAC Grade: {filtered_df['naac_grade'].mode()[0]}")
    
    with col3:
        # Risk analysis
        high_risk_count = len(filtered_df[filtered_df['risk_level'] == 'High'])
        if high_risk_count > 0:
            st.warning(f"🚨 {high_risk_count} high-risk institutions need immediate attention")
        else:
            st.success("✅ No high-risk institutions in current selection")

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
            
            if recommendations["compliance_suggestions"]:
                st.info("**📋 Compliance & Documentation:**")
                for suggestion in recommendations["compliance_suggestions"]:
                    st.write(f"• {suggestion}")
    
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
            
            st.write("**🎯 Key Factors:**")
            for factor, importance in explanation['key_factors'][:3]:
                st.write(f"• {factor}")

def create_rag_system():
    st.header("🔍 Guidelines Query System")
    
    st.success("Query UGC/AICTE guidelines and regulations.")
    
    # Simple knowledge base
    guidelines_db = {
        "Faculty Requirements": [
            "Minimum 1:20 faculty-student ratio for undergraduate programs",
            "At least 30% of faculty should hold PhD degrees",
            "Faculty must have minimum 3 years of teaching experience"
        ],
        "Infrastructure Standards": [
            "Digital classrooms with smart boards and projectors required",
            "Library must have minimum 5000 books and digital resources",
            "Laboratories should have modern equipment less than 5 years old"
        ],
        "Placement Criteria": [
            "Minimum 60% placement rate for technical institutions",
            "Industry collaborations and MoUs are mandatory",
            "Career guidance and placement cell must be established"
        ],
        "Documentation": [
            "Complete land and building documents required",
            "Faculty qualification certificates must be verified",
            "Financial statements for last 3 years audited by CA"
        ],
        "Compliance": [
            "Must meet all statutory UGC/AICTE requirements",
            "Regular audits and inspections will be conducted",
            "Student feedback mechanism with 70%+ satisfaction required"
        ]
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query interface
        query = st.text_input("Search guidelines:", 
                            placeholder="e.g., faculty qualification requirements")
        
        if st.button("🔍 Search Guidelines") and query:
            st.subheader("📚 Relevant Guidelines:")
            
            found_results = False
            for category, guidelines in guidelines_db.items():
                for guideline in guidelines:
                    if query.lower() in guideline.lower():
                        if not found_results:
                            found_results = True
                        with st.expander(f"📄 {category}"):
                            st.write(guideline)
            
            if not found_results:
                st.warning("No relevant guidelines found. Try different keywords.")
    
    with col2:
        st.subheader("💡 Sample Queries")
        sample_queries = [
            "Faculty student ratio",
            "Infrastructure standards", 
            "NAAC accreditation",
            "Document requirements",
            "Placement criteria"
        ]
        
        for sample in sample_queries:
            if st.button(sample):
                st.session_state.last_query = sample

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
            research_publications = st.number_input("Research Publications", 0, 1000, 50)
            
            submitted = st.form_submit_button("🚀 Predict Approval")
            
            if submitted:
                institution_data = {
                    'naac_grade': naac_grade,
                    'placement_rate': placement_rate,
                    'infrastructure_score': infrastructure_score,
                    'compliance_score': compliance_score,
                    'document_sufficiency': document_sufficiency,
                    'research_publications': research_publications
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
        
        fig = px.bar(
            x=list(features_importance.values()), 
            y=list(features_importance.keys()),
            orientation='h',
            title="Feature Importance in Approval Prediction",
            color=list(features_importance.values()),
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**📊 Prediction Guidelines:**")
        st.write("• **>70%**: High approval chance")
        st.write("• **50-70%**: Requires improvements")  
        st.write("• **<50%**: Significant issues detected")

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
        ["🏢 Institutional Dashboard", "💡 AI Recommendation Engine", "🔍 Guidelines Query", 
         "🔮 Predictive Analytics"])
    
    if app_mode == "🏢 Institutional Dashboard":
        create_enhanced_dashboard(df)
    
    elif app_mode == "💡 AI Recommendation Engine":
        create_recommendation_engine(df, analytics)
    
    elif app_mode == "🔍 Guidelines Query":
        create_rag_system()
    
    elif app_mode == "🔮 Predictive Analytics":
        create_predictive_analytics(analytics)

if __name__ == "__main__":
    main()
