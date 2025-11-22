import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="UGC/AICTE Analytics Platform",
    page_icon="🎓",
    layout="wide"
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'institutions_data' not in st.session_state:
    st.session_state.institutions_data = None

def simple_approval_predictor(performance, infrastructure, faculty_ratio):
    """Simple rule-based approval predictor (no scikit-learn needed)"""
    score = (performance * 0.4 + infrastructure * 0.3 + (100 - faculty_ratio) * 0.3)
    
    if score >= 75:
        return "🟢 APPROVED", score, "High performance across all metrics"
    elif score >= 60:
        return "🟡 PENDING REVIEW", score, "Meets basic requirements, needs additional review"
    else:
        return "🔴 NEEDS IMPROVEMENT", score, "Significant improvements required"

def main():
    st.markdown('<div class="main-header">🎓 UGC/AICTE Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Approval Process Analysis • 100% Working**")
    
    # Sidebar navigation
    menu = st.sidebar.selectbox("Navigation", [
        "🏠 Dashboard", "📊 Data Management", "🤖 AI Analysis", "📈 Reports"
    ])
    
    if menu == "🏠 Dashboard":
        show_dashboard()
    elif menu == "📊 Data Management":
        show_data_management()
    elif menu == "🤖 AI Analysis":
        show_ai_analysis()
    elif menu == "📈 Reports":
        show_reports()

def show_dashboard():
    st.header("🏠 Institutional Analytics Dashboard")
    
    if st.session_state.institutions_data is None:
        # Generate sample data for demo
        sample_data = generate_sample_data()
        st.session_state.institutions_data = sample_data
        st.info("📊 Sample data loaded for demonstration")
    
    df = st.session_state.institutions_data
    
    # Key Metrics
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Institutions", len(df))
    with col2:
        st.metric("Average Performance", f"{df['performance_score'].mean():.1f}")
    with col3:
        approved = (df['approval_status'] == 'APPROVED').sum()
        st.metric("Approved", f"{approved}/{len(df)}")
    with col4:
        st.metric("High Risk", f"{(df['performance_score'] < 60).sum()}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.pie(df, names='institution_type', title='Institution Type Distribution')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.histogram(df, x='performance_score', title='Performance Score Distribution')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Performance trends
    fig3 = px.line(df, x='establishment_year', y='performance_score', 
                  color='institution_type', title='Performance by Establishment Year')
    st.plotly_chart(fig3, use_container_width=True)

def show_data_management():
    st.header("📊 Data Management")
    
    upload_option = st.radio("Choose data source:", ["Upload CSV/Excel", "Use Sample Data"])
    
    if upload_option == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload institutional data", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded {len(df)} records")
                st.dataframe(df.head())
                
                # Basic validation
                required_cols = ['institution_name', 'performance_score']
                missing = [col for col in required_cols if col not in df.columns]
                
                if missing:
                    st.warning(f"Missing columns: {missing}")
                else:
                    st.session_state.institutions_data = df
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:
        if st.button("Generate Sample Data"):
            sample_df = generate_sample_data()
            st.session_state.institutions_data = sample_df
            st.success("✅ Sample data generated!")
            st.dataframe(sample_df)

def show_ai_analysis():
    st.header("🤖 AI-Powered Analysis")
    
    st.info("🎯 Smart approval prediction using rule-based algorithms")
    
    # Single institution analysis
    st.subheader("Single Institution Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        performance = st.slider("Academic Performance", 0, 100, 75)
        inst_name = st.text_input("Institution Name", "New Institution")
    
    with col2:
        infrastructure = st.slider("Infrastructure Score", 0, 100, 70)
        students = st.number_input("Student Strength", 1000, 50000, 5000)
    
    with col3:
        faculty_ratio = st.slider("Faculty:Student Ratio", 5, 50, 15)
        pass_rate = st.slider("Pass Percentage", 0, 100, 75)
    
    if st.button("Predict Approval"):
        status, score, reasoning = simple_approval_predictor(performance, infrastructure, faculty_ratio)
        
        st.subheader("Prediction Results")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Approval Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightcoral"},
                    {'range': [60, 75], 'color': "lightyellow"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Approval Status", status)
            st.metric("Overall Score", f"{score:.1f}")
        with col2:
            st.info(f"**Reasoning:** {reasoning}")
        
        # Recommendations
        st.subheader("Recommendations")
        if "APPROVED" in status:
            st.success("✅ Institution meets all approval criteria")
            st.write("- Maintain current performance levels")
            st.write("- Continue quality improvement initiatives")
        elif "REVIEW" in status:
            st.warning("⚠️ Additional review required")
            st.write("- Submit additional documentation")
            st.write("- Address minor improvement areas")
        else:
            st.error("🚨 Significant improvements needed")
            st.write("- Focus on academic performance improvement")
            st.write("- Enhance infrastructure facilities")
            st.write("- Optimize faculty-student ratio")

def show_reports():
    st.header("📈 Reports & Analytics")
    
    if st.session_state.institutions_data is None:
        st.warning("Please load data first")
        return
    
    df = st.session_state.institutions_data
    
    report_type = st.selectbox("Report Type", [
        "Performance Summary",
        "Comparative Analysis", 
        "Approval Recommendations"
    ])
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            
            if report_type == "Performance Summary":
                st.subheader("📋 Performance Summary Report")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Institutions", len(df))
                    st.metric("Average Score", f"{df['performance_score'].mean():.1f}")
                with col2:
                    approved = (df['approval_status'] == 'APPROVED').sum()
                    st.metric("Approval Rate", f"{(approved/len(df))*100:.1f}%")
                
                # Top performers
                st.subheader("🏆 Top Performing Institutions")
                top_5 = df.nlargest(5, 'performance_score')[['institution_name', 'performance_score', 'approval_status']]
                st.dataframe(top_5)
                
            elif report_type == "Comparative Analysis":
                st.subheader("📊 Comparative Analysis")
                
                # By institution type
                comparison = df.groupby('institution_type').agg({
                    'performance_score': ['mean', 'count'],
                    'student_strength': 'mean'
                }).round(2)
                st.dataframe(comparison)
                
            else:
                st.subheader("🎯 Approval Recommendations")
                
                # Risk analysis
                high_risk = df[df['performance_score'] < 60]
                medium_risk = df[(df['performance_score'] >= 60) & (df['performance_score'] < 75)]
                
                st.write(f"**High Risk Institutions:** {len(high_risk)}")
                st.write(f"**Medium Risk Institutions:** {len(medium_risk)}")
                
                if len(high_risk) > 0:
                    st.warning("The following institutions need immediate attention:")
                    st.dataframe(high_risk[['institution_name', 'performance_score']])

def generate_sample_data():
    """Generate realistic sample data"""
    np.random.seed(42)
    
    data = {
        'institution_name': [f'Institution {i:03d}' for i in range(1, 21)],
        'institution_type': np.random.choice(['Public', 'Private', 'Deemed'], 20),
        'establishment_year': np.random.randint(1950, 2020, 20),
        'student_strength': np.random.randint(1000, 25000, 20),
        'faculty_ratio': np.random.randint(10, 25, 20),
        'pass_percentage': np.random.uniform(60, 95, 20),
        'infrastructure_score': np.random.uniform(50, 95, 20),
        'performance_score': np.random.uniform(50, 95, 20),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate approval status based on performance
    df['approval_status'] = np.where(
        df['performance_score'] >= 75, 'APPROVED',
        np.where(df['performance_score'] >= 60, 'PENDING', 'REVIEW NEEDED')
    )
    
    return df

if __name__ == "__main__":
    main()
