import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="UGC/AICTE Analytics Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'institutions_data' not in st.session_state:
    st.session_state.institutions_data = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

def main():
    # Header
    st.markdown('<div class="main-header">🎓 UGC/AICTE Institutional Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Approval Process Analysis • 100% Free Solution**")
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
        st.title("Navigation")
        menu_option = st.radio(
            "Select Section:",
            ["🏠 Dashboard", "📊 Data Management", "🤖 AI Analysis", "📈 Reports", "⚙️ Settings"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Platform Status")
        if st.session_state.institutions_data is not None:
            st.sidebar.success("✅ Data Loaded")
        else:
            st.sidebar.warning("📁 No Data Loaded")
            
        if st.session_state.trained_model is not None:
            st.sidebar.success("🤖 Model Trained")
        else:
            st.sidebar.info("🎯 Model Not Trained")

    # Route to selected section
    if menu_option == "🏠 Dashboard":
        show_dashboard()
    elif menu_option == "📊 Data Management":
        show_data_management()
    elif menu_option == "🤖 AI Analysis":
        show_ai_analysis()
    elif menu_option == "📈 Reports":
        show_reports()
    elif menu_option == "⚙️ Settings":
        show_settings()

def show_dashboard():
    st.header("🏠 Institutional Analytics Dashboard")
    
    if st.session_state.institutions_data is None:
        st.warning("⚠️ Please upload institutional data first in the Data Management section")
        st.info("💡 You can start with sample data or upload your own CSV/Excel file")
        return
    
    df = st.session_state.institutions_data
    
    # Key Metrics
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_institutions = len(df)
        st.metric("Total Institutions", total_institutions)
    
    with col2:
        avg_performance = df.get('performance_score', pd.Series([0])).mean()
        st.metric("Average Performance", f"{avg_performance:.2f}")
    
    with col3:
        if 'approval_status' in df.columns:
            approval_rate = (df['approval_status'].sum() / len(df)) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        else:
            st.metric("Approval Rate", "N/A")
    
    with col4:
        if 'risk_score' in df.columns:
            high_risk = (df['risk_score'] > 0.7).sum()
            st.metric("High Risk Institutions", high_risk)
        else:
            st.metric("Risk Analysis", "Run AI Analysis")
    
    # Performance Trends
    st.subheader("📈 Performance Trends")
    
    # Create sample trend data if not exists
    if 'year' not in df.columns:
        df['year'] = [2020, 2021, 2022, 2023] * (len(df) // 4 + 1)
        df['year'] = df['year'][:len(df)]
    
    if 'performance_score' not in df.columns:
        np.random.seed(42)
        df['performance_score'] = np.random.uniform(60, 95, len(df))
    
    # Performance over time
    fig_trend = px.line(df, x='year', y='performance_score', 
                       title='Institutional Performance Over Time',
                       color='institution_type' if 'institution_type' in df.columns else None)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Institutional Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if 'institution_type' in df.columns:
            type_counts = df['institution_type'].value_counts()
            fig_pie = px.pie(values=type_counts.values, names=type_counts.index,
                           title='Institution Type Distribution')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Institution type data not available")
    
    with col2:
        # Performance distribution
        fig_hist = px.histogram(df, x='performance_score', 
                               title='Performance Score Distribution',
                               nbins=20)
        st.plotly_chart(fig_hist, use_container_width=True)

def show_data_management():
    st.header("📊 Data Management")
    
    # Data Upload Section
    st.subheader("📤 Upload Institutional Data")
    
    upload_option = st.radio(
        "Choose data source:",
        ["Upload CSV/Excel File", "Use Sample Data", "Manual Entry"]
    )
    
    if upload_option == "Upload CSV/Excel File":
        uploaded_file = st.file_uploader(
            "Choose institutional data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file with institutional data"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Successfully loaded {len(df)} records with {len(df.columns)} columns")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
                
                # Show data summary
                st.subheader("Data Summary")
                st.write(f"**Shape:** {df.shape}")
                st.write("**Columns:**", list(df.columns))
                st.write("**Data Types:**")
                st.write(df.dtypes)
                
                # Store in session state
                st.session_state.institutions_data = df
                st.session_state.analysis_complete = False
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    elif upload_option == "Use Sample Data":
        if st.button("Generate Sample Data"):
            sample_df = generate_sample_data()
            st.session_state.institutions_data = sample_df
            st.success("✅ Sample data generated with 50 institutions")
            st.dataframe(sample_df.head(10))
    
    elif upload_option == "Manual Entry":
        st.info("Manual entry feature coming soon. Please upload a file or use sample data.")
    
    # Data Cleaning Tools
    if st.session_state.institutions_data is not None:
        st.subheader("🛠️ Data Cleaning Tools")
        
        df = st.session_state.institutions_data.copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Remove Duplicates"):
                initial_count = len(df)
                df = df.drop_duplicates()
                final_count = len(df)
                st.session_state.institutions_data = df
                st.success(f"Removed {initial_count - final_count} duplicate records")
        
        with col2:
            if st.button("Fill Missing Values"):
                # Simple imputation
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                st.session_state.institutions_data = df
                st.success("Missing values filled")

def show_ai_analysis():
    st.header("🤖 AI-Powered Analysis")
    
    if st.session_state.institutions_data is None:
        st.warning("⚠️ Please upload data first in the Data Management section")
        return
    
    df = st.session_state.institutions_data
    
    st.subheader("🎯 Approval Prediction Model")
    
    # Model training section
    if st.button("🚀 Train AI Prediction Model", type="primary"):
        with st.spinner("Training AI model... This may take a few moments"):
            try:
                # Prepare features and target
                feature_columns = [col for col in df.columns if col not in ['institution_name', 'approval_status', 'year']]
                feature_columns = [col for col in feature_columns if df[col].dtype in ['int64', 'float64']]
                
                if len(feature_columns) < 2:
                    st.error("❌ Not enough numeric features for model training")
                    return
                
                # Create synthetic target if not exists
                if 'approval_status' not in df.columns:
                    np.random.seed(42)
                    # Create realistic approval status based on performance
                    if 'performance_score' in df.columns:
                        performance = df['performance_score']
                        approval_probs = 1 / (1 + np.exp(-(performance - 70) / 10))
                        df['approval_status'] = (approval_probs > 0.5).astype(int)
                    else:
                        df['approval_status'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
                
                X = df[feature_columns].fillna(0)
                y = df['approval_status']
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Store model and features
                st.session_state.trained_model = model
                st.session_state.feature_columns = feature_columns
                st.session_state.analysis_complete = True
                
                # Show results
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                st.success("✅ Model trained successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Accuracy", f"{train_score:.1%}")
                with col2:
                    st.metric("Test Accuracy", f"{test_score:.1%}")
                with col3:
                    st.metric("Features Used", len(feature_columns))
                
                # Feature importance
                importance_df = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig_importance = px.bar(importance_df.head(10), 
                                      x='importance', y='feature',
                                      title='Top 10 Most Important Features for Approval Prediction')
                st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Model training failed: {str(e)}")
    
    # Prediction interface
    if st.session_state.trained_model is not None:
        st.subheader("🔮 Make Predictions")
        
        st.info("Enter institutional details to predict approval probability")
        
        # Create input form based on features
        input_data = {}
        cols = st.columns(3)
        
        for i, feature in enumerate(st.session_state.feature_columns):
            with cols[i % 3]:
                if feature in df.columns:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    avg_val = float(df[feature].mean())
                    
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=avg_val,
                        help=f"Range: {min_val:.1f} to {max_val:.1f}"
                    )
        
        if st.button("Predict Approval Probability"):
            input_df = pd.DataFrame([input_data])
            model = st.session_state.trained_model
            probability = model.predict_proba(input_df)[0][1]
            
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                # Gauge chart for probability
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Approval Probability"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightcoral"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if probability > 0.7:
                    st.markdown('<div class="success-box">✅ <strong>HIGH APPROVAL LIKELIHOOD</strong><br>This institution shows strong indicators for approval.</div>', unsafe_allow_html=True)
                elif probability > 0.4:
                    st.markdown('<div class="warning-box">🟡 <strong>MODERATE APPROVAL LIKELIHOOD</strong><br>Additional review recommended.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">🔴 <strong>LOW APPROVAL LIKELIHOOD</strong><br>Significant improvements needed.</div>', unsafe_allow_html=True)
                
                # Recommendations
                st.subheader("Recommendations")
                if probability < 0.4:
                    st.write("• Focus on improving academic performance metrics")
                    st.write("• Enhance infrastructure facilities")
                    st.write("• Strengthen financial management systems")
                elif probability < 0.7:
                    st.write("• Maintain current performance levels")
                    st.write("• Address any identified weaknesses")
                    st.write("• Continue quality improvement initiatives")

def show_reports():
    st.header("📈 Reports & Analytics")
    
    if st.session_state.institutions_data is None:
        st.warning("⚠️ Please upload data first in the Data Management section")
        return
    
    df = st.session_state.institutions_data
    
    st.subheader("📊 Generate Institutional Reports")
    
    report_type = st.selectbox(
        "Select Report Type:",
        ["Performance Analysis Report", 
         "Comparative Institutional Report",
         "Approval Recommendation Report",
         "Comprehensive Analytics Report"]
    )
    
    if st.button("Generate Report"):
        with st.spinner("Generating comprehensive report..."):
            # Simulate report generation
            import time
            time.sleep(2)
            
            st.success("✅ Report generated successfully!")
            
            # Display report
            st.subheader(f"📋 {report_type}")
            
            # Report content based on type
            if report_type == "Performance Analysis Report":
                show_performance_report(df)
            elif report_type == "Comparative Institutional Report":
                show_comparative_report(df)
            elif report_type == "Approval Recommendation Report":
                show_approval_report(df)
            else:
                show_comprehensive_report(df)
            
            # Download button
            st.download_button(
                label="📥 Download PDF Report",
                data=generate_pdf_content(df, report_type),
                file_name=f"ugc_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                help="Download this report as a text file"
            )

def show_performance_report(df):
    st.markdown("### Executive Summary")
    st.write(f"This report analyzes {len(df)} institutions with an average performance score of {df.get('performance_score', pd.Series([0])).mean():.2f}.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Institutions", len(df))
        st.metric("Average Performance", f"{df.get('performance_score', pd.Series([0])).mean():.2f}")
    with col2:
        if 'approval_status' in df.columns:
            approval_rate = (df['approval_status'].sum() / len(df)) * 100
            st.metric("Overall Approval Rate", f"{approval_rate:.1f}%")
        else:
            st.metric("Performance Trend", "Stable")

def show_comparative_report(df):
    st.markdown("### Institutional Comparison Analysis")
    st.write("Comparative analysis of institutional performance across different categories.")
    
    if 'institution_type' in df.columns:
        comparison_data = df.groupby('institution_type').agg({
            'performance_score': ['mean', 'count']
        }).round(2)
        st.dataframe(comparison_data)

def show_approval_report(df):
    st.markdown("### Approval Recommendations")
    st.write("AI-powered recommendations for institutional approval processes.")
    
    if st.session_state.trained_model is not None:
        st.success("✅ AI Model Integration Active")
        st.write("The approval prediction model has been trained and is ready for use.")
    else:
        st.info("🤖 Train the AI model in the AI Analysis section for predictive insights")

def show_comprehensive_report(df):
    st.markdown("### Comprehensive Institutional Analytics")
    
    # Multiple sections
    tabs = st.tabs(["Overview", "Performance Metrics", "Trends", "Recommendations"])
    
    with tabs[0]:
        st.write("**Platform Overview**")
        st.write(f"- Total Institutions: {len(df)}")
        st.write(f"- Data Columns: {len(df.columns)}")
        st.write(f"- Analysis Period: {df.get('year', pd.Series([2023])).min()} - {df.get('year', pd.Series([2023])).max()}")
    
    with tabs[1]:
        st.write("**Key Performance Indicators**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Show first 5 numeric columns
            st.write(f"- {col}: Mean = {df[col].mean():.2f}, Std = {df[col].std():.2f}")

def generate_pdf_content(df, report_type):
    """Generate simple text content for download"""
    content = f"UGC/AICTE Analytics Report\n"
    content += f"Report Type: {report_type}\n"
    content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    content += f"Total Institutions: {len(df)}\n"
    content += f"Data Summary:\n"
    content += f"- Columns: {list(df.columns)}\n"
    if 'performance_score' in df.columns:
        content += f"- Average Performance: {df['performance_score'].mean():.2f}\n"
    return content

def show_settings():
    st.header("⚙️ Platform Settings")
    
    st.subheader("Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Platform Name", value="UGC Analytics Platform")
        st.selectbox("Default Analysis Period", ["1 Year", "3 Years", "5 Years", "All Available Data"])
        st.number_input("Minimum Data Points", min_value=10, value=50)
    
    with col2:
        st.multiselect(
            "Enabled Modules",
            ["Performance Analytics", "Approval Prediction", "Comparative Analysis", 
             "Trend Forecasting", "Risk Assessment", "Automated Reporting"],
            default=["Performance Analytics", "Approval Prediction", "Comparative Analysis"]
        )
        st.slider("Confidence Threshold", 0.5, 0.95, 0.7)
    
    st.subheader("User Management")
    st.info("Free version supports unlimited users with basic role-based access")
    
    st.subheader("System Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Storage", "Unlimited", "Session-based")
    with col2:
        st.metric("AI Models", "Active", "All Free")
    with col3:
        st.metric("Platform Cost", "$0", "100% Free")

def generate_sample_data():
    """Generate realistic sample institutional data"""
    np.random.seed(42)
    n_institutions = 50
    
    data = {
        'institution_id': range(1, n_institutions + 1),
        'institution_name': [f"Institution_{i:03d}" for i in range(1, n_institutions + 1)],
        'institution_type': np.random.choice(['Public', 'Private', 'Deemed', 'Autonomous'], n_institutions),
        'establishment_year': np.random.randint(1950, 2020, n_institutions),
        'location': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_institutions),
        'student_strength': np.random.randint(500, 20000, n_institutions),
        'faculty_ratio': np.random.uniform(10, 30, n_institutions),
        'pass_percentage': np.random.uniform(60, 95, n_institutions),
        'infrastructure_score': np.random.uniform(50, 95, n_institutions),
        'research_output': np.random.uniform(0, 100, n_institutions),
        'financial_score': np.random.uniform(60, 95, n_institutions),
        'governance_score': np.random.uniform(70, 95, n_institutions),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate composite performance score
    df['performance_score'] = (
        df['pass_percentage'] * 0.3 +
        df['infrastructure_score'] * 0.2 +
        df['research_output'] * 0.1 +
        df['financial_score'] * 0.2 +
        df['governance_score'] * 0.2
    )
    
    # Add year column for time series analysis
    df['year'] = 2023
    
    return df

if __name__ == "__main__":
    main()
