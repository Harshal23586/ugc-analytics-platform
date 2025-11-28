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
            avg_performance = current_year_data['performance_score'].mean()
            st.metric("Avg Performance Score", f"{avg_performance:.2f}/10")
        
        with col4:
            approval_ready = (current_year_data['performance_score'] >= 6.0).sum()
            st.metric("Approval Ready", approval_ready)
            
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
    
    elif app_mode == "🔄 Approval Workflow":
        create_approval_workflow(analyzer)
    
    elif app_mode == "⚙️ System Settings":
        st.header("⚙️ System Settings & Configuration")
        st.info("System administration and configuration panel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            st.json(analyzer.performance_metrics)
        
        with col2:
            st.subheader("Document Requirements")
            st.json(analyzer.document_requirements)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d;'>
    <p><strong>UGC/AICTE Institutional Analytics Platform</strong> | AI-Powered Decision Support System</p>
    <p>Version 1.0 | For authorized use only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
