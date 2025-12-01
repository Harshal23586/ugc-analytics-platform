# Add to imports at the top of your main file:
from report_generator import PDFReportGenerator

# Add to InstitutionalAIAnalyzer __init__:
class InstitutionalAIAnalyzer:
    def __init__(self):
        self.init_database()
        self.historical_data = self.load_or_generate_data()
        self.performance_metrics = self.define_performance_metrics()
        self.document_requirements = self.define_document_requirements()
        self.report_generator = PDFReportGenerator(self)  # Add this line
        # ... rest of init

# Create a new module for PDF report generation in your main app:
def create_pdf_report_module(analyzer):
    st.header("📄 PDF Report Generation")
    
    st.info("Generate professional PDF reports for institutional assessments and approvals")
    
    # Institution selection
    current_institutions = analyzer.historical_data[analyzer.historical_data['year'] == 2023]
    institution_options = {}
    
    for _, row in current_institutions.iterrows():
        institution_options[f"{row['institution_name']} ({row['institution_id']})"] = row['institution_id']
    
    selected_institution_display = st.selectbox(
        "Select Institution",
        list(institution_options.keys()),
        key="pdf_institution"
    )
    
    selected_institution_id = institution_options[selected_institution_display]
    
    # Report type selection
    report_type = st.radio(
        "Select Report Type",
        ["📋 Comprehensive Report", 
         "🎯 Executive Summary", 
         "📊 Detailed Analytical Report",
         "🏛️ Official Approval Report"],
        horizontal=True
    )
    
    # Map display names to internal types
    report_type_map = {
        "📋 Comprehensive Report": "comprehensive",
        "🎯 Executive Summary": "executive",
        "📊 Detailed Analytical Report": "detailed",
        "🏛️ Official Approval Report": "approval"
    }
    
    selected_type = report_type_map[report_type]
    
    # Show preview of selected institution
    with st.expander("👁️ Institution Preview"):
        institution_data = current_institutions[
            current_institutions['institution_id'] == selected_institution_id
        ].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Performance Score", f"{institution_data['performance_score']:.2f}")
            st.metric("Risk Level", institution_data['risk_level'])
        
        with col2:
            st.metric("NAAC Grade", institution_data.get('naac_grade', 'N/A'))
            st.metric("Placement Rate", f"{institution_data.get('placement_rate', 0):.1f}%")
        
        with col3:
            st.metric("Approval Status", institution_data['approval_recommendation'])
            st.metric("Type", institution_data['institution_type'])
    
    # Customization options
    with st.expander("⚙️ Report Customization"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_charts = st.checkbox("Include Charts & Graphs", value=True)
            include_benchmarks = st.checkbox("Include Benchmark Comparisons", value=True)
            confidential = st.checkbox("Confidential Report", value=False)
        
        with col2:
            watermark = st.checkbox("Add UGC/AICTE Watermark", value=True)
            executive_summary = st.checkbox("Add Executive Summary", value=True)
            detailed_appendix = st.checkbox("Include Detailed Appendix", value=False)
        
        # Additional notes
        report_notes = st.text_area(
            "Additional Notes/Comments for Report",
            placeholder="Add any specific comments or observations to include in the report...",
            height=100
        )
    
    # Generate report
    if st.button("🖨️ Generate PDF Report", type="primary"):
        with st.spinner(f"Generating {report_type}..."):
            try:
                # Generate the report
                pdf_path = analyzer.report_generator.generate_institutional_report(
                    selected_institution_id,
                    selected_type
                )
                
                # Read the generated PDF
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                # Provide download button
                st.success("✅ Report generated successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf"
                    )
                
                with col2:
                    # Preview option
                    if st.button("👁️ Preview Report"):
                        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                
                with col3:
                    # Email option (future enhancement)
                    if st.button("📧 Email Report"):
                        st.info("Email functionality would be integrated here")
                
                # Report details
                st.info(f"**Report Details:**")
                st.write(f"- **File:** {os.path.basename(pdf_path)}")
                st.write(f"- **Size:** {os.path.getsize(pdf_path) / 1024:.1f} KB")
                st.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"- **Type:** {report_type}")
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
    
    # Batch report generation
    st.markdown("---")
    st.subheader("🔄 Batch Report Generation")
    
    st.info("Generate reports for multiple institutions at once")
    
    selected_institutions = st.multiselect(
        "Select Institutions for Batch Processing",
        list(institution_options.keys()),
        default=[]
    )
    
    batch_report_type = st.selectbox(
        "Report Type for Batch",
        ["Executive Summary", "Comprehensive Report"],
        key="batch_type"
    )
    
    if st.button("🖨️ Generate Batch Reports", type="secondary"):
        if not selected_institutions:
            st.warning("Please select at least one institution")
        else:
            with st.spinner(f"Generating reports for {len(selected_institutions)} institutions..."):
                progress_bar = st.progress(0)
                generated_reports = []
                
                for i, inst_display in enumerate(selected_institutions):
                    inst_id = institution_options[inst_display]
                    try:
                        pdf_path = analyzer.report_generator.generate_institutional_report(
                            inst_id,
                            "executive" if batch_report_type == "Executive Summary" else "comprehensive"
                        )
                        generated_reports.append((inst_display, pdf_path))
                    except Exception as e:
                        st.warning(f"Failed to generate report for {inst_display}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(selected_institutions))
                
                # Create zip file of all reports
                if generated_reports:
                    import zipfile
                    zip_buffer = BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for inst_display, pdf_path in generated_reports:
                            zip_file.write(pdf_path, os.path.basename(pdf_path))
                    
                    zip_buffer.seek(0)
                    
                    st.success(f"✅ Generated {len(generated_reports)} reports successfully!")
                    
                    st.download_button(
                        label="📦 Download All Reports (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"institutional_reports_{datetime.now().strftime('%Y%m%d')}.zip",
                        mime="application/zip"
                    )
    
    # Report templates
    st.markdown("---")
    st.subheader("📋 Report Templates")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download Report Template"):
            st.info("Template would be downloaded here")
    
    with col2:
        if st.button("🔄 Reset to Default Template"):
            st.success("Template reset to defaults")
    
    with col3:
        if st.button("💾 Save Custom Template"):
            st.success("Custom template saved")
