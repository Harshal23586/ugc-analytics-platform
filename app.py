import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="UGC Analytics", layout="wide")

def main():
    st.title("🎓 UGC/AICTE Analytics Platform")
    st.success("✅ Platform deployed successfully!")
    
    # Simple data management
    st.header("📊 Data Management")
    
    uploaded_file = st.file_uploader("Upload CSV/Excel file")
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"Loaded {len(df)} records")
        st.dataframe(df.head())
        
        # Basic analytics
        st.header("📈 Basic Analytics")
        st.write(f"**Total Institutions:** {len(df)}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("**Numerical Columns:**", list(numeric_cols))
            
            # Simple statistics
            for col in numeric_cols[:3]:  # Show first 3 numeric columns
                st.write(f"- {col}: Mean = {df[col].mean():.2f}")

if __name__ == "__main__":
    main()
