# ADD THESE IMPORTS AT THE VERY TOP OF app.py
import sys
import os
try:
    from distutils import version
except ImportError:
    # distutils is deprecated in newer Python, use packaging instead
    from packaging import version

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Rest of your code continues...

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="UGC Analytics", layout="wide")

def main():
    st.title("🎓 UGC/AICTE Analytics Platform - LIVE!")
    st.success("✅ Platform is successfully deployed!")
    
    st.header("📊 Quick Start")
    
    # Sample data demo
    if st.button("Show Sample Data"):
        data = {
            'Institution': [f'Inst_{i}' for i in range(1, 6)],
            'Performance': [85, 72, 90, 68, 78],
            'Status': ['Approved', 'Pending', 'Approved', 'Rejected', 'Approved']
        }
        df = pd.DataFrame(data)
        st.dataframe(df)
        
        st.metric("Total Institutions", len(df))
        st.metric("Average Performance", f"{df['Performance'].mean():.1f}")

if __name__ == "__main__":
    main()
