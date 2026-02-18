import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from database import get_all_entries
import os


st.set_page_config(page_title="Parking Analytics", layout="wide")

st.title(" Smart Parking Analytics Dashboard")
st.markdown("Real-time monitoring and statistics from the Gate Camera")


df = get_all_entries()

if not df.empty:
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    
    col1, col2, col3 = st.columns(3)
    
    total_entries = len(df)
    unique_cars = df['plate_number'].nunique()
    unauthorized = len(df[df['is_authorized'] == 0])

    col1.metric("Total Entries", total_entries)
    col2.metric("Unique Vehicles", unique_cars)
    col3.metric("Unauthorized Alerts", unauthorized, delta_color="inverse")

    st.markdown("---")

    
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Activity by Hour")
        
        hourly_counts = df['timestamp'].dt.hour.value_counts().sort_index()
        st.bar_chart(hourly_counts)

    with c2:
        st.subheader("Authorized vs Guest")
        auth_counts = df['is_authorized'].value_counts()
        
        auth_counts.index = ['Guest' if x==0 else 'Authorized' for x in auth_counts.index]
        
        fig, ax = plt.subplots()
        ax.pie(auth_counts, labels=auth_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
        st.pyplot(fig)

    
    st.markdown("### Recent Logs")
    
    
    for index, row in df.head(5).iterrows():
        c_img, c_info = st.columns([1, 4])
        
        with c_img:
            if os.path.exists(row['image_path']):
                st.image(row['image_path'], width=150)
            else:
                st.write("No Image")
        
        with c_info:
            status = " Authorized" if row['is_authorized'] else " Unauthorized"
            st.write(f"**Plate:** {row['plate_number']} | **Time:** {row['timestamp']} | **Status:** {status}")
            st.progress(int(row['confidence']))

else:
    st.info("No data available yet. Start detecting cars!")