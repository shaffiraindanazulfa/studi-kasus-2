import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from streamlit_folium import folium_static

# Load data
file_path = r"C:\Users\Ideapad Slim 1\OneDrive\Documents\Semester 4\Data maning\dea panda numpy\covid_19_indonesia_time_series_all.csv"
df = pd.read_csv(file_path)

# Preprocessing
df = df.dropna(subset=['Latitude', 'Longitude'])
df['Case Fatality Rate'] = df['Case Fatality Rate'].str.replace('%', '').astype(float)
df['Population Density'] = np.random.randint(500, 15000, size=len(df))  # Simulasi

# Supervised Learning: Prediksi Total Cases
features = ['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']
X = df[features]
y = df['Total Cases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Clustering: KMeans
cluster_features = ['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[cluster_features])

# Sidebar
st.sidebar.title("COVID-19 Indonesia Dashboard")
menu = st.sidebar.radio("Pilih Halaman", ["Peta Cluster", "Tren Kasus Harian", "Prediksi Total Kasus"])

# Peta Cluster
if menu == "Peta Cluster":
    st.title("Peta Interaktif Hasil Clustering Wilayah")

    m = folium.Map(location=[-2.5489, 118.0149], zoom_start=5)

    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=f"{row['Location']} | Cluster: {row['Cluster']}",
            color=['red', 'blue', 'green', 'purple'][row['Cluster']],
            fill=True,
        ).add_to(m)
    
    folium_static(m)

# Tren Kasus Harian
elif menu == "Tren Kasus Harian":
    st.title("Grafik Tren Kasus Harian")

    harian = df.groupby('Date').sum().reset_index()

    fig = px.line(harian, x='Date', y='New Cases', title='Kasus Harian Indonesia')
    st.plotly_chart(fig)

# Prediksi Total Kasus
elif menu == "Prediksi Total Kasus":
    st.title("Prediksi Total Kasus Berdasarkan Fitur")

    st.write(f"Mean Absolute Error Model: **{mae:.2f}**")

    death = st.slider("Total Deaths", 0, int(df['Total Deaths'].max()), 10)
    recovered = st.slider("Total Recovered", 0, int(df['Total Recovered'].max()), 20)
    density = st.slider("Population Density", 500, 15000, 3000)
    cfr = st.slider("Case Fatality Rate (%)", 0.0, 50.0, 2.0)

    data_pred = pd.DataFrame([[death, recovered, density, cfr]], columns=features)
    pred_result = model.predict(data_pred)

    st.success(f"Prediksi Total Kasus: {int(pred_result[0])} kasus")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Dibuat oleh Shaffira")
