import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import confusion_matrix

# Konfigurasi Halaman
st.set_page_config(page_title="Skripsi Bagas - Maxim", layout="wide")

# Fungsi Load Assets
@st.cache_resource
def load_assets():
    df = pd.read_csv('maxim_reviews.csv')
    df['label'] = df['score'].apply(lambda x: 'Puas' if x >= 4 else ('Netral' if x == 3 else 'Tidak Puas'))
    
    # Load Model (Pastikan file ini ada di GitHub Anda)
    model_xgb = pickle.load(open('model_xgb.pkl', 'rb'))
    model_rf = pickle.load(open('model_rf.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    return df, model_xgb, model_rf, tfidf

# Menangani error jika file pkl belum ada
try:
    df, xgb, rf, tfidf = load_assets()
except FileNotFoundError:
    st.error("File Model (.pkl) tidak ditemukan di GitHub! Pastikan Anda sudah menguploadnya.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Menu Utama")
    menu = st.radio("Pilih Halaman:", ["Dashboard", "Dataset", "Model Klasifikasi", "Implementasi Algoritma"])

# --- MENU 1: DASHBOARD ---
if menu == "Dashboard":
    st.title("📈 Visualisasi Data Maxim")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribusi Sentimen")
        fig, ax = plt.subplots()
        sns.countplot(x='label', data=df, palette='magma', ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Statistik Dataset")
        st.write(f"Total Data: **{len(df)}** baris")
        st.write("Target: **Tingkat Kepuasan**")
        st.write("Fitur: **Ulasan Teks**")

# --- MENU 2: DATASET ---
elif menu == "Dataset":
    st.title("📂 Dataset Mentah")
    st.dataframe(df[['userName', 'score', 'content', 'label']], use_container_width=True)

# --- MENU 3: MODEL KLASIFIKASI ---
elif menu == "Model Klasifikasi":
    st.title("🧠 Penjelasan Algoritma")
    st.info("**XGBoost:** Algoritma Gradient Boosting yang efisien. Menggunakan prinsip perbaikan kesalahan secara berurutan.")
    st.warning("**Random Forest:** Algoritma Ensemble berbasis Bagging. Membangun banyak pohon keputusan dan mengambil voting terbanyak.")

# --- MENU 4: IMPLEMENTASI ---
elif menu == "Implementasi Algoritma":
    st.title("⚖️ Performa & Uji Coba")
    
    # 1. Tabel Metrik (Input dari hasil skripsi Anda)
    st.subheader("Metrik Evaluasi")
    metrics_data = {
        'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score'],
        'XGBoost': [0.93, 0.88, 0.93, 0.90],
        'Random Forest': [0.80, 0.73, 0.80, 0.77]
    }
    st.table(pd.DataFrame(metrics_data))

    # 2. Live Testing
    st.divider()
    st.subheader("🔍 Uji Coba Prediksi")
    teks_input = st.text_area("Masukkan ulasan untuk diuji:")
    pilih_model = st.selectbox("Pilih Algoritma", ["XGBoost", "Random Forest"])
    
    if st.button("Analisis"):
        if teks_input:
            # Preprocessing
            clean_text = re.sub(r'[^a-z\s]', '', teks_input.lower())
            vectorized_text = tfidf.transform([clean_text])
            
            # Predict
            if pilih_model == "XGBoost":
                pred = xgb.predict(vectorized_text)[0]
            else:
                pred = rf.predict(vectorized_text)[0]
            
            mapping = {0: "Tidak Puas ❌", 1: "Netral 😐", 2: "Puas ✅"}
            st.success(f"Hasil Prediksi ({pilih_model}): **{mapping[pred]}**")
