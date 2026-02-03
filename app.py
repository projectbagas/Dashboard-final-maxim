import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Skripsi Bagas - Maxim Analisis", layout="wide")

# 2. FUNGSI LOAD ASSETS (Model & Data)
@st.cache_resource
def load_assets():
    # Load dataset
    df = pd.read_csv('maxim_reviews.csv')
    # Pelabelan untuk visualisasi dashboard
    df['label'] = df['score'].apply(lambda x: 'Puas' if x >= 4 else ('Netral' if x == 3 else 'Tidak Puas'))
    
    # Load Model menggunakan joblib (Lebih stabil untuk Cloud)
    # Pastikan nama file ini sesuai dengan yang ada di GitHub kamu
    model_xgb = joblib.load('model_xgb.pkl')
    model_rf = joblib.load('model_rf.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    
    return df, model_xgb, model_rf, tfidf

# Menjalankan fungsi load_assets
try:
    df, xgb, rf, tfidf = load_assets()
except Exception as e:
    st.error(f"Gagal memuat file: {e}")
    st.info("Pastikan file maxim_reviews.csv, model_xgb.pkl, model_rf.pkl, dan tfidf_vectorizer.pkl sudah ada di GitHub.")
    st.stop()

# 3. SIDEBAR NAVIGASI
with st.sidebar:
    st.title("Menu Utama")
    st.markdown("---")
    menu = st.radio("Pilih Halaman:", 
                    ["📊 Dashboard", "📂 Dataset", "🧠 Model Klasifikasi", "⚖️ Implementasi Algoritma"])
    st.markdown("---")
    st.write("**Oleh:** Bagas Dwi Ardianto")
    st.write("**NIM:** 217006516109")

# --- HALAMAN 1: DASHBOARD ---
if menu == "📊 Dashboard":
    st.title("📈 Visualisasi Data Maxim")
    st.write("Halaman ini menampilkan gambaran umum sebaran data ulasan pengguna.")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Distribusi Sentimen")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x='label', data=df, palette='viridis', ax=ax)
        plt.xlabel("Kategori")
        plt.ylabel("Jumlah Ulasan")
        st.pyplot(fig)
    
    with c2:
        st.subheader("Statistik Dataset")
        st.metric("Total Data", f"{len(df)} Ulasan")
        st.write("Data diambil dari: *Google Play Store*")
        st.write("Kategori: *Multi-class (3 Label)*")

# --- HALAMAN 2: DATASET ---
elif menu == "📂 Dataset":
    st.title("📂 Dataset Mentah")
    st.write("Menampilkan data ulasan aplikasi Maxim yang digunakan dalam penelitian.")
    st.dataframe(df[['userName', 'score', 'content', 'label']], use_container_width=True)

# --- HALAMAN 3: MODEL KLASIFIKASI ---
elif menu == "🧠 Model Klasifikasi":
    st.title("🧠 Penjelasan Algoritma")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("### XGBoost")
        st.write("""
        **Extreme Gradient Boosting** adalah algoritma yang memperbaiki kesalahan 
        prediksi dari pohon keputusan sebelumnya secara berurutan. 
        Sangat efektif untuk data teks yang kompleks.
        """)
    
    with col_b:
        st.warning("### Random Forest")
        st.write("""
        **Random Forest** membangun banyak pohon keputusan secara acak dan mandiri, 
        lalu mengambil hasil keputusan terbanyak (voting). 
        Stabil namun terkadang kalah akurasi dibanding boosting.
        """)

# --- HALAMAN 4: IMPLEMENTASI ---
elif menu == "⚖️ Implementasi Algoritma":
    st.title("⚖️ Performa & Uji Coba Model")
    
    # Menampilkan Tabel Metrik sesuai Skripsi
    st.subheader("Metrik Evaluasi (Hasil Pengujian)")
    metrics_data = {
        'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score'],
        'XGBoost': ["0.93", "0.88", "0.93", "0.90"],
        'Random Forest': ["0.80", "0.73", "0.80", "0.77"]
    }
    st.table(pd.DataFrame(metrics_data))

    st.divider()

    # Fitur Live Testing
    st.subheader("🔍 Live Testing (Uji Coba Ulasan)")
    user_text = st.text_area("Masukkan teks ulasan pelanggan:")
    model_choice = st.selectbox("Pilih Model untuk Prediksi:", ["XGBoost", "Random Forest"])
    
    if st.button("Analisis Sentimen"):
        if user_text:
            # 1. Preprocessing (Cleaning)
            cleaned = re.sub(r'[^a-z\s]', '', user_text.lower())
            # 2. Transform ke TF-IDF
            vec = tfidf.transform([cleaned])
            
            # 3. Prediksi
            if model_choice == "XGBoost":
                res = xgb.predict(vec)[0]
            else:
                res = rf.predict(vec)[0]
            
            # 4. Mapping Label (0: Tidak Puas, 1: Netral, 2: Puas)
            # Pastikan urutan ini sesuai dengan LabelEncoder saat training
            labels = {0: "Tidak Puas ❌", 1: "Netral 😐", 2: "Puas ✅"}
            
            st.markdown(f"### Hasil Prediksi: **{labels[res]}**")
        else:
            st.warning("Mohon masukkan teks terlebih dahulu!")
