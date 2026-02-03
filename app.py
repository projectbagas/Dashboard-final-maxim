import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dashboard Kepuasan Pengguna Maxim",
    layout="wide"
)

st.title("📊 Dashboard Analisis Kepuasan Pengguna Aplikasi Maxim")

@st.cache_data
def load_data():
    df = pd.read_csv("maxim_siap_pakai.csv")

    def label_sentiment(r):
        if r >= 4:
            return "Puas"
        elif r == 3:
            return "Netral"
        else:
            return "Tidak Puas"

    df["sentimen"] = df["rating"].apply(label_sentiment)
    return df

data = load_data()

menu = st.sidebar.selectbox(
    "Menu",
    ["Dataset", "Distribusi Rating", "Distribusi Sentimen", "Ringkasan"]
)

if menu == "Dataset":
    st.subheader("Dataset Ulasan")
    st.dataframe(data.head(20))

elif menu == "Distribusi Rating":
    st.subheader("Distribusi Rating")
    fig, ax = plt.subplots()
    data["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
    st.pyplot(fig)

elif menu == "Distribusi Sentimen":
    st.subheader("Distribusi Sentimen")
    fig, ax = plt.subplots()
    data["sentimen"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    st.pyplot(fig)

elif menu == "Ringkasan":
    st.subheader("Ringkasan Analisis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Puas", (data["sentimen"] == "Puas").sum())
    col2.metric("Netral", (data["sentimen"] == "Netral").sum())
    col3.metric("Tidak Puas", (data["sentimen"] == "Tidak Puas").sum())

