import streamlit as st
import pandas as pd
import numpy as np

# For plotting
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# For LDA topic modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# For sentiment analysis and emotion classification
from textblob import TextBlob
from transformers import pipeline

# Set page config
st.set_page_config(page_title="Analisis Topik, Sentimen, Emosi", layout="wide")

st.title("Analisis Topik, Sentimen, dan Emosi")
st.write("Aplikasi ini menggunakan LDA untuk analisis topik, TextBlob dan RoBERTa untuk analisis sentimen, serta BERT untuk analisis emosi (bahasa Inggris & Indonesia).")

# File uploader for data
uploaded_file = st.file_uploader("Unggah file data (data_stemm.csv)", type=["csv"])
if not uploaded_file:
    st.warning("Silakan unggah file `data_stemm.csv` untuk melanjutkan.")
    st.stop()

# Read data
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

st.write(f"Dataset berhasil diunggah. Data shape: {df.shape}")
st.write("Contoh data:")
st.write(df.head())

# Select text column
text_columns = df.select_dtypes(include=['object']).columns.tolist()
if not text_columns:
    st.error("Tidak ada kolom teks (tipe object) di dataset.")
    st.stop()
text_col = st.selectbox("Pilih kolom teks untuk analisis", text_columns, index=0)
if not text_col:
    st.error("Silakan pilih kolom teks.")
    st.stop()

# Ensure text column is string
df[text_col] = df[text_col].astype(str)

# Number of topics slider
num_topics = st.slider("Pilih jumlah topik untuk LDA", min_value=2, max_value=10, value=5, help="Jumlah topik untuk analisis LDA")
st.write(f"Memproses LDA dengan {num_topics} topik...")

# Compute LDA
@st.cache_data
def compute_lda(docs, n_topics):
    # Vectorize documents (Bag-of-words)
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(docs)
    # LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_dist = lda.fit_transform(dtm)
    # Get topic keywords
    topic_keywords = []
    words = vectorizer.get_feature_names_out()
    for topic_idx, comp in enumerate(lda.components_):
        # top 10 words for each topic
        word_idx = np.argsort(comp)[::-1][:10]
        topic_keywords.append([words[i] for i in word_idx])
    return topic_dist, topic_keywords

docs = df[text_col].tolist()
topic_dist, topic_keywords = compute_lda(docs, num_topics)

# Assign main topic for each document
df['Topic'] = np.argmax(topic_dist, axis=1)
df['Topic'] = df['Topic'].astype(int)
st.write("Topik utama setiap dokumen dihitung.")

# Visualization: jumlah dokumen per topik
topic_counts = df['Topic'].value_counts().sort_index()
topic_labels = [f"Topik {i}" for i in range(len(topic_counts))]
bar_df = pd.DataFrame({'Topik': topic_labels, 'Jumlah Dokumen': topic_counts.values})
fig_topic = px.bar(bar_df, x='Topik', y='Jumlah Dokumen', text='Jumlah Dokumen', title="Jumlah Dokumen per Topik")
fig_topic.update_layout(xaxis_title='Topik', yaxis_title='Jumlah Dokumen')
st.plotly_chart(fig_topic, use_container_width=True)

# Load models with caching to avoid reloading
@st.cache_resource
def load_sentiment_model():
    model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

@st.cache_resource
def load_emotion_models():
    # English emotion classifier
    en_model = "bhadresh-savani/bert-base-uncased-emotion"
    # Indonesian emotion classifier
    id_model = "thoriqfy/indobert-emotion-classification"
    nlp_en = pipeline("text-classification", model=en_model, return_all_scores=False)
    nlp_id = pipeline("text-classification", model=id_model, return_all_scores=False)
    return nlp_en, nlp_id

st.write("Analisis Sentimen dan Emosi sedang dijalankan...")

sentiment_nlp = load_sentiment_model()
emotion_nlp_en, emotion_nlp_id = load_emotion_models()

# Analyze sentiment
sentiments = sentiment_nlp(docs)
sentiment_labels = []
for i, sent in enumerate(sentiments):
    label = sent['label'].upper()
    if label == "NEGATIVE":
        sentiment_labels.append("Negatif")
    elif label == "POSITIVE":
        sentiment_labels.append("Positif")
    else:
        # Fallback: use TextBlob for neutral classification
        tb_score = TextBlob(docs[i]).sentiment.polarity
        if tb_score > 0:
            sentiment_labels.append("Positif")
        elif tb_score < 0:
            sentiment_labels.append("Negatif")
        else:
            sentiment_labels.append("Netral")

df['Sentiment'] = sentiment_labels

# Distribution of sentiment per topic
sentiment_counts = df.groupby(['Topic','Sentiment']).size().reset_index(name='Count')
fig_sent = px.bar(sentiment_counts, x='Topic', y='Count', color='Sentiment', 
                  text='Count', title="Distribusi Sentimen per Topik", barmode='stack')
fig_sent.update_layout(xaxis_title='Topik', yaxis_title='Jumlah Dokumen')
st.plotly_chart(fig_sent, use_container_width=True)

# Analyze emotion
emotion_labels = []
for doc in docs:
    # Predict emotions using both models
    res_en = emotion_nlp_en(doc)[0]
    res_id = emotion_nlp_id(doc)[0]
    # Select label with higher score
    if res_en['score'] >= res_id['score']:
        emotion_labels.append(res_en['label'])
    else:
        emotion_labels.append(res_id['label'])

df['Emotion'] = emotion_labels

# Distribution of emotion per topic
emotion_counts = df.groupby(['Topic','Emotion']).size().reset_index(name='Count')
fig_emot = px.bar(emotion_counts, x='Topic', y='Count', color='Emotion', 
                  text='Count', title="Distribusi Emosi per Topik", barmode='stack')
fig_emot.update_layout(xaxis_title='Topik', yaxis_title='Jumlah Dokumen')
st.plotly_chart(fig_emot, use_container_width=True)

# WordCloud per Topik
st.subheader("WordCloud Topik")
topic_option = st.selectbox("Pilih topik untuk WordCloud", sorted(df['Topic'].unique().tolist()))
topic_docs = df[df['Topic'] == topic_option][text_col]
if not topic_docs.empty:
    text_combined = " ".join(topic_docs)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("Tidak ada dokumen untuk topik ini.")

st.write("Selesai.")
