import streamlit as st
import nltk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
from io import StringIO

# Téléchargement des ressources nltk
nltk.download('punkt')
nltk.download('stopwords')

# Chargement BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# === FONCTIONS ===

# Prétraitement des textes : tokenisation, suppression des stop-words, stemming
def preprocess(text, lang):
    text = text.lower()
    tokens = nltk.word_tokenize(re.sub(r'\W+', ' ', text), language=lang)
    stop_words = set(stopwords.words(lang))
    stemmer = SnowballStemmer(lang) if lang != "english" else PorterStemmer()
    filtered = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(filtered), filtered

# TF-IDF
def tfidf_vectorize(corpus):
    return TfidfVectorizer().fit_transform(corpus)

# Word2Vec
def train_word2vec(tokenized):
    return Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1)

# Calcul de l'average vector pour Word2Vec
def get_avg_vector(tokens, model):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Encodage avec BERT
def bert_vectorize(corpus):
    input_ids = tokenizer(corpus, padding=True, truncation=True, return_tensors='pt', max_length=512)['input_ids']
    with torch.no_grad():
        outputs = bert_model(input_ids)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Calcul des similarités
def cosine_similarity_matrix(vectors):
    return cosine_similarity(vectors)

# Ranking des documents similaires
def rank_documents(sim_matrix, index, docs):
    scores = list(enumerate(sim_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [{"Document": docs[i], "Score": round(score, 3)} for i, score in scores if i != index]

# Recherche par regex
def search_with_regex(documents, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    return [doc for doc in documents if regex.search(doc)]

# Fonction pour afficher les graphiques
def plot_comparisons(tfidf_result, w2v_result, bert_result):
    labels = [f'Doc {i+1}' for i in range(len(tfidf_result))]
    tfidf_scores = [item['Score'] for item in tfidf_result]
    w2v_scores = [item['Score'] for item in w2v_result]
    bert_scores = [item['Score'] for item in bert_result]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, tfidf_scores, width, label='TF-IDF')
    ax.bar(x, w2v_scores, width, label='Word2Vec')
    ax.bar(x + width, bert_scores, width, label='BERT')

    ax.set_ylabel('Score de Similarité')
    ax.set_title('Comparaison des Méthodes')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    st.pyplot(fig)

# Export CSV
def export_to_csv(name, data):
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

# === STREAMLIT ===

st.set_page_config(page_title="Analyseur de similarité de textes", layout="wide")
st.title("🔎 Analyseur de Similarité de Textes")
st.markdown("Ce programme compare des textes selon 3 méthodes : **TF-IDF**, **Word2Vec**, **BERT**.")

col1, col2 = st.columns([2, 1])

with col1:
    documents_text = st.text_area("📄 Colle tes documents (un par ligne)", height=200, value=
"""Machine learning is a subfield of artificial intelligence.
Deep learning is part of machine learning.
Natural language processing is used in AI.
How to cook pasta with tomato sauce.
Artificial intelligence is everywhere."""
    )

with col2:
    lang = st.selectbox("🌍 Langue des textes", options=["english", "french"], index=0)

documents = [doc.strip() for doc in documents_text.splitlines() if doc.strip()]

if len(documents) < 2:
    st.warning("🟡 Ajoute au moins deux documents.")
    st.stop()

doc_ref = st.selectbox("🎯 Choisis le document de référence :", options=range(len(documents)), format_func=lambda i: f"{i+1}. {documents[i][:40]}...")

# Prétraitement
preprocessed, tokens_list = [], []
for doc in documents:
    clean, tokens = preprocess(doc, lang)
    preprocessed.append(clean)
    tokens_list.append(tokens)

# Lancer l'analyse
if st.button("🚀 Lancer l’analyse"):
    with st.spinner("Calcul des similarités..."):

        # TF-IDF
        tfidf_vecs = tfidf_vectorize(preprocessed)
        tfidf_sim = cosine_similarity(tfidf_vecs)
        tfidf_result = rank_documents(tfidf_sim, doc_ref, documents)

        # Word2Vec
        w2v_model = train_word2vec(tokens_list)
        w2v_vecs = np.array([get_avg_vector(tokens, w2v_model) for tokens in tokens_list])
        w2v_sim = cosine_similarity(w2v_vecs)
        w2v_result = rank_documents(w2v_sim, doc_ref, documents)

        # BERT
        bert_vecs = bert_vectorize(documents)
        bert_sim = cosine_similarity(bert_vecs)
        bert_result = rank_documents(bert_sim, doc_ref, documents)

    st.success("✅ Analyse terminée.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📊 TF-IDF")
        st.dataframe(pd.DataFrame(tfidf_result))
        st.download_button("⬇️ Télécharger TF-IDF", data=export_to_csv("tfidf", tfidf_result), file_name="tfidf_result.csv")

    with col2:
        st.subheader("📊 Word2Vec")
        st.dataframe(pd.DataFrame(w2v_result))
        st.download_button("⬇️ Télécharger Word2Vec", data=export_to_csv("w2v", w2v_result), file_name="word2vec_result.csv")

    with col3:
        st.subheader("📊 BERT")
        st.dataframe(pd.DataFrame(bert_result))
        st.download_button("⬇️ Télécharger BERT", data=export_to_csv("bert", bert_result), file_name="bert_result.csv")

    st.markdown("---")
    st.subheader("📈 Graphique Comparatif")
    plot_comparisons(tfidf_result, w2v_result, bert_result)

    # Analyse rapide
    st.markdown("### 📌 Analyse comparative")
    st.write(f"**TF-IDF moyenne :** {np.mean([d['Score'] for d in tfidf_result]):.2f}")
    st.write(f"**Word2Vec moyenne :** {np.mean([d['Score'] for d in w2v_result]):.2f}")
    st.write(f"**BERT moyenne :** {np.mean([d['Score'] for d in bert_result]):.2f}")

# Recherche Regex
st.markdown("---")
st.subheader("🔍 Recherche par expression régulière")
regex_input = st.text_input("Tape une regex (ex: `ai|machine`)")

if regex_input:
    results = search_with_regex(documents, regex_input)
    if results:
        st.success(f"{len(results)} document(s) trouvé(s) :")
        for r in results:
            st.markdown(f"- ✅ {r}")
    else:
        st.warning("Aucun résultat trouvé.")

