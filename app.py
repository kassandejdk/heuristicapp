import streamlit as st
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from collections import Counter
import plotly.express as px
import os
import spacy
import gensim.downloader as api
from io import StringIO


# Configuration de l'application
st.set_page_config(page_title="Analyseur de similarité de textes", layout="wide")
st.title("Analyseur de similarité de textes")
st.markdown("""
Cette application permet d'analyser la similarité entre des documents textuels en utilisant différentes méthodes.
""")

# Télécharger les ressources NLTK nécessaires
def init_nltk():
    import os

    nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
    
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)

    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        st.error("Erreur : les données NLTK sont absentes. Veuillez vérifier le dossier 'nltk_data'.")

init_nltk()  

# Initialisation des modèles (chargés en lazy loading)
word2vec_model = None
bert_model = None

# Fonctions de prétraitement
def preprocess_text(text, remove_stopwords=True, apply_stemming=True, apply_lemmatization=True):
    """Prétraite un texte avec tokenisation, suppression stopwords et stemming"""
    # Mise en minuscule
    text = text.lower()
    
    # Suppression de la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des stop-words
    if remove_stopwords:
        stop_words = set(stopwords.words('french') + list(stopwords.words('english')))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    if apply_stemming:
        stemmer = SnowballStemmer("french")
        tokens = [stemmer.stem(word) for word in tokens]
        
    # Lemmatisation
    elif apply_lemmatization:
        nlp = spacy.load("fr_core_news_sm")
        doc = nlp(" ".join(tokens)) 
        tokens = [token.lemma_ for token in doc]
        
    return tokens

def apply_regex_filter(text, pattern):
    """Applique un filtre d'expression régulière sur le texte"""
    if not pattern:
        return text
    matches = re.findall(pattern, text, re.IGNORECASE)
    return ' '.join(matches)


# Fonctions de similarité
def load_word2vec_model():
    """Charge automatiquement et sauvegarde localement le modèle GloVe."""
    global word2vec_model
    model_local_path = "glove-wiki-gigaword-50.model"  

    if word2vec_model is None:
        try:
            if os.path.exists(model_local_path):
                with st.spinner("Chargement du modèle depuis le disque..."):
                    word2vec_model = KeyedVectors.load(model_local_path, mmap='r')
            else:
                with st.spinner("Téléchargement du modèle pré-entraîné..."):
                    model_name = "glove-wiki-gigaword-50"
                    word2vec_model = api.load(model_name)
                    # Sauvegarder localement
                    word2vec_model.save(model_local_path)
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")
            return None
    return word2vec_model


def load_bert_model():
    """Charge le modèle BERT"""
    global bert_model
    if bert_model is None:
        try:
            bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        except:
            st.error("Erreur lors du chargement du modèle BERT.")
            return None
    return bert_model

def calculate_tfidf_cosine_similarity(documents, ref_index):
    """Calcule la similarité cosinus basée sur TF-IDF"""
    texts = [' '.join(doc['processed']) for doc in documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[ref_index], tfidf_matrix).flatten()
    
    results = []
    for i, sim in enumerate(cosine_similarities):
        results.append({
            "Document": documents[i]['name'],
            "Similarité": sim,
            "Méthode": "TF-IDF"
        })
    return results

def calculate_word2vec_similarity(documents, ref_index):
    """Calcule la similarité basée sur Word2Vec"""
    model = load_word2vec_model()
    if model is None:
        return []
    
    def get_doc_vector(tokens):
        vectors = []
        for token in tokens:
            if token in model:
                vectors.append(model[token])
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
    
    doc_vectors = [get_doc_vector(doc['processed']) for doc in documents]
    ref_vector = doc_vectors[ref_index].reshape(1, -1)
    other_vectors = np.array(doc_vectors)
    cosine_similarities = cosine_similarity(ref_vector, other_vectors).flatten()
    
    results = []
    for i, sim in enumerate(cosine_similarities):
        results.append({
            "Document": documents[i]['name'],
            "Similarité": sim,
            "Méthode": "Word2Vec"
        })
    return results

def calculate_bert_similarity(documents, ref_index):
    """Calcule la similarité basée sur BERT"""
    model = load_bert_model()
    if model is None:
        return []
    
    texts = [doc['text'] for doc in documents]
    embeddings = model.encode(texts)
    cosine_similarities = cosine_similarity(
        embeddings[ref_index].reshape(1, -1), 
        embeddings
    ).flatten()
    
    results = []
    for i, sim in enumerate(cosine_similarities):
        results.append({
            "Document": documents[i]['name'],
            "Similarité": sim,
            "Méthode": "BERT"
        })
    return results

def calculate_keyword_overlap(documents, ref_index):
    """Calcule la similarité basée sur le chevauchement des mots-clés"""
    word_counts = [Counter(doc['processed']) for doc in documents]
    ref_words = set(word_counts[ref_index].keys())
    
    results = []
    for i, counter in enumerate(word_counts):
        doc_words = set(counter.keys())
        overlap = len(ref_words & doc_words)
        total_unique = len(ref_words | doc_words)
        similarity = overlap / total_unique if total_unique > 0 else 0
        
        results.append({
            "Document": documents[i]['name'],
            "Similarité": similarity,
            "Méthode": "Keywords"
        })
    return results

def calculate_similarity(documents, ref_index, method="TF-IDF", regex_pattern=None):
    """Fonction principale pour calculer la similarité"""
    if regex_pattern:
        filtered_docs = []
        for doc in documents:
            filtered_text = apply_regex_filter(doc['text'], regex_pattern)
            if filtered_text.strip(): 
                filtered_docs.append({
                    **doc,
                    "text": filtered_text,
                    "processed": preprocess_text(filtered_text)
                })
        documents = filtered_docs
        if not documents:
            st.warning("Aucun document ne correspond au motif regex que vous avez saisi. Veuillez essayer un autre motif.")
            return []
        
        print(documents)
    if method == "TF-IDF + Cosine":
        return calculate_tfidf_cosine_similarity(documents, ref_index)
    elif method == "Word2Vec":
        return calculate_word2vec_similarity(documents, ref_index)
    elif method == "BERT":
        return calculate_bert_similarity(documents, ref_index)
    elif method == "Mots-clés fréquents":
        return calculate_keyword_overlap(documents, ref_index)
    else:
        st.error(f"Méthode inconnue: {method}")
        return []

def compare_methods(documents, ref_index=0):
    if not documents or len(documents) < 2:
        return pd.DataFrame(columns=["method", "document", "score", "error"])
    
    results = []
    methods = {
        "TF-IDF": calculate_tfidf_cosine_similarity,
        "Word2Vec": calculate_word2vec_similarity,
        "BERT": calculate_bert_similarity,
        "Keywords": calculate_keyword_overlap
    }
    
    for name, func in methods.items():
        try:
            method_results = func(documents, ref_index)
            print("Pour chaque methode", method_results)
            for res in method_results:
                if res['Document'] != documents[0]['name']:
                    results.append({
                        "method": name,
                        "document": res.get("Document", "N/A"), 
                        "score": res.get("Similarité", 0.0),
                        "error": ""
                    })
        except Exception as e:
            results.append({
                "method": name,
                "document": "N/A",
                "score": 0.0,
                "error": str(e)
            })
    
    return pd.DataFrame(results)

# Fonctions de visualisation
def plot_similarity_results(results_df, method):
    """Crée un graphique des résultats de similarité"""

    df = pd.DataFrame(results_df)

    # Adapter au bon nom de colonne
    similarity_col = None
    if 'Similarité' in df.columns:
        similarity_col = 'Similarité'
    elif 'score' in df.columns:
        similarity_col = 'score'
    else:
        st.error("Impossible de trouver la colonne de similarité dans les résultats.")
        return None

    df = df.sort_values(similarity_col, ascending=False)

    fig = px.bar(
        df,
        x='document' if 'document' in df.columns else 'Document',
        y=similarity_col,
        color=similarity_col,
        title=f"Similarité selon {method}",
        labels={similarity_col: "Score de Similarité", 'document': "Document"}
    )

    fig.update_layout(xaxis_title="Document", yaxis_title="Similarité")
    return fig

# Interface de streamlit

# Sidebar pour les paramètres
with st.sidebar:
    st.header("Paramètres")
    
    method = st.selectbox(
        "Méthode de similarité",
        ["TF-IDF + Cosine", "Word2Vec", "BERT", "Mots-clés fréquents"],
        help="Choisissez la méthode pour calculer la similarité entre les textes"
    )
    
    st.subheader("Options de prétraitement")
    remove_stopwords = st.checkbox("Supprimer les stop-words", value=True)
    apply_stemming = st.checkbox("Appliquer le stemming", value=True)
    apply_lemmatization = st.checkbox("Appliquer la lemmatisation", value=True)
    
    st.subheader("Recherche par expression régulière")
    regex_pattern = st.text_input("Motif regex (optionnel)", "")
    
    st.subheader("Charger des documents")
    uploaded_files = st.file_uploader(
        "Ajoutez des fichiers texte", 
        type=["txt", "pdf", "docx"], 
        accept_multiple_files=True
    )

tab1, tab2, tab3 = st.tabs(["Analyse", "Comparaison", "À propos"])

with tab1:
    st.header("Analyse de similarité")
    
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            text = uploaded_file.read().decode("utf-8")
            documents.append({
                "name": uploaded_file.name,
                "text": text,
                "processed": preprocess_text(text, remove_stopwords, apply_stemming, apply_lemmatization)
            })
        
        doc_names = [doc["name"] for doc in documents]
        ref_doc = st.selectbox("Document de référence", doc_names)
        
        if st.button("Calculer la similarité"):
            with st.spinner("Calcul en cours..."):
                ref_index = doc_names.index(ref_doc)
                results = calculate_similarity(documents, ref_index, method, regex_pattern)
                
                st.subheader("Résultats de similarité")
                if results:
                    st.dataframe(pd.DataFrame(results).style.background_gradient(
                        cmap="YlOrRd", 
                        subset=["Similarité"]
                    ))
                    
                    fig = plot_similarity_results(results, method)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Voir le document de référence"):
                        st.write(documents[ref_index]["text"])
                else:
                    st.warning("Aucun résultat à afficher. Vérifiez vos paramètres.")
    else:
        st.warning("Veuillez charger des documents pour commencer l'analyse.")

with tab2:
    st.header("Comparaison des méthodes")
    
    if uploaded_files and len(uploaded_files) >= 2:
        documents = []
        for uploaded_file in uploaded_files:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
            documents.append({
                "name": uploaded_file.name,
                "text": text,
                "processed": preprocess_text(text, remove_stopwords, apply_stemming, apply_lemmatization)
            })

        # Choisir le document de référence
        document_names = [doc["name"] for doc in documents]
        ref_name = st.selectbox("Choisissez le document de référence :", document_names)
        ref_index = document_names.index(ref_name)
        
        if st.button("Comparer les méthodes"):
            with st.spinner("Comparaison en cours..."):
                st.write(documents)
                comparison_results = compare_methods(documents, ref_index=ref_index)
                
                st.subheader(f"Résultats de comparaison (référence : {ref_name})")
                st.dataframe(comparison_results)
                
                st.subheader("Visualisation des similarités")
                methods = ["TF-IDF", "Word2Vec", "BERT", "Keywords"]
                for method in methods:
                    method_results = comparison_results[comparison_results["method"] == method] 
                    print(method_results)
                    if not method_results.empty:
                        fig = plot_similarity_results(method_results, method)
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Veuillez charger au moins 2 documents pour comparer les méthodes.")
        
with tab3:
    st.header("À propos de cette application")
    st.markdown("""
    ### Analyseur de similarité de textes
    
    **Méthodes implémentées:**
    
    1. **TF-IDF**: 

    2. **Word2Vec**:
       
    3. **BERT**:

    4. **Mots-clés fréquents**:

    **Options de prétraitement:**
    - Suppression des stop-words
    - Stemming des mots
    - La lemmatisation
    - Filtrage par expression régulière

    """)