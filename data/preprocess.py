from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch

def get_embeddings(texts):
    """Generate Sentence-BERT embeddings"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts, show_progress_bar=True)

def reduce_dimensions(embeddings, n_components=256):
    """PCA dimensionality reduction"""
    return PCA(n_components=n_components).fit_transform(embeddings)