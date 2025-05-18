import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def plot_embeddings(embeddings, labels, method="PCA"):
    """Generate and return cluster visualization figure"""
    fig = plt.figure(figsize=(10, 8))
    
    # Dimensionality reduction
    if method == "PCA":
        coords = PCA(n_components=2).fit_transform(embeddings)
    else:  # t-SNE
        coords = TSNE(n_components=2, perplexity=30).fit_transform(embeddings)
    
    # Plot
    scatter = sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=labels,
        palette="viridis",
        alpha=0.7,
        s=50
    )
    plt.title(f"{method} Projection of Latent Space")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    
    return fig

def plot_elbow(latent, max_k=10):
    """Generate and return elbow method figure"""
    distortions = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(latent)
        distortions.append(kmeans.inertia_)
    
    fig = plt.figure()
    plt.plot(range(2, max_k + 1), distortions, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    
    return fig