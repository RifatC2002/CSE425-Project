import torch
import numpy as np
import os
from datetime import datetime
from sklearn.cluster import KMeans
from models.autoencoder import Autoencoder
from utils.metrics import evaluate_clustering
from utils.visualization import plot_embeddings
import yaml

def evaluate(checkpoint_path, features, true_labels=None):
    # Create output directory
    eval_dir = f"outputs/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path)
    
    model = Autoencoder(
        input_dim=features.shape[1],
        latent_dim=checkpoint['config']['latent_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get latent representations
    with torch.no_grad():
        _, latent = model(torch.tensor(features).float().to(device))
    latent_np = latent.cpu().numpy()
    
    # Cluster and evaluate
    kmeans = KMeans(n_clusters=checkpoint['config']['num_clusters'])
    pred_labels = kmeans.fit_predict(latent_np)
    
    metrics = evaluate_clustering(latent_np, pred_labels, true_labels)
    
    # Save results
    with open(f"{eval_dir}/metrics.txt", "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
            print(f"{name}: {value:.4f}")
    
    # Save visualization
    fig = plot_embeddings(latent_np, pred_labels)
    fig.savefig(f"{eval_dir}/cluster_projection.png")
    plt.close(fig)
    
    print(f"Evaluation complete. Results saved to {eval_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="ag_news")
    args = parser.parse_args()
    
    # Load and preprocess data
    texts, labels = load_data(args.dataset)
    if args.dataset in ["ag_news", "glue/sst2"]:
        embeddings = get_embeddings(texts)
        features = reduce_dimensions(embeddings)
    else:
        features = np.array(texts)
    
    evaluate(args.checkpoint, features, labels)