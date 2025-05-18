import torch
import torch.optim as optim
from torchviz import make_dot
import numpy as np
import yaml
import os
from datetime import datetime
from data.loader import load_data
from data.preprocess import get_embeddings, reduce_dimensions
from models.autoencoder import Autoencoder
from models.losses import DECLoss
from utils.metrics import evaluate_clustering
from utils.visualization import plot_embeddings, plot_elbow

def main():
    # Load config
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{timestamp}"
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    
    # Prepare data
    texts, true_labels = load_data(config["dataset"])
    if config["dataset"] in ["ag_news", "glue/sst2"]:
        embeddings = get_embeddings(texts)
        features = reduce_dimensions(embeddings)
    else:
        features = np.array(texts)
    
    # Initialize training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(features.shape[1], config["latent_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    recon_loss_fn = torch.nn.MSELoss()
    dec_loss_fn = DECLoss(alpha=config["alpha"])
    
    # Convert data to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        recon, latent = model(features_tensor)
        loss_recon = recon_loss_fn(recon, features_tensor)
        
        # DEC loss after warmup
        if epoch >= config["warmup_epochs"]:
            loss_dec = dec_loss_fn(latent, epoch, 
                                 config["warmup_epochs"], 
                                 config["lambda"])
            loss = loss_recon + loss_dec
        else:
            loss = loss_recon
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Logging
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {loss.item():.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, f"{output_dir}/checkpoints/model_final.pt")
    
    # Generate and save visualizations
    with torch.no_grad():
        _, latent = model(features_tensor)
    latent_np = latent.cpu().numpy()
    
    # Cluster and plot
    kmeans = KMeans(n_clusters=config["num_clusters"])
    pred_labels = kmeans.fit_predict(latent_np)
    
    # Save plots
    fig = plot_embeddings(latent_np, pred_labels)
    fig.savefig(f"{output_dir}/figures/cluster_projection.png")
    plt.close(fig)
    
    fig = plot_elbow(latent_np)
    fig.savefig(f"{output_dir}/figures/elbow_curve.png")
    plt.close(fig)
    
    # Save architecture diagram
    make_dot(recon, params=dict(model.named_parameters())).render(
        f"{output_dir}/figures/architecture", format="png")
    
    print(f"Training complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()