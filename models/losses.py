import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class DECLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, latent, epoch, warmup_epochs, lambda_val):
        if epoch < warmup_epochs:
            return torch.tensor(0.0).to(latent.device)
            
        latent_np = latent.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=latent.shape[1], random_state=42).fit(latent_np)
        centroids = torch.tensor(kmeans.cluster_centers_, device=latent.device)
        
        pairwise_dist = torch.cdist(latent, centroids)**2
        q = (1 + pairwise_dist / self.alpha)**(-(self.alpha + 1)/2)
        q = q / q.sum(dim=1, keepdim=True)
        
        p = q**2 / q.sum(dim=0)
        p = p / p.sum(dim=1, keepdim=True)
        
        loss = (p * (torch.log(p) - torch.log(q)).sum()
        return lambda_val * min(1.0, (epoch - warmup_epochs)/20) * loss