# Deep Embedded Clustering for Hugging Face Datasets


Implementation of Deep Embedded Clustering (DEC) adapted for Hugging Face datasets, following the architecture described in my IEEE conference paper.

## Features

- Hybrid neural autoencoder with trainable clustering head
- Support for text (AG News, SST-2) and image (MNIST) datasets
- Comprehensive evaluation metrics:
  - Silhouette Score
  - Davies-Bouldin Index  
  - Adjusted Rand Index (supervised)
- Visualization tools:
  - t-SNE/PCA projections
  - Elbow method plots
  - Architecture visualization

## 🚀 Installation & Execution

### Prerequisites
- Python 3.8+
- pip package manager
- Git (for cloning)

### Step-by-Step Setup

#### 1. Clone the Repository
```bash```
git clone https://github.com/yourusername/deep-embedded-clustering.git
cd deep-embedded-clustering

# For Linux/Mac
python -m venv venv
source venv/bin/activate

# For Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

```Install dependencies```
pip install -r requirements.txt

```Optional Configuration```
nano configs/default.yaml  # Linux/Mac
notepad configs/default.yaml  # Windows

```Run```
python train.py

# Replace with your actual checkpoint path
python evaluate.py --checkpoint outputs/20240101_120000/checkpoints/model_final.pt

# View Results``
## Linux/Mac
open outputs/latest/figures/cluster_projection.png  
eog outputs/latest/figures/cluster_projection.png  # Alternative

## Windows
start outputs\latest\figures\cluster_projection.png

```Expected Output Structure```
outputs/
└── YYYYMMDD_HHMMSS/  # Timestamped run
    ├── checkpoints/
    │   └── model_final.pt
    └── figures/
        ├── architecture.png
        ├── cluster_projection.png
        └── elbow_curve.png