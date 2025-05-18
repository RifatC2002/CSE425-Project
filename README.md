# Deep Embedded Clustering for Hugging Face Datasets

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Implementation of Deep Embedded Clustering (DEC) adapted for Hugging Face datasets, following the architecture described in our IEEE conference paper.

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

## Installation

```bash
git clone https://github.com/yourusername/deep-embedded-clustering.git
cd deep-embedded-clustering
pip install -r requirements.txt