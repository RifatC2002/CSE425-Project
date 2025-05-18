from datasets import load_dataset
import numpy as np
import torch

def load_data(dataset_name, sample_size=5000):
    """Load dataset from Hugging Face"""
    dataset = load_dataset(dataset_name, split=f"train[:{sample_size}]")
    
    if dataset_name == "ag_news":
        return dataset["text"], dataset["label"]
    elif dataset_name == "mnist":
        images = [np.array(img, dtype=np.float32).flatten() / 255.0 
                   for img in dataset["image"]]
        return images, dataset["label"]
    elif dataset_name == "glue/sst2":
        return dataset["sentence"], dataset["label"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")