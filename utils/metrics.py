from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

def evaluate_clustering(features, pred_labels, true_labels=None):
    metrics = {
        "silhouette": silhouette_score(features, pred_labels),
        "davies_bouldin": davies_bouldin_score(features, pred_labels),
        "calinski_harabasz": calinski_harabasz_score(features, pred_labels)
    }
    
    if true_labels is not None:
        metrics["ari"] = adjusted_rand_score(true_labels, pred_labels)
    
    return metrics