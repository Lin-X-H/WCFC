import numpy as np
import torch

def extract_features(model, dataloader, device, max_samples=50):
    feats = []
    labels = []

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_samples:
                break
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            x = x.to(device)
            feat = model(x).cpu().numpy()[0]
            feats.append(feat)
            labels.append(y.item())

    return np.array(feats), np.array(labels)

def continuous_feature_stats(feats):
    stats = {
        "mean": feats.mean(),
        "std": feats.std(),
        "min": feats.min(),
        "max": feats.max(),
        "per_dim_std": feats.std(axis=0)
    }

    low_var_ratio = np.mean(stats["per_dim_std"] < 1e-3)
    stats["low_variance_ratio"] = low_var_ratio
    return stats

def binarize(feats, threshold="median"):
    if threshold == "median":
        th = np.median(feats, axis=1, keepdims=True)
    elif threshold == "mean":
        th = np.mean(feats, axis=1, keepdims=True)
    else:
        th = threshold
    return (feats > th).astype(np.int8)

def hamming(a, b):
    return np.sum(a != b)

def hamming_analysis(bins, labels):
    intra = []
    inter = []

    for i in range(len(bins)):
        for j in range(i + 1, len(bins)):
            d = hamming(bins[i], bins[j])
            if labels[i] == labels[j]:
                intra.append(d)
            else:
                inter.append(d)

    return {
        "intra_mean": np.mean(intra) if intra else None,
        "inter_mean": np.mean(inter),
        "inter_std": np.std(inter),
        "intra_list": intra,
        "inter_list": inter
    }

def bit_entropy(bins):
    p = bins.mean(axis=0)
    eps = 1e-6
    entropy = -(p*np.log2(p+eps) + (1-p)*np.log2(1-p+eps))
    return {
        "mean_entropy": entropy.mean(),
        "low_entropy_ratio": np.mean(entropy < 0.3),
        "p_mean": p.mean()
    }
