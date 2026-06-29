import copy

import numpy as np
import pandas as pd
import toad
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler


class early_stopper(object):
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize the early stopper
        :param patience: the maximum number of rounds tolerated
        :param verbose: whether to stop early
        :param delta: the regularization factor
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_value = None
        self.best_cv = None
        self.is_earlystop = False
        self.count = 0
        self.best_model = None
        # self.val_preds = []
        # self.val_logits = []

    def earlystop(self, loss, model=None):  # , preds, logits):
        """
        :param loss: the loss score on validation set
        :param model: the model
        """
        value = -loss
        cv = loss
        # value = ap

        if self.best_value is None:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            # self.val_preds = preds
            # self.val_logits = logits
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print('EarlyStoper count: {:02d}'.format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            # self.val_preds = preds
            # self.val_logits = logits
            self.count = 0

def visualize_score(risk_scores_vec, batch_labels_all, epoch, step, val_test):
    matplotlib.use('Agg')

    # Convert to CPU numpy arrays
    if not isinstance(risk_scores_vec, np.ndarray):
        scores_np = risk_scores_vec.detach().cpu().numpy()
    else:
        scores_np = risk_scores_vec

    labels_np = np.array(batch_labels_all)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scores_scaled = scaler.fit_transform(scores_np)

    # Filter valid labels (0 and 1)
    mask = (labels_np == 0) | (labels_np == 1)
    valid_scores = scores_scaled
    valid_labels = labels_np

    plt.figure(figsize=(10, 8))

    # 按标签绘制散点
    scatter0 = plt.scatter(
        valid_scores[valid_labels == 0, 0],
        valid_scores[valid_labels == 0, 1],
        c='#6e85b7', alpha=0.6, label='Normal'
    )
    scatter1 = plt.scatter(
        valid_scores[valid_labels == 1, 0],
        valid_scores[valid_labels == 1, 1],
        c='#ff8b8b', alpha=0.6, label='Anomaly'
    )
    scatter2 = plt.scatter(
        valid_scores[valid_labels == 2, 0],
        valid_scores[valid_labels == 2, 1],
        c='#DC050C', alpha=0.6, label='Unlabeled'
    )

    # plt.xlabel('Action 0 Score', fontsize=12)
    # plt.ylabel('Action 1 Score', fontsize=12)
    # plt.title('Risk Score Distribution', fontsize=14)
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.6)

    plot_dir = Path("outputs") / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f'final_h_rlsc_{val_test}_epoch{epoch}_step{step}.png', dpi=300, bbox_inches='tight')
    plt.close()


def get_prior_distribution(node_feat, anomaly_idx, unlabeled_idx,device='cpu'):
    # 1. Get feature of anomaly samples and unlabeled samples
    anomaly_samples = node_feat[anomaly_idx].cpu().detach().numpy()
    unlabeled_samples = node_feat[unlabeled_idx].cpu().detach().numpy()

    # 2. Cluster anomaly samples and find cluster centers
    dbscan = DBSCAN(metric='euclidean')
    anomaly_clusters = dbscan.fit_predict(anomaly_samples)

    # Consider noise points as cluster centers
    noise_indices = np.where(anomaly_clusters == -1)
    cluster_centers = []
    # Obtain different types of abnormal clustering centers
    for i in range(len(np.unique(anomaly_clusters)) - 1):
        cluster_samples = anomaly_samples[anomaly_clusters == i]
        center = np.mean(cluster_samples, axis=0)
        cluster_centers.append(center)
    # Add noise points to the list of abnormal clustering centers
    for noise_index in noise_indices:
        cluster_centers.append(anomaly_samples[noise_index])
    cluster_centers = np.array(cluster_centers)

    # 3. Calculate distance scores for unlabeled samples
    distance_scores = []
    for xu in unlabeled_samples:
        distances = []
        for cj in cluster_centers:
            dist = np.sqrt(((xu - cj) ** 2).sum())
            distances.append(dist)
        distu = np.min(distances)
        distance_scores.append(distu)
    max_dist = np.max(distance_scores)
    min_dist = np.min(distance_scores)
    for i in range(len(unlabeled_samples)):
        distance_scores[i] = (max_dist - distance_scores[i])/(max_dist-min_dist)

    # 4. Calculate isolation scores for unlabeled samples
    isolation_forest = IsolationForest(max_samples=100)
    isolation_forest.fit(unlabeled_samples)
    isolation_scores = isolation_forest.decision_function(unlabeled_samples)
    isolation_scores = (isolation_scores - np.min(isolation_scores)) / (
            np.max(isolation_scores) - np.min(isolation_scores))

    # 5. Calculate prior anomaly scores
    tau = 0.5
    prior_scores = {}
    for u_idx,ds, is_score in zip(unlabeled_idx, distance_scores, isolation_scores):
        p = (1 - tau) * ds + tau * is_score
        prior_scores[u_idx] = p
    for a_i in anomaly_idx:
        prior_scores[a_i] = 1

    # 6. Calculate mean and variance of prior anomaly scores
    mu = np.mean(list(prior_scores.values()))
    sigma = np.sqrt(((np.array(list(prior_scores.values())) - mu) ** 2).sum() / (len(prior_scores) - 1))

    return prior_scores, mu, sigma
