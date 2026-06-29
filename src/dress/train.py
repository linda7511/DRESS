import os
import random
from argparse import ArgumentParser
from pathlib import Path

import yaml
import time
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import NodeDataLoader
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from scipy.io import loadmat
from tqdm import tqdm
from .env import Env
from .utils import get_prior_distribution, early_stopper
from .model import DRESS

def main(args):
    feat_df, labels, train_idx, test_idx, graph, cat_features, neigh_features = load_rgtan_data(
        args['dataset'],
        args['test_size'],
        args.get('data_dir', 'data'),
    )

    device = args['device']
    graph = graph.to(device)
    oof_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    update_K_steps = 10

    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(
        device) for col in cat_features}

    nei_feat = []
    if isinstance(neigh_features, pd.DataFrame):
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(
            device) for col in neigh_features.columns}

    # trn_idx, val_idx = train_test_split(train_idx, test_size=0.5)

    # Split training set into 10% trn and 10% validation.
    # Balanced sampling with the proportion of positive samples increased to 0.1
    anomaly_idx = []
    nonanomaly_idx = []
    val_idx = []
    num_trn = len(train_idx) // 2
    num_labeled = num_trn // 10
    for trn_idx in train_idx:
        if len(anomaly_idx) < num_labeled and labels[trn_idx] == 1:
            anomaly_idx.append(trn_idx)
        elif len(nonanomaly_idx) < num_trn - num_labeled and (labels[trn_idx] == 0 or labels[trn_idx] == 2):
            nonanomaly_idx.append(trn_idx)
        else:
            val_idx.append(trn_idx)
    trn_idx = anomaly_idx + nonanomaly_idx
    print(labels[trn_idx].value_counts())
    print(labels[val_idx].value_counts())
    print(labels[test_idx].value_counts())

    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    num_feat = torch.from_numpy(feat_df.values).float().to(device)

    # Estimate prior distribution for unlabeled samples.
    # This is used when using ML to extract feature risks.
    if args['ml_risk_score']:
        anomaly_idx = []
        unlabeled_idx = []
        for i in trn_idx:
            if labels[i] == 1:
                anomaly_idx.append(i)
            elif labels[i] == 2:
                unlabeled_idx.append(i)
        p, miu, sigma = get_prior_distribution(num_feat, anomaly_idx, unlabeled_idx, device)

    best_cv = 100
    for fold in range(args['n_fold']):
        print(f'Training fold {fold + 1}')

        nei_att_head = args['nei_att_heads'][args['dataset']]
        nei_feat_dim = len(nei_feat.keys()) if isinstance(nei_feat, dict) else 0
        train_env = Env(graph,trn_idx,args, num_feat, cat_feat, nei_feat, labels, mode='train')
        val_env = Env(graph,val_idx,args, num_feat, cat_feat, nei_feat, labels,mode='val')

        model = DRESS(in_dim=feat_df.shape[1],
                        nei_feat_dim = nei_feat_dim,
                        cat_feat=cat_feat,
                        feat_df=feat_df,
                        nei_feat=nei_feat,
                        nei_att_head = nei_att_head,
                        sample_size=1,
                        args=args,
                        device=device
                        ).to(device)

        earlystoper = early_stopper(
            patience=args['early_stopping'], verbose=True)

        result_list = [] # To record performance at different epochs
        for epoch in range(0, args['max_epochs']):
            train_loss_list = []
            model.train()
            batch_next = None
            train_env.reset()
            while not train_env.done:
                train_batch_logits, loss_drl, batch_labels, seeds, p_score, _ = model(train_env, update=(
                            (train_env.steps + 1) % update_K_steps == 0))

                # for ml_risk_score
                if args['ml_risk_score']:
                    L_score = 0
                    alpha = 0.75
                    beta = 2
                    m = 0.5
                    for i,idx in zip(range(len(seeds)),seeds):
                        gap = (p_score[i] - miu)/sigma
                        L_score += (1-alpha)*(1-batch_labels[i])*pow(1-p[idx.item()],beta)*gap + alpha*batch_labels[i]*max(0,m-gap)
                    loss_drl = L_score / len(seeds)

                mask = batch_labels == 2
                # train_batch_logits = train_batch_logits[~mask]
                # batch_labels = batch_labels[~mask]
                batch_labels[mask] = 0

                if loss_drl is not None:
                    train_loss = model.optimize(train_batch_logits, batch_labels, loss_drl)

                    train_loss_list.append(train_loss)

                if train_env.steps % 10 == 0:
                    # visualize_score(p_score,batch_labels_all,epoch,step,'train')

                    tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone(
                    ).detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                    score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[
                        :, 1].cpu().numpy()
                    try:
                        print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                              'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, train_env.steps,
                                                                                           np.mean(
                                                                                               train_loss_list),
                                                                                           average_precision_score(
                                                                                               batch_labels.cpu().numpy(), score),
                                                                                           tr_batch_pred.detach(),
                                                                                           roc_auc_score(batch_labels.cpu().numpy(), score)))
                    except:
                        pass

            # mini-batch for validation
            val_loss_list = 0
            val_acc_list = 0
            val_all_list = 0
            val_env.reset()
            model.eval()
            with torch.no_grad():
                final_h_epoch = []
                batch_labels_epoch = []
                score_epoch = []
                pred_epoch = []
                while not val_env.done:
                    val_batch_logits, _, batch_labels, seeds, risk_scores_vec, final_h = model(val_env)

                    oof_predictions[seeds] = val_batch_logits
                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]
                    final_h = final_h[~mask]
                    final_h_epoch.extend(final_h.detach().cpu().numpy())
                    batch_labels_epoch.extend(batch_labels.detach().cpu().numpy())

                    # batch_labels[mask] = 0
                    val_loss_list = val_loss_list + \
                        loss_fn(val_batch_logits, batch_labels)
                    # val_all_list += 1
                    val_batch_pred = torch.sum(torch.argmax(
                        val_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                    val_acc_list = val_acc_list + val_batch_pred * \
                        torch.tensor(
                            batch_labels.shape[0])  # how many in this batch is right!
                    val_all_list = val_all_list + \
                        batch_labels.shape[0]  # how many val nodes

                    pred_epoch.extend(torch.argmax(val_batch_logits, dim=1).detach().cpu().numpy())

                    score_epoch.extend(torch.softmax(val_batch_logits.clone().detach(), dim=1)[:, 1].cpu().numpy())

                    if val_env.steps % 10 == 0:
                        # visualize_score(final_h_embedded, batch_labels_all, epoch, step, 'val')

                        score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[
                            :, 1].cpu().numpy()
                        try:
                            print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                  'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                          val_env.steps,
                                                                          val_loss_list/val_all_list,
                                                                          average_precision_score(
                                                                              batch_labels.cpu().numpy(), score),
                                                                          val_batch_pred.detach(),
                                                                          roc_auc_score(batch_labels.cpu().numpy(), score)))
                        except:
                            pass
                result_list.append({'epoch': epoch,
                                    'val_auc': roc_auc_score(batch_labels_epoch,score_epoch),
                                    'val_f1':f1_score(batch_labels_epoch, pred_epoch,average="macro"),
                                    'val_ap':average_precision_score(batch_labels_epoch,score_epoch)})
                # Print epoch-result records.
                # print('In epoch:{:03d}| val_loss:{:4f}, val_ap:{:.4f}, '
                #       'val_acc:{:.4f}, val_f1:{:.4f}, val_auc:{:.4f}'.format(epoch,
                #                                                              val_loss_list / val_all_list,
                #                                                              average_precision_score(
                #                                                                  batch_labels_epoch, score_epoch),
                #                                                              val_batch_pred.detach(),
                #                                                              f1_score(batch_labels_epoch, pred_epoch,
                #                                                                       average="macro"),
                #                                                              roc_auc_score(batch_labels_epoch,
                #                                                                            score_epoch)))

                # Visualize node embeddings
                # tsne = TSNE()
                # final_h_embedded = tsne.fit_transform(final_h_epoch)
                # visualize_score(final_h_embedded, batch_labels_epoch, epoch, 0, 'val')

            earlystoper.earlystop(val_loss_list/val_all_list, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))
        if best_cv > earlystoper.best_cv:
            b_model = earlystoper.best_model.to(device)
            best_cv = earlystoper.best_cv

        for result in result_list:
            print(result)
    b_model.eval()
    final_h_all = torch.from_numpy(
        np.zeros([len(feat_df), 256])).float().to(device)
    test_env = Env(graph, test_idx, args,num_feat, cat_feat, nei_feat, labels, mode='test')
    with torch.no_grad():
        test_env.reset()
        total_infer_time = 0.0
        total_infer_samples = 0
        per_sample_time_list = []
        while not test_env.done:
            # measure inference time for this batch
            t0 = time.perf_counter()
            test_batch_logits, _, batch_labels, seeds, risk_scores_vec, final_h = b_model(test_env)
            t1 = time.perf_counter()

            # accumulate timing (distribute batch time equally to samples in batch)
            batch_n = len(seeds)
            elapsed = t1 - t0
            if batch_n > 0:
                per_sample = elapsed / batch_n
                per_sample_time_list.extend([per_sample] * batch_n)
                total_infer_time += elapsed
                total_infer_samples += batch_n

            test_predictions[seeds] = test_batch_logits
            final_h_all[seeds] = final_h
            test_batch_pred = torch.sum(torch.argmax(
                test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
            if test_env.steps % 10 == 0:
                # visualize_score(risk_scores_vec, batch_labels_all, 0, step, 'test')
                print('In test batch:{:04d}'.format(test_env.steps))
    val_gnn_0, test_gnn_0 = oof_predictions, test_predictions

    test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
    y_target = labels[test_idx].cpu().numpy()
    test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()

    mask = y_target != 2
    test_score = test_score[mask]
    y_target = y_target[mask]
    test_score1 = test_score1[mask]

    final_h_all = final_h_all[test_idx].cpu().numpy()
    final_h_all = final_h_all[mask]

    # Visulize node embeddings
    # try:
    #     tsne = TSNE()
    #     final_h_embedded = tsne.fit_transform(final_h_all)
    #     visualize_score(final_h_embedded, y_target, 'test', 0, 'val')
    # except:
    #     print('TSNE failed')

    print("test AUC:", roc_auc_score(y_target, test_score))
    print("test f1:", f1_score(y_target, test_score1, average="macro"))
    print("test AP:", average_precision_score(y_target, test_score))
    # Print inference latency statistics (per-sample, milliseconds)
    try:
        if total_infer_samples > 0:
            arr = np.array(per_sample_time_list)
            avg_ms = total_infer_time / total_infer_samples * 1000.0
            med_ms = np.median(arr) * 1000.0
            p95_ms = np.percentile(arr, 95) * 1000.0
            print("num_samples:", total_infer_samples, "total_infer_time (s):", total_infer_time)
            print(f"Inference latency per sample (ms): mean={avg_ms:.4f}, median={med_ms:.4f}, p95={p95_ms:.4f}")
        else:
            print("No inference samples were measured.")
    except Exception as e:
        print("Failed to compute inference latency stats:", e)
    label_prob = (np.array(test_score) >= 0.5).astype(int)
    print("test Precision:", precision_score(y_target, label_prob))
    print("test Recall:", recall_score(y_target, label_prob))

    checkpoint_path = Path(args.get('checkpoint_path', 'checkpoints/DRESS_ckpt.pth'))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(b_model.state_dict(), checkpoint_path)
    print(f"model saved to {checkpoint_path}")

def load_rgtan_data(dataset: str, test_size: float, data_dir: str = "data"):
    data_dir = Path(data_dir)
    if dataset == 'S-FFSD':
        cat_features = ["Target", "Location", "Type"]
        
        df = pd.read_csv(data_dir / "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        #####
        neigh_features = []
        #####
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in tqdm(data.groupby(column), desc=column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))
        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        #######
        g.ndata['label'] = torch.from_numpy(
            labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        #######

        graph_path = data_dir / "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(str(graph_path), [g])
        index = list(range(len(labels)))

        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2, shuffle=True)

        feat_neigh = pd.read_csv(data_dir / "S-FFSD_neigh_feat.csv")
        print("neighborhood feature loaded for nn input.")
        neigh_features = feat_neigh

    elif dataset == 'yelp':
        cat_features = []
        neigh_features = []
        data_file = loadmat(data_dir / 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(data_dir / 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = data_dir / "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(str(graph_path), [g])

        try:
            feat_neigh = pd.read_csv(data_dir / "yelp_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    elif dataset == 'amazon':
        cat_features = []
        neigh_features = []
        data_file = loadmat(data_dir / 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(data_dir / 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                test_size=test_size, random_state=2, shuffle=True)

        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = data_dir / "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(str(graph_path), [g])
        try:
            feat_neigh = pd.read_csv(data_dir / "amazon_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features


def parse_args():
    parser = ArgumentParser(description="Train DRESS for semi-supervised graph fraud detection.")
    parser.add_argument(
        "--config",
        default="configs/DRESS_cfg.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


if __name__ == "__main__":
    cli_args = parse_args()
    main(load_config(cli_args.config))
