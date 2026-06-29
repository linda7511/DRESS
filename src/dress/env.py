import copy
import os
import random

from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import NodeDataLoader
import numpy as np
import torch

class Env():
    """
    Environment class for interacting with the graph data, suiting for the training/validation process.
    It handles data loading, state management, and reward calculation.
    """
    def __init__(self, graph, idx, args,num_feat, cat_feat, nei_feat, labels, mode='train'):
        """
        Initialize the Environment.
        :param graph: The graph structure.
        :param idx: Indices of nodes for this environment (training/validation/test).
        :param args: Configuration arguments.
        :param num_feat: Numerical features of nodes.
        :param cat_feat: Categorical features of nodes.
        :param nei_feat: Neighbor features of nodes.
        :param labels: Labels of nodes.
        :param mode: Mode of operation ('train', 'val', or 'test').
        """
        super().__init__()
        self.device = args['device']
        ind = torch.from_numpy(np.array(idx)).long().to(self.device)
        sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        self.NodeDataLoader = NodeDataLoader(graph,
                                          ind,
                                          sampler,
                                          device=self.device,
                                          use_ddp=False,
                                          batch_size=args['batch_size'],
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=0
                                          )
        self.dataloader = iter(self.NodeDataLoader)

        self.num_feat = num_feat
        self.cat_feat = cat_feat
        self.nei_feat = nei_feat
        self.labels = labels

        self.done = False
        self.mode = mode
        self.steps = 0
        # self.state = {'input_nodes':[],'seeds':[],'blocks':[]}
        self.state_next = None
        self.state_input = None
        self.state_input_next = None


    def load_lpa_subtensor(self,
            seeds,  # (|batch|,)
            input_nodes,  # (|batch_all|,)
            blocks,
            input_nodes_next=None,
    ):
        """
        Put the input data into the device
        :param node_feat: the feature of input nodes
        :param work_node_feat: the feature of work nodes
        :param neigh_feat: neighborhood stat feature -> pd.DataFrame
        :param neigh_padding_dict: padding length of neighstat features
        :param labels: the labels of nodes
        :param seeds: the index of one batch data
        :param input_nodes: the index of batch input nodes -> batch all size!!!
        :param device: where to train model
        :param blocks: dgl blocks
        """
        # masking to avoid label leakage
        if "1hop_riskstat" in self.nei_feat.keys() and len(blocks) >= 2:
            # nei_hop1 = get_k_neighs(graph, seeds, 1)
            nei_hop1 = blocks[-2].dstdata['_ID']
            self.nei_feat['1hop_riskstat'][nei_hop1] = 0

        if "2hop_riskstat" in self.nei_feat.keys() and len(blocks) >= 3:
            # nei_hop2 = get_k_neighs(graph, seeds, 2)
            nei_hop2 = blocks[-3].dstdata['_ID']
            self.nei_feat['2hop_riskstat'][nei_hop2] = 0

        batch_inputs = self.num_feat[input_nodes].to(self.device)
        batch_work_inputs = {i: self.cat_feat[i][input_nodes].to(
            self.device) for i in self.cat_feat if i not in {"labels"}}  # cat feats

        batch_neighstat_inputs = None

        if self.nei_feat:
            batch_neighstat_inputs = {col: self.nei_feat[col][input_nodes].to(
                self.device) for col in self.nei_feat.keys()}

        batch_labels = self.labels[seeds].to(self.device)
        train_labels = copy.deepcopy(self.labels)
        propagate_labels = train_labels[input_nodes]  # (|input_nodes|,) 45324
        propagate_labels[:seeds.shape[0]] = 2

        blocks = [block.to(self.device) for block in blocks]

        batch_inputs_next = None

        # batch_labels_all = self.labels[input_nodes].to(self.device)
        if input_nodes_next is not None:
            batch_inputs_next = self.num_feat[input_nodes_next].to(self.device)
            batch_work_inputs_next = {i: self.cat_feat[i][input_nodes_next].to(
                self.device) for i in self.cat_feat if i not in {"labels"}}
            batch_inputs_next = [batch_inputs_next, batch_work_inputs_next]
        else:
            batch_inputs_next = None


        return [batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, propagate_labels, blocks, seeds], batch_inputs_next

    def extrinsic_reward(self, action):
        # Calculate extrinsic reward based on actions and ground truth labels
        batch_labels = self.state_input[3]
        reward1 = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for a, p in zip(action, batch_labels):
            if a == 1 and p.item() == 1:
                reward1 += 1
                TP += 1
            elif a == 0 and p.item() == 1:
                reward1 -= 1
                FN += 1
            elif a == 0 and p.item() == 0:
                TN += 1
            elif a == 1 and (p.item() == 0 or p.item() == 2):
                FP += 1
        if TP + FP == 0 or TP + FN == 0:
            reward = reward1/len(action)
        else:
            reward = (TP / (TP + FP) + TP / (TP + FN))/2
        # print("reward: ", reward,"TP: ",TP,"FP: ",FP,"TN: ",TN, 'FN: ',FN)
        return reward

    def step(self, action=None):
        """
        Take a step in the environment.
        In training mode, calculate the extrinsic reward based on the action and load the next batch of data.
        In validation/test mode, load the next batch of data.
        :param action: Action taken (only used in training mode).
        :return: Extrinsic reward (in training mode) or None (in validation/test mode).
        """
        self.steps += 1

        if self.mode == "train":
            extrinsic_reward = self.extrinsic_reward(action)
            # Get the current batch's input nodes, seeds, and blocks
            input_nodes, seeds, blocks = self.state_next
            try:
                # Load the next batch of data
                self.state_next = next(self.dataloader)
                input_nodes_next, _, _ = self.state_next
            except StopIteration:
                # If there's no more data, set state_next to None and mark as done
                self.state_next = None
                input_nodes_next = None
                self.done = True

            # Load the current batch's data into the environment's state
            self.state_input, self.state_input_next = self.load_lpa_subtensor(
                seeds, input_nodes, blocks, input_nodes_next)
            return extrinsic_reward

        else:
            try:
                # Load the next batch of data
                input_nodes, seeds, blocks = next(self.dataloader)
                self.state_input, _ = self.load_lpa_subtensor(
                     seeds, input_nodes, blocks)
            except StopIteration:
                self.state_input = None
                self.done = True

    def reset(self):
        # Reset the status of environment
        self.steps = 0
        self.done = False
        self.dataloader = iter(self.NodeDataLoader)
        input_nodes, seeds, blocks = next(self.dataloader)
        input_nodes_next = None
        if self.mode == "train":
            self.state_next = next(self.dataloader)
            input_nodes_next, _, _ = self.state_next

        self.state_input, self.state_input_next = self.load_lpa_subtensor(
            seeds, input_nodes, blocks, input_nodes_next)

        # return self.state_input, self.state_input_next
