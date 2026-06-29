import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np
import pandas as pd
from math import sqrt
import random

from sklearn.metrics._ranking import _binary_clf_curve, roc_curve, auc
from torch.optim.lr_scheduler import MultiStepLR


class PosEncoding(nn.Module):

    def __init__(self, dim, device, base=10000, bias=0):

        super(PosEncoding, self).__init__()
        """
        Initialize the posencoding component
        :param dim: the encoding dimension 
		:param device: where to train model
		:param base: the encoding base
		:param bias: the encoding bias
        """
        p = []
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(
            sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)


class TransformerConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 # feat_drop=0.6,
                 # attn_drop=0.6,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 activation=nn.PReLU()):
        """
        Initialize the transformer layer.
        Attentional weights are jointly optimized in an end-to-end mechanism with graph neural networks and fraud detection networks.
            :param in_feat: the shape of input feature
            :param out_feats: the shape of output feature
            :param num_heads: the number of multi-head attention 
            :param bias: whether to use bias
            :param allow_zero_in_degree: whether to allow zero in degree
            :param skip_feat: whether to skip some feature 
            :param gated: whether to use gate
            :param layer_norm: whether to use layer regularization
            :param activation: the type of activation function   
        """

        super(TransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        self.lin_query = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_key = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_value = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)

        # self.feat_dropout = nn.Dropout(p=feat_drop)
        # self.attn_dropout = nn.Dropout(p=attn_drop)
        if skip_feat:
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        else:
            self.skip_feat = None
        if gated:
            self.gate = nn.Linear(
                3*self._out_feats*self._num_heads, 1, bias=bias)
        else:
            self.gate = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats*self._num_heads)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, graph, feat, get_attention=False):
        """
        Description: Transformer Graph Convolution
        :param graph: input graph
            :param feat: input feat
            :param get_attention: whether to get attention
        """

        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        # Step 0. q, k, v
        q_src = self.lin_query(
            h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(
            h_src).view(-1, self._num_heads, self._out_feats)
        # Assign features to nodes
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax(
            graph, graph.edata['a'] / self._out_feats**0.5)

        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'),
                         fn.sum('attn', 'agg_u'))

        # output results to the destination nodes
        rst = graph.dstdata['agg_u'].reshape(-1,
                                             self._out_feats*self._num_heads)

        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            if self.gate is not None:
                gate = torch.sigmoid(
                    self.gate(
                        torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst


class Tabular1DCNN2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        K: int = 4,  # K*input_dim -> hidden dim
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hid_dim = input_dim * embed_dim * 2
        self.cha_input = self.cha_output = input_dim
        self.cha_hidden = (input_dim*K) // 2
        self.sign_size1 = 2 * embed_dim
        self.sign_size2 = embed_dim
        self.K = K

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(input_dim, self.hid_dim)

        self.bn_cv1 = nn.BatchNorm1d(self.cha_input)
        self.conv1 = nn.Conv1d(
            in_channels=self.cha_input,
            out_channels=self.cha_input*self.K,
            kernel_size=5,
            padding=2,
            groups=self.cha_input,
            bias=False
        )

        self.ave_pool1 = nn.AdaptiveAvgPool1d(self.sign_size2)

        self.bn_cv2 = nn.BatchNorm1d(self.cha_input*self.K)
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=self.cha_input*self.K,
            out_channels=self.cha_input*(self.K),
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.bn_cv3 = nn.BatchNorm1d(self.cha_input*self.K)
        self.conv3 = nn.Conv1d(
            in_channels=self.cha_input*(self.K),
            out_channels=self.cha_input*(self.K//2),
            kernel_size=3,
            padding=1,
            # groups=self.cha_hidden,
            bias=True
        )

        self.bn_cvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(6):
            self.bn_cvs.append(nn.BatchNorm1d(self.cha_input*(self.K//2)))
            self.convs.append(nn.Conv1d(
                in_channels=self.cha_input*(self.K//2),
                out_channels=self.cha_input*(self.K//2),
                kernel_size=3,
                padding=1,
                # groups=self.cha_hidden,
                bias=True
            ))

        self.bn_cv10 = nn.BatchNorm1d(self.cha_input*(self.K//2))
        self.conv10 = nn.Conv1d(
            in_channels=self.cha_input*(self.K//2),
            out_channels=self.cha_output,
            kernel_size=3,
            padding=1,
            # groups=self.cha_hidden,
            bias=True
        )

    def forward(self, x):
        x = self.dropout1(self.bn1(x))
        x = nn.functional.celu(self.dense1(x))
        x = x.reshape(x.shape[0], self.cha_input,
                      self.sign_size1)

        x = self.bn_cv1(x)
        x = nn.functional.relu(self.conv1(x))
        x = self.ave_pool1(x)

        x_input = x
        x = self.dropout2(self.bn_cv2(x))
        x = nn.functional.relu(self.conv2(x))  # -> (|b|,24,32)
        x = x + x_input

        x = self.bn_cv3(x)
        x = nn.functional.relu(self.conv3(x))  # -> (|b|,6,32)

        for i in range(6):
            x_input = x
            x = self.bn_cvs[i](x)
            x = nn.functional.relu(self.convs[i](x))
            x = x + x_input

        x = self.bn_cv10(x)
        x = nn.functional.relu(self.conv10(x))

        return x


class TransEmbedding(nn.Module):

    def __init__(
        self,
        df=None,
        device='cpu',
        dropout=0.2,
        in_feats_dim=82,
        cat_features=None,
        neigh_features: dict = None,
        att_head_num: int = 4,  # yelp 4 amazon 5 S-FFSD 9
        neighstat_uni_dim=64
    ):
        """
        Initialize the attribute embedding and feature learning compoent

        :param df: the feature (|train_idx|, |feat|)
        :param device: where to train model
        :param dropout: the dropout rate
        :param in_feats_dim: the shape of input feature in dimension 1
        :param cat_features: category features
        :param neigh_features: neighbor riskstat features
        :param att_head_num: attention head number for riskstat embeddings
        """
        super(TransEmbedding, self).__init__()
        self.time_pe = PosEncoding(dim=in_feats_dim, device=device, base=100)

        self.cat_table = nn.ModuleDict(
            {col: nn.Embedding(max(df[col].unique()) + 10, in_feats_dim).to(device) for col in cat_features if
             col not in {"Labels", "Time"}})
        # self.cat_table = nn.ModuleDict({col: nn.Embedding(max(df[col].unique(
        # ))+1, in_feats_dim).to(device) for col in cat_features if col not in {"Labels", "Time"}})

        if isinstance(neigh_features, dict):
            self.nei_table = Tabular1DCNN2(input_dim=len(
                neigh_features), embed_dim=in_feats_dim)

        self.att_head_num = att_head_num
        self.att_head_size = int(in_feats_dim / att_head_num)
        self.total_head_size = in_feats_dim
        self.lin_q = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_k = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_v = nn.Linear(in_feats_dim, self.total_head_size)

        self.lin_final = nn.Linear(in_feats_dim, in_feats_dim)
        self.layer_norm = nn.LayerNorm(in_feats_dim, eps=1e-8)

        self.neigh_mlp = nn.Linear(in_feats_dim, 1)

        self.neigh_add_mlp = nn.ModuleList([nn.Linear(in_feats_dim, in_feats_dim) for i in range(
            len(neigh_features.columns))]) if isinstance(neigh_features, pd.DataFrame) else None

        self.label_table = nn.Embedding(
            3, in_feats_dim, padding_idx=2).to(device)
        self.time_emb = None
        self.emb_dict = None
        self.label_emb = None
        self.cat_features = cat_features
        self.neigh_features = neigh_features
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(in_feats_dim, in_feats_dim) for i in range(len(cat_features))])
        self.dropout = nn.Dropout(dropout)

    def forward_emb(self, cat_feat):
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
        # print(self.emb_dict)
        # print(df['trans_md'])
        support = {col: self.emb_dict[col](
            cat_feat[col]) for col in self.cat_features if col not in {"Labels", "Time"}}
        return support

    def transpose_for_scores(self, input_tensor):
        new_x_shape = input_tensor.size(
        )[:-1] + (self.att_head_num, self.att_head_size)
        # (|batch|, feat_num, dim) -> (|batch|, feta_num, head_num, head_size)
        input_tensor = input_tensor.view(*new_x_shape)
        return input_tensor.permute(0, 2, 1, 3)

    def forward_neigh_emb(self, neighstat_feat):
        cols = neighstat_feat.keys()
        tensor_list = []
        for col in cols:
            tensor_list.append(neighstat_feat[col])
        neis = torch.stack(tensor_list).T
        input_tensor = self.nei_table(neis)

        mixed_q_layer = self.lin_q(input_tensor)
        mixed_k_layer = self.lin_k(input_tensor)
        mixed_v_layer = self.lin_v(input_tensor)

        q_layer = self.transpose_for_scores(mixed_q_layer)
        k_layer = self.transpose_for_scores(mixed_k_layer)
        v_layer = self.transpose_for_scores(mixed_v_layer)

        att_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        att_scores = att_scores / sqrt(self.att_head_size)

        att_probs = nn.Softmax(dim=-1)(att_scores)
        # dropout?
        context_layer = torch.matmul(att_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.total_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        hidden_states = self.lin_final(context_layer)
        # dropout?
        # hidden_states = self.layer_norm(hidden_states + input_tensor)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, cols
        # return input_tensor, cols

    def forward(self, cat_feat: dict, neighstat_feat: dict):
        support = self.forward_emb(cat_feat)
        cat_output = 0
        nei_output = 0
        for i, k in enumerate(support.keys()):
            # if k =='time_span':
            #    print(df[k].shape)
            support[k] = self.dropout(support[k])
            support[k] = self.forward_mlp[i](support[k])
            cat_output = cat_output + support[k]

        if neighstat_feat is not None:
            nei_embs, cols_list = self.forward_neigh_emb(neighstat_feat)
            nei_output = self.neigh_mlp(nei_embs).squeeze(-1)

            # nei_output = nei_embs.mean(axis=-1)

        return cat_output, nei_output


class TemporalGatedAttn(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 post_proc=True,
                 n2v_feat=True,
                 drop=None,
                 ref_df=None,
                 cat_features=None,
                 neigh_features=None,
                 nei_att_head=4,
                 device='cpu'):
        """
        Initialize the Temporal Gated Attention Module(TGAT)
        :param in_feats: the shape of input feature
        :param hidden_dim: model hidden layer dimension
        :param n_layers: the number of TGAT layers
        :param n_classes: the number of classification
        :param heads: the number of multi-head attention 
        :param activation: the type of activation function
        :param skip_feat: whether to skip some feature
        :param gated: whether to use gate
        :param layer_norm: whether to use layer regularization
        :param post_proc: whether to use post processing
        :param n2v_feat: whether to use n2v features
        :param drop: whether to use drop
        :param ref_df: whether to refer other node features
        :param cat_features: category features
        :param neigh_features: neighbor statistic features
        :param nei_att_head: multihead attention for neighbor riskstat features
        :param device: where to train model
        """

        super(TemporalGatedAttn, self).__init__()
        self.in_feats = in_feats  # feature dimension
        self.hidden_dim = hidden_dim  # 64
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads  # [4,4,4]
        self.activation = activation  # PRelu
        # self.input_drop = lambda x: x
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        # self.pn = PairNorm(mode=pairnorm)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(
            n_classes + 1, in_feats, padding_idx=n_classes))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim*self.heads[0]),
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim *
                                                   self.heads[0], in_feats)
                                         ))

        # build multiple layers
        self.layers.append(TransformerConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           num_heads=self.heads[0],
                                           skip_feat=skip_feat,
                                           gated=gated,
                                           layer_norm=layer_norm,
                                           activation=self.activation))

        for l in range(0, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(TransformerConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                               out_feats=self.hidden_dim,
                                               num_heads=self.heads[l],
                                               skip_feat=skip_feat,
                                               gated=gated,
                                               layer_norm=layer_norm,
                                               activation=self.activation))
        if post_proc:
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                                             nn.BatchNorm1d(
                                                 self.hidden_dim * self.heads[-1]),
                                             nn.PReLU(),
                                             nn.Dropout(self.drop),
                                             nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))
        else:
            self.layers.append(nn.Linear(self.hidden_dim *
                               self.heads[-1], self.n_classes))

    def forward(self, blocks, features, risk_scores, labels, neighstat_feat=None):
        """
        :param blocks: train blocks
        :param features: train features
        :param labels: train labels
        :param n2v_feat: whether to use n2v features
        :param neighstat_feat: neighbor riskstat features
        """
        # if not risk_scores.dim() == 2:
        #     risk_scores = risk_scores.unsqueeze(1)
        # h = torch.cat([features, risk_scores], dim=1)
        h = features

        label_embed = self.input_drop(self.layers[0](labels))
        label_embed = self.layers[1](
            h) + self.layers[2](label_embed)  # 2926, 2926, 256
        # label_embed = self.layers[1](h)
        label_embed = self.layers[3](label_embed)
        h = h + label_embed

        # label_embed = self.layers[0](torch.cat([h, risk_scores], dim=1))
        # label_embed = self.layers[1](label_embed)
        # h = label_embed

        for l in range(self.n_layers):
            h = self.output_drop(self.layers[l+4](blocks[l], h))

        logits = self.layers[-1](h)

        return logits, h

class DQN_RiskExtractor(nn.Module):
    """
    The Q-Network used in a DQN designed for feature risk extraction in the DRESS model.
    It takes input features and outputs risk vector containing scores for different actions.
    """
    def __init__(self, input_dim,n_actions=2):
        """
        Initialize the DQN_RiskExtractor model.
        :param input_dim: The dimension of input features.
        :param n_actions: The number of possible actions (default is 2).
        """
        super(DQN_RiskExtractor, self).__init__()
        self.risk_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4,n_actions),
            nn.Softmax(dim=1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the network.
        :param x: Input features.
        :return: Risk scores for each action.
        """
        return self.risk_extractor(x)

class DRESS(nn.Module):
    def __init__(self, in_dim, nei_feat_dim, cat_feat, feat_df, nei_feat,nei_att_head, sample_size, args, device = "cpu"):
        """
        Initialize the DRESS model.
        :param in_dim: Input dimension of node features.
        :param nei_feat_dim: Dimension of neighbor features.
        :param cat_feat: Categorical features.
        :param feat_df: DataFrame containing node features.
        :param nei_feat: Neighbor features.
        :param nei_att_head: Number of attention heads for neighbor features.
        :param sample_size: Size of experience replay samples.
        :param args: Additional configuration arguments.
        :param device: Device to run the model on (default is "cpu").
        """
        super(DRESS, self).__init__()
        self.ml_risk_score = args['ml_risk_score']
        self.intrinsic_reward = args['intrinsic_reward']
        if self.intrinsic_reward:
            self.attention_net = nn.Sequential(
                nn.Linear(in_dim, in_dim//4),
                nn.ReLU(),
                nn.Linear(in_dim//4, in_dim),
                nn.Sigmoid()  # Output continuous attention weights (0~1)
            )
        self.agent_DQN = DQN_RiskExtractor(in_dim)
        self.target_DQN = DQN_RiskExtractor(in_dim)
        # Disable gradient updates for target_DQN
        for param in self.target_DQN.parameters():
            param.requires_grad = False

        self.tgat = TemporalGatedAttn(in_feats=in_dim+nei_feat_dim,
                      hidden_dim=args['hid_dim']//4,
                      n_classes=2,
                      heads=[4]*args['n_layers'],
                      activation=nn.PReLU(),
                      n_layers=args['n_layers'],
                      drop=args['dropout'],
                      device=device,
                      gated=args['gated'],
                      ref_df=feat_df,
                      cat_features=cat_feat,
                      neigh_features=nei_feat,
                      nei_att_head=nei_att_head).to(device)

        self.n2v_mlp = TransEmbedding(
            feat_df, device=device, in_feats_dim=feat_df.shape[1], cat_features=cat_feat, neigh_features=nei_feat,
            att_head_num=nei_att_head)

        self.experience_list = []
        self.sample_size = 2#sample_size
        self.gamma = 0.99
        self.device = device

        self.best_thr = 0.5

        self.alpha = 0.6

        self.temperature = 0.1
        self.top_k_percent = 0.5

        self.loss_fn = nn.CrossEntropyLoss().to(device)
        lr = args['lr'] * np.sqrt(args['batch_size'] / 1024)
        self.optimizer = optim.Adam(self.parameters(), lr=lr,
                               weight_decay=args['wd'])
        # self.lr_scheduler = MultiStepLR(optimizer=self.optimizer, milestones=[
        #     400, 1600], gamma=0.3)

        if args['dataset'] == 'amazon':
            self.lr_scheduler = MultiStepLR(optimizer=self.optimizer, milestones=[
                800], gamma=0.3)  # for amazon
        elif args['dataset'] == 'yelp':
            self.lr_scheduler = MultiStepLR(optimizer=self.optimizer, milestones=[
                                   40,160], gamma=0.3)#for yelp
        elif args['dataset'] == 'S-FFSD':
            self.lr_scheduler = MultiStepLR(optimizer=self.optimizer, milestones=[
                800,1600], gamma=0.3)  # for S-FFSD

    def generate_pseudo_features(self, x, attn_weights, temperature=0.1, top_k_percent=0.25):
        """
        Generate pseudo features based on attention weights.
        :param x: Input features.
        :param attn_weights: Attention weights.
        :param temperature: Temperature parameter for softmax.
        :param top_k_percent: Percentage of top features to retain.
        :return: Pseudo features.
        """
        attn_weights_normalized = F.softmax(attn_weights, dim=1)
        # Select top K% of high-attention features
        num_features = attn_weights_normalized.size(1)
        k = max(1, int(num_features * top_k_percent))
        threshold = torch.topk(attn_weights_normalized, k=k, dim=1)[0][:, -1].unsqueeze(1)
        # Generate soft mask
        soft_mask = torch.sigmoid((attn_weights_normalized - threshold) / temperature)
        # Calculate pseudo features
        masked_x = x * (1 - soft_mask)
        mean_value = torch.mean(x, dim=1, keepdim=True)
        inverted_x = mean_value * soft_mask
        pseudo_x = masked_x + inverted_x

        return pseudo_x

    def select_action(self, risk_scores, true_labels=None):
        """
        Select actions based on risk scores.
        :param risk_scores: Predicted risk scores.
        :param true_labels: True labels (if available).
        :return: Selected actions.
        """
        if true_labels is None:
            risk_scores = risk_scores.detach().cpu().numpy()
            return np.where(risk_scores > self.best_thr, 1, 0)
        mask = true_labels == 2
        true_labels[mask] = 0
        # true_labels = true_labels[~mask]
        # risk_scores1 = risk_scores[~mask]

        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.detach().cpu().numpy()
        if isinstance(risk_scores, torch.Tensor):
            risk_scores1 = risk_scores.detach().cpu().numpy()
            risk_scores = risk_scores.detach().cpu().numpy()

        fps, tps, thresholds = _binary_clf_curve(true_labels, risk_scores)
        n_pos = np.sum(true_labels)
        n_neg = len(true_labels) - n_pos
        fns = n_pos - tps
        tns = n_neg - fps

        f11 = 2 * tps / (2 * tps + fns + fps)
        f10 = 2 * tns / (2 * tns + fns + fps)
        marco_f1 = (f11 + f10) / 2
        best_f1_thr = thresholds[np.argmax(marco_f1)]

        fpr, tpr, thresholds = roc_curve(true_labels, risk_scores)
        auc_values = auc(fpr, tpr)

        # Choose actions based on risk scores
        action = np.where(risk_scores > best_f1_thr, 1, 0)
        self.best_thr = best_f1_thr

        return action

    def intrinsic_reward_calculate(self, chosen_reward, rejected_reward):
        """
        Calculate intrinsic reward based on chosen and rejected risk scores.
        :param chosen_reward: Risk scores for chosen samples
        :param rejected_reward: Risk scores for rejected samples
        :return: Intrinsic reward.
        """
        diff = rejected_reward - chosen_reward
        pairwise_loss = F.softplus(diff)  # softplus(x) = log(1 + e^x)
        # print("rejected_score:",rejected_reward[:3],"chosen_score:",chosen_reward[:3],"pairwise_loss:",loss.mean())
        return -pairwise_loss.mean()

    def get_loss_DQN(self, update = False):
        """
        Calculate DQN loss using experience replay.
        :param update: Whether to update the target network.
        :return: DQN loss.
        """
        if len(self.experience_list) < self.sample_size:
            return
        experience = random.sample(self.experience_list, self.sample_size)

        s = [item['state'] for item in experience][0]
        s_next = [item['next_state'] for item in experience][0]
        reward = [item['reward'] for item in experience][0]
        a = [item['action'] for item in experience][0]
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self.device)

        Q_next =self.target_DQN(s_next).max(1)[0]
        target_values = reward + self.gamma * Q_next
        agent_values = self.agent_DQN(s).gather(1, a)
        if target_values.size(0) > agent_values.size(0):
            target_values = target_values[:agent_values.size(0)]
        elif target_values.size(0) < agent_values.size(0):
            agent_values = agent_values[:target_values.size(0)]
        loss_DQN = nn.SmoothL1Loss()(agent_values, target_values.unsqueeze(1))
        if update:
            self.target_DQN.load_state_dict(self.agent_DQN.state_dict())

        return loss_DQN

    def forward(self, env, update=False):
        """
        Forward pass of the DRESS model.
        :param env: Environment containing state information.
        :param update: Whether to update the target network.
        :return: Model outputs including logits, loss, labels, seeds, risk scores, and final hidden states.
        """
        s, s_next = env.state_input, env.state_input_next
        num_feat, cat_feat, neighstat_feat, batch_labels, propagate_labels, blocks, seeds = s
        batch_labels_all = propagate_labels.clone()
        batch_labels_all[:len(batch_labels)] = batch_labels
        cat_h, nei_h = self.n2v_mlp(cat_feat, neighstat_feat)
        h = num_feat + cat_h

        if s_next is not None:
            # For training
            num_feat_next, cat_feat_next = s_next
            cat_h_next,_ = self.n2v_mlp(cat_feat_next,None)
            h_next = num_feat_next + cat_h_next
            if self.intrinsic_reward:
                attn_weights = self.attention_net(h)
                pseudo_h = self.generate_pseudo_features(h, attn_weights, self.temperature, self.top_k_percent)
                h = h * attn_weights
                attn_weights_next = self.attention_net(h_next)
                h_next = h_next * attn_weights_next

                mask = (propagate_labels == 1)
                risk_scores_vec = self.agent_DQN(h)
                risk_scores = risk_scores_vec[:,1]
                # risk_scores_vec1 = self.target_DQN(h)
                pseudo_scores_vec = self.agent_DQN(pseudo_h)
                risk_scores_anomaly = risk_scores_vec[:,1][mask]
                pseudo_scores_anomaly = pseudo_scores_vec[:,1][mask]

                intrinsic_reward = self.intrinsic_reward_calculate(risk_scores_anomaly, pseudo_scores_anomaly)
            else:
                risk_scores_vec = self.agent_DQN(h)
                risk_scores = risk_scores_vec[:, 1]

            # risk_score = self.agent_DQN(h)[:,1]
        else:
            # For predicting
            if self.intrinsic_reward:
                pairwise_loss = 0
                attn_weights = self.attention_net(h)
                h = h * attn_weights
                risk_scores_vec = self.target_DQN(h)
            else:
                if self.ml_risk_score:
                    self.target_DQN.load_state_dict(self.agent_DQN.state_dict())
                risk_scores_vec = self.target_DQN(h)
            risk_scores = risk_scores_vec[:, 1]

        action = self.select_action(risk_scores.clone(), batch_labels_all.clone())

        extrinsic_reward = env.step(action)

        if s_next is not None:
            if self.intrinsic_reward:
                reward = extrinsic_reward + intrinsic_reward.detach()
            else:
                reward = extrinsic_reward
            experience = {'state':h.detach(),'action':action,'reward':reward,'next_state':h_next.detach()}
            self.experience_list.append(experience)
            # Prevent experience buffer from explosion
            if len(self.experience_list) > 10:
                self.experience_list.pop()

            loss_DQN = self.get_loss_DQN(update)
        else:
            loss_DQN = 0

        h = torch.cat([h, nei_h], dim=-1)

        if not self.ml_risk_score:
            y_logits, final_h = self.tgat(blocks, h, risk_scores_vec.detach(), propagate_labels)
        else:
            y_logits, final_h = self.tgat(blocks, h, risk_scores.detach(), propagate_labels)

        y_predict = y_logits.max(1)[1]
        if not self.ml_risk_score:
            loss_drl = loss_DQN
        else:
            loss_drl = 0

        return y_logits, loss_drl, batch_labels, seeds, risk_scores_vec, final_h

    def optimize(self, y_logits, labels, loss_drl):
        """
        Optimize the model parameters.
        :param y_logits: Model outputs.
        :param labels: True labels.
        :param loss_drl: DRL loss.
        :return: Training loss.
        """
        if self.ml_risk_score:
            train_loss = self.loss_fn(y_logits, labels) + 0.3 * loss_drl
        else:
            train_loss = self.loss_fn(y_logits, labels) + self.alpha * loss_drl
        # backward
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return train_loss.cpu().detach().numpy()