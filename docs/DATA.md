# Data Preparation

DRESS expects datasets under the `data/` directory by default. This directory is intentionally ignored by Git because the benchmark files can be large or subject to dataset licenses.

The experiments use YelpChi and Amazon from [CARE-GNN](https://dl.acm.org/doi/abs/10.1145/3340531.3411903), with processed files available from the [CARE-GNN repository](https://github.com/YingtongDou/CARE-GNN/tree/master/data), and S-FFSD from the [AntiFraud framework](https://github.com/AI4Risk/antifraud).

Expected layout:

```text
data/
  S-FFSDneofull.csv
  S-FFSD_neigh_feat.csv
  YelpChi.mat
  yelp_homo_adjlists.pickle
  yelp_neigh_feat.csv            # optional
  Amazon.mat
  amz_homo_adjlists.pickle
  amazon_neigh_feat.csv          # optional
```

You can change the directory with the `data_dir` field in `configs/DRESS_cfg.yaml`.

The training script writes generated DGL graph binaries such as `graph-S-FFSD.bin` back into the same data directory.
