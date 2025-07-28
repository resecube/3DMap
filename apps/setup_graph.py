import os
import pickle
import hydra
import numpy as np
import torch
from tqdm import tqdm
from dataset import ScanNetDataset, MatterportDataset
from aggregation.graph import Graph


@hydra.main(config_path="../conf", config_name="setup_graph.yaml", version_base="1.3")
def setup_graph(cfg):
    for seq_name in tqdm(cfg.splits.seq_name_list):
        cfg.seq_name = seq_name
        dataset = (
            ScanNetDataset(cfg.dataset.data_root, cfg.seq_name)
            if cfg.dataset.name == "scannet"
            else MatterportDataset(cfg.dataset.data_root, cfg.seq_name)
        )

        os.makedirs(f"{cfg.seq_name}", exist_ok=True)
        os.makedirs(f"{cfg.semantic_info_path}", exist_ok=True)

        with open(dataset.feature_path, "rb") as f:
            mask_features = pickle.load(f)
        mask_features = {
            k: [torch.tensor(f).cpu() for f in v] for k, v in mask_features.items()
        }
        graph = Graph(cfg, mask_features, dataset)
        np.save(cfg.object_dict_path, graph.object_dict, allow_pickle=True)

if __name__ == "__main__":
    setup_graph()