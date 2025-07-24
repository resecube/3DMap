import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from utils.ovutils import construct_instances
from dataset.matterport import MatterportDataset
from dataset.scannet import ScanNetDataset
import numpy as np
import torch
from dataset.constants import (
    SCANNET_LABELS,
    SCANNET_IDS,
    SCANNET_LABELS_DESCRIPTION,
    MATTERPORT_IDS,
    MATTERPORT_LABELS,
    MATTERPORT_LABEL_DESCRIPTIONS,
)
import os


@hydra.main(
    config_path="../conf", config_name="app/export_instances.yaml", version_base="1.3"
)
def export_instances(cfg: DictConfig):
    target_labels = globals()[cfg.dataset.target_labels]
    target_ids = globals()[cfg.dataset.target_ids]
    for seq_name in tqdm(cfg.splits.seq_name_list):
        cfg.seq_name = seq_name
        dataset = (
            ScanNetDataset(cfg.dataset.data_root, cfg.seq_name)
            if cfg.dataset.name == "scannet"
            else MatterportDataset(cfg.dataset.data_root, cfg.seq_name)
        )
        object_dict = np.load(cfg.object_dict_path, allow_pickle=True).item()
        feature_dict = torch.load(cfg.clip_features_save_path)
        label_features = torch.load(cfg.label_feature_save_path)
        instances = construct_instances(
            object_dict,
            feature_dict,
            dataset.get_scene_points(),
            label_features,
            target_labels,
            target_ids,
        )
        torch.save(instances, cfg.instances_save_path)
