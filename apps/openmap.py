from omegaconf import OmegaConf, DictConfig
from typing import Any, ItemsView, List
import os
import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
import torch
from copy import deepcopy
from pathlib import Path
import torch.nn.functional as F
from open_clip import tokenizer
from utils.llm_utils import Dialogue
from utils.viz_utils import VizUtils


class OpenMap:
    def __init__(self, instance_path, clip_model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.instances = torch.load(instance_path, map_location=self.device)
        self.kdtree = KDTree(self.instances["center"])
        self.clip_model = clip_model
        

    def get_labels(
        self,
    ):
        labels = set(self.instances["label"])
        return labels

    def filter_features(self, features):
        return features

    def get_instances_by_desc(self, desc, k=1):
        desc = tokenizer.tokenize(desc).to(self.device)
        desc_embedding = self.clip_model.encode_text(desc)
        features = self.instances["feature"]
        match_scores = torch.mm(desc_embedding, features.T)
        self.instances["score"] = match_scores
        topk_scores, topk_indices = torch.topk(match_scores, k=k, dim=1)
        return topk_indices, topk_scores

    def get_surr_labels(self, inst_idx, k=5):
        neighbor_idx, neighbor_dist = self.kdtree.query(
            self.instances["center"][inst_idx], k=k
        )
        # 确保索引是整数类型
        neighbor_idx = np.asarray(neighbor_idx).astype(int)

        # 处理k=1时的特殊情况
        if neighbor_idx.ndim == 0:
            neighbor_idx = np.array([neighbor_idx])
        neighbor_labels = self.instances["label"][neighbor_idx]
        return neighbor_labels, neighbor_dist

    def answer_query(self, query):
        dialogue = Dialogue(self.cfg)
        dialogue.sys_say(
            "The existing scene navigation task needs to identify the target accurately and expand the description of the target.\
            The following items are known to exist in the current scene. In subsequent instructions, if the goal is not clear, guess the possible goal according to\
            the requirements of the instruction (for example, 'I am thirsty', guess that I may need to find a water dispenser or glass).\
            If there are multiple navigation targets, disassemble the target first. Please give a natural description of an object in each line according to the instructions provided later, \
            expand the object in terms of material, shape, use, etc., and do not output any superfluous content."
        )
        resp = dialogue.user_query(query)
        topk_indices, topk_scores = self.get_instances_by_desc(resp, k=5)
        surr_labels, surr_dist = self.get_surr_labels(topk_indices, k=5)

        req = "Using the description you provided, I looked up the highly relevant instances in the scene, and then I will give you a list with the labels of the instances found, several objects around the instances found, you need to choose the most likely target instance, just output the index of the instances in the last line"
        for topk_idx, topk_score, surrlabel, surrdist in zip(topk_indices, topk_scores, surr_labels, surr_dist):
            surr_desc = ",".join(
                [
                    f" object {label} is {dist} meters away from the instance"
                    for label, dist in zip(surrlabel, surrdist)
                ]
            )
            req += f"instance with index {topk_idx} has a score of {topk_score}, and {surr_desc}"
        resp = dialogue.user_query(req)
        idx = resp.split("\n")[-1]
        return (
            topk_idx[idx],
            self.instances["label"][topk_idx[idx]],
            self.instances["center"][topk_idx[idx]],
        )
