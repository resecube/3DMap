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
from utils.ovutils import load_clip, desolve_instances
import hydra
from utils.time_utils import timeit
import pandas as pd


class OpenMap:

    SYSTEM_PROMPT = "You are a indoor navigation assistant optimizing scene understanding"

    def __init__(self, instances, clip_model, cfg: DictConfig):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.instances = instances
        self.kdtree = KDTree(self.instances["center"])
        self.clip_model = clip_model
        self.cfg = cfg
        self.dialogue = Dialogue(cfg)
    
    def get_labels(self,):
        labels = set(self.instances["label"])
        return labels
    
    def get_surr_labels(self, inst_idx, k=5):
        print(f"[INFO] getting surroundings of instance {inst_idx}")
        neighbor_idx, _ = self.kdtree.query(
            self.instances["center"][inst_idx], k=k
        )
        neighbor_idx = np.asarray(neighbor_idx).astype(int)

        if neighbor_idx.ndim == 0:
            neighbor_idx = np.array([neighbor_idx])
        labels = self.instances["label"][neighbor_idx]

        if labels.ndim == 1:
            neighbor_labels = list(set(labels))
        else:
            neighbor_labels = [list(set(row)) for row in labels]

        return neighbor_labels
    
    def get_better_desc(self,instruction_type, instruction_query):
        self.dialogue.clear()
        existing_items = list(self.get_labels())
        templates = {
            "object_goal": """
Focus:
- Refine generic object names to scene-specific variants
- Add critical attributes (position/material/state)
- use a clear, descriptive phrase that captures both the functional purpose,and no more than 12 words""",

            "language_guided": """
Focus:
- Convert descriptive phrases to attribute-focused queries
- use a clear, descriptive phrase that captures both the functional purpose,and no more than 12 words""",

            "demand_driven": """
Focus:
- Map abstract needs to concrete objects
- Emphasize functionality over appearance
- use a clear, descriptive phrase that captures both the functional purpose,and no more than 12 words"""
        }
        base_prompt = f"Help me optimize the navigation instructions to better match the feature vectors of the images encoded by clip.\nYou may use your common sense, and if you find it helps, you can refer to the environment information.\nNOTICE: Output the content of a optimized instruction only!(No Explanation!No Quotation Marks)\nEnvironment items: {', '.join(existing_items)}\nOriginal instruction: {instruction_query}\n"
        full_prompt = base_prompt + templates[instruction_type]
        
        self.dialogue.sys_say(OpenMap.SYSTEM_PROMPT)
        resp = self.dialogue.user_query(full_prompt)
        return resp
    
    def select_reasonable_target(self, topk_indices, topk_scores, surr_labels):
        candidate_str = "\n".join(
            f"[Index {i}] Score: {s:.2f} | Surroundings: {sl}"
            for i, (s, sl) in enumerate(zip(topk_scores, surr_labels))
        )
        prompt = f"""Next, I will provide the instance information matched by CLIP based on the instructions you optimized. You can also refer to the item label information around the candidate instances I provided below (which may not be accurate).
Select and output the index number of the instance in the last line that you think is the most suitable for the task. (no explanations)
Candidates:\n{candidate_str}"""
        resp = self.dialogue.user_query(prompt)
        idx = int(resp.split("\n")[-1].split()[-1])
        return (
            topk_indices[idx],
            self.instances["label"][topk_indices[idx]],
            self.instances["center"][topk_indices[idx]],
        )
    @timeit
    def get_instances_by_desc(self, desc, k=1):
        desc = tokenizer.tokenize(desc).to(self.device)
        desc_embedding = self.clip_model.encode_text(desc)
        features = self.instances["feature"]
        match_scores = torch.mm(desc_embedding, features.T)
        self.instances["score"] = match_scores
        topk_scores, topk_indices = torch.topk(match_scores, k=k, dim=1)
        return topk_indices.squeeze(), topk_scores.squeeze()
    
    def pipeline(self,instruction_type, instruction_query):
        # get instance description
        resp = self.get_better_desc(instruction_type, instruction_query)
        self.dialogue.save("dialogue.txt")
        print(f"[INFO] better description: {resp}")
        # filter instances by description, and get surrounding environment information
        topk_indices, topk_scores = self.get_instances_by_desc(resp, k=5)
        surr_labels = self.get_surr_labels(topk_indices, k=5)
        # choosing the most suitable instance
        target_idx, target_label, target_center = self.select_reasonable_target(
            topk_indices, topk_scores, surr_labels
        )
        self.dialogue.save("dialogue.txt")
        print(f"[INFO] target label: {target_label}, target center: {target_center}")
        return target_label, resp, target_center

@hydra.main(config_path="../conf", config_name="base_config.yaml", version_base="1.3")
def main(config: DictConfig):
    clip_model, _ = load_clip()
    instances = torch.load(config.data_root + '/instances.pth', map_location=config.device)

    openmap = OpenMap(instances, clip_model, config)
    df = pd.read_csv(config.data_root + '/task.csv')
    from utils.viz_utils import vis_heatmap
    for i,instruction_query in enumerate(df['origin_instr']):
        
        target_label, target_desc, target_center = openmap.pipeline(df.loc[i,'instr_type'], instruction_query)
        vis_heatmap(o3d.io.read_triangle_mesh(config.data_root + f'/{config.seq_name}.ply'),openmap.instances)
        df.loc[i,'optimized_instr'] = target_desc
    df.to_csv(config.data_root + '/task.csv', index=False)
if __name__ == "__main__":
    main()