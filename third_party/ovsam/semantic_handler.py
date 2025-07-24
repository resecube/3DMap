import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area
import sys
from third_party.ovsam.models.model import SAMSegmentor
from typing import Any, Dict, List, Optional, Tuple
from mmengine import Config, print_log
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmengine import Config, print_log
from mmengine.structures import InstanceData
import cv2
from pathlib import Path
from .predictor import OVSAM_Predictor
from dataset.matterport import MatterportDataset
import pickle
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from third_party.ovsam.ext.class_names.lvis_list import LVIS_CLASSES

LVIS_NAMES = LVIS_CLASSES
mask_threshold = 0.0


def get_cls(cls_pred):
    # 把第二维度压缩掉
    cls_pred = cls_pred.squeeze(1)
    scores, indices = torch.topk(cls_pred, 1, dim=1)
    scores, indices = scores.squeeze(1), indices.squeeze(1)
    names = [LVIS_NAMES[ind].replace("_", " ") for ind in indices]

    # cls_info = []
    # score_pred = []
    # for name, score in zip(names, scores):
    #     cls_info.append("{} ({:.2f})".format(name, score))
    #     score_pred.append(score)
    return names, scores


class SemanticHandler:
    def __init__(self, config_path):
        model_cfg = Config.fromfile(config_path)
        model = MODELS.build(model_cfg.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device=self.device)
        model = model.eval()
        model.init_weights()
        self.predictor = OVSAM_Predictor(model)

    def preprocess(self, mask_img):
        # 获取所有唯一的掩码 ID（去除背景 0）
        mask_ids = np.unique(mask_img)
        mask_ids = mask_ids[mask_ids > 0]  # 过滤掉背景 ID（假设背景是 0）

        bbox_list = []

        for mask_id in mask_ids:
            # 获取当前 ID 的掩码
            mask = (mask_img == mask_id).astype(np.uint8)

            # 计算边界框
            x, y, w, h = cv2.boundingRect(mask)
            upper_left = (x, y)
            lower_right = (x + w - 1, y + h - 1)

            bbox_list.append((x, y, x + w - 1, y + h - 1))

        return bbox_list

    def predict(self, rgb_img, mask_img):
        # 调用掩码预处理方法，得到框提示数据
        bbox_list = self.preprocess(mask_img)
        bbox_list = torch.tensor(bbox_list, dtype=torch.float32, device=self.device)
        # 读取图像
        # rgb_img = cv2.imread(rgb_img_path)
        self.predictor.set_image(rgb_img)
        # 调用预测器进行预测
        masks, _, cls_pred, semantic_pred = self.predictor.predict_with_boxes(bbox_list)
        # 完成类别预测
        cls_info, score_pred = get_cls(cls_pred)
        return cls_info, semantic_pred, bbox_list

    def visualize(self, img, label, bbox_list):
        # 在图像上在框提示中央绘制标签，并绘制出框和框的索引
        img_copy = img.copy()  # 避免修改原图

        # 设置字体
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, bbox in enumerate(bbox_list):
            min_x, min_y, max_x, max_y = map(int, bbox)
            cv2.rectangle(img_copy, (min_x, min_y), (max_x, max_y), (0, 255, 0), 1)

            # 在左上角绘制 label
            cv2.putText(
                img_copy,
                label[i],
                (min_x, max(10, min_y - 5)),
                font,
                0.5,
                (0, 255, 0),
                2,
            )

            # 在框的中央标注索引
            # center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
            # cv2.putText(img_copy, str(i), (center_x, center_y), font, 0.6, (255, 0, 0), 2)

        # 保存图像
        save_path = "visualized_img.png"
        cv2.imwrite(save_path, img_copy)


@hydra.main(config_path="../../conf", config_name="app/ovsam_extract.yaml")
def get_ovsam_features(cfg: DictConfig):
    semantic_handler = SemanticHandler(cfg.ovsam_model_path)
    data_root = Path(cfg.ovsam_src_root_path)
    for scene_dir in data_root.iterdir():
        print(str(scene_dir))
        if not scene_dir.is_dir():
            continue
        if cfg.debug:
            print(f"Processing scene {scene_dir.name}")
        dataset = MatterportDataset(str(data_root), scene_dir.name)
        frame_list = dataset.get_frame_list(1)
        result_dict = {}
        frame_list = tqdm(frame_list)
        for frame_id in frame_list:
            rgb_image = dataset.get_rgb(frame_id)
            mask_image = dataset.get_segmentation(frame_id)
            cls_info, semantic_pred, bbox_list = semantic_handler.predict(
                rgb_image, mask_image
            )
            result_dict[frame_id] = semantic_pred.cpu().numpy()
        with open(str(scene_dir / cfg.ovsam_save_feature_path), "wb") as f:
            pickle.dump(result_dict, f)
        if cfg.debug:
            print(f"Scene {scene_dir.name} done.")


if __name__ == "__main__":
    get_ovsam_features()
