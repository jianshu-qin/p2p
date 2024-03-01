import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def segment_person(image_path):
    # 加载图像
    image = cv2.imread(image_path)

    # 设置 Detectron2 配置
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # 创建 Detectron2 预测器
    predictor = DefaultPredictor(cfg)

    # 进行预测
    outputs = predictor(image)

    # 获取人物 mask
    masks = outputs["instances"].pred_masks.cpu().numpy()

    # 显示或保存人物 mask
    for i, mask in enumerate(masks):
        person_mask = (mask * 255).astype("uint8")
        # cv2.imshow(f"Person {i+1} Mask", person_mask)
        cv2.imwrite(f"person_{i+1}_mask.png", person_mask)


# 示例用法
image_path = "../p2p_vis/girl.png"
segment_person(image_path)
