# you need to put every block to jupyter notebook and will get result.

# train and test to make json file

# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# dataset
register_coco_instances("class_20", {}, "./data/train/trainval.json", "./data/train/images")
class_20_metadata = MetadataCatalog.get("class_20")
dataset_dicts = DatasetCatalog.get("class_20")

# show some image with mask
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=class_20_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow(str(d["file_name"]), vis.get_image()[:, :, ::-1])
    cv2.waitKey()
    cv2.destroyAllWindows()

# setting model and hyper-parameters
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("class_20",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "X-101-32x8d.pkl"   #initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 120000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # 20 classes 

# training
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

##
##test and make json file
##
# test dataset
register_coco_instances("test", {}, "./test_images/test.json", "./test_images/test_images") # test dataset path
class_20_metadata = MetadataCatalog.get("class_20")
dataset_dicts = DatasetCatalog.get("test")

# setting
cfg.MODEL.WEIGHTS = os.path.join("./output", "model_0119999.pth") # trained weight path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("test", )
predictor = DefaultPredictor(cfg)

# test and make json file
anno = []
for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    instances = outputs['instances']

    for i in range(len(instances.pred_classes)):
        instances.pred_masks_rle = [mask_util.encode(np.asfortranarray(mask)) for mask in instances.pred_masks.detach().cpu().numpy()]
        for rle in instances.pred_masks_rle:
            rle['counts'] = rle['counts'].decode('utf-8')
        code = instances.pred_masks_rle
        predict = {'image_id': d['image_id'], 'category_id': instances.pred_classes[i].item() + 1,
                   'score': instances.scores[i].item(), 'segmentation': code[i]}
        anno.append(predict)

# write json file
with open('output_trained/0856605.json', 'w') as f:
    json.dump(anno, f)
