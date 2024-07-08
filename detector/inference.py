import os
import json
import mmcv
import torch
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
from mmdet.structures import DetDataSample
import numpy as np
import argparse


def det_data_sample_to_coco_predictions(det_data_sample, image_id, person_label_index=0):
    predictions = []
    bboxes = det_data_sample.pred_instances.bboxes.cpu().numpy()
    scores = det_data_sample.pred_instances.scores.cpu().numpy()
    labels = det_data_sample.pred_instances.labels.cpu().numpy()
    
    person_indices = np.where(labels == person_label_index)[0]

    for idx in person_indices:
        bbox = bboxes[idx]
        score = scores[idx]
        
        coco_bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]
        
        prediction = {
            'image_id': image_id,
            'category_id': 0,
            'bbox': coco_bbox,
            'score': float(score)
        }
        predictions.append(prediction)
    
    return predictions

def infer_and_save_coco_predictions(config_file, checkpoint_file, img_folder, result_file, person_label_index=0, device='cuda:0'):
    model = init_detector(config_file, checkpoint_file, device=device)

    img_files = [os.path.join(img_folder, img) for img in os.listdir(img_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]

    coco_predictions = []

    for img_file in tqdm(img_files):
        image_id = os.path.basename(img_file)

        result = inference_detector(model, img_file)
        if isinstance(result, DetDataSample):
            predictions = det_data_sample_to_coco_predictions(result, image_id, person_label_index)
            coco_predictions.extend(predictions)
    
    with open(result_file, 'w') as f:
        json.dump(coco_predictions, f, ensure_ascii=False, indent=4)
        

def main(opt):
    config_file = opt.config
    checkpoint_file = opt.checkpoint
    img_folder = opt.img_folder
    result_file = opt.save_path
    
    infer_and_save_coco_predictions(config_file, checkpoint_file, img_folder, result_file, person_label_index=0)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
    parser.add_argument('--config', type=str, default='', help='config file path')
    parser.add_argument('--img_folder', type=str, default='', help='img file path')
    parser.add_argument('--save_path', type=str, default='runs/results.json', help='json save path')
    opt = parser.parse_args()
    print(vars(opt))
    return opt
    
    
    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
