import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import argparse
    
    
def calculate_overlap(box1, box2):
    """
    calculate IoU

    paras:
    - box1: [x1, y1, w1, h1]
    - box2: [x2, y2, w2, h2]

    return:
    - overlap: IoU
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    area_box1 = w1 * h1
    area_box2 = w2 * h2

    overlap = intersection_area / (area_box1 + area_box2 - intersection_area)

    return overlap


def calculate_tp(cocoGt, cocoDt, conf_threshold=0.25, IoU_threshold=0.5):
    img_ids = cocoGt.getImgIds()
    
    gt_label_count = 0
    correct_detection_count = 0

    for img_id in img_ids:
        ann_ids = cocoGt.getAnnIds(imgIds=img_id)
        gt_anns = cocoGt.loadAnns(ann_ids)
    
        gt_label_count += len(gt_anns)
    
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img_id)
        dt_anns = cocoDt.loadAnns(dt_ann_ids)
        
        dt_anns = [ann for ann in dt_anns if ann['score'] >= conf_threshold]
    
        gt_boxes = np.array([ann['bbox'] for ann in gt_anns])
    
        detected = [False] * len(dt_anns)
    
        for gt_ann in gt_anns:
            gt_box = gt_ann['bbox']
            gt_category = gt_ann['category_id']
    
            match_found = False
    
            # 对该图像的每个检测结果进行处理
            for i, dt_ann in enumerate(dt_anns):
                if detected[i]:
                    continue
                
                dt_box = dt_ann['bbox']
                dt_category = dt_ann['category_id']
    
                if dt_category != gt_category:
                    continue
    
                overlap = calculate_overlap(gt_box, dt_box)
    
                if overlap >= IoU_threshold:
                    correct_detection_count += 1
                    detected[i] = True
                    break  
        
    return gt_label_count, correct_detection_count


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_gt', type=str, default='', help='GT Json file path')
    parser.add_argument('--json_benign', type=str, default='', help='Benign Json file path')
    parser.add_argument('--json_attack', type=str, default='', help='Attack Json file path')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--image_suffix', type=str, default='.jpg', help='Attack Json file path')
    opt = parser.parse_args()
    print(vars(opt))
    return opt
    
    
def main(opt):
    cocoGt = COCO(opt.json_gt)
    
    filename_to_id = {img['file_name']: img['id'] for img in cocoGt.loadImgs(cocoGt.getImgIds())}
    
    #==========================================================================
    with open(opt.json_benign) as f:
        results = json.load(f)
    
    image_suffix = opt.image_suffix
    for ann in results:
        if 'coco_'+ann['image_id']+image_suffix in filename_to_id:
            ann['image_id'] = filename_to_id['coco_'+ann['image_id']+image_suffix]
        else:
            print(f"Warning: {ann['image_id']} not found in COCO dataset")
    
    with open('runs/converted_results.json', 'w') as f:
        json.dump(results, f)
        
    cocoDt_benign = cocoGt.loadRes('runs/converted_results.json')
    
    #==========================================================================
    with open(opt.json_attack) as f:
        results = json.load(f)
    
    for ann in results:
        if ann['image_id']+image_suffix in filename_to_id:
            ann['image_id'] = filename_to_id[ann['image_id']+image_suffix]
        else:
            print(f"Warning: {ann['image_id']} not found in COCO dataset")
    
    with open('runs/converted_results.json', 'w') as f:
        json.dump(results, f)
    
    cocoDt_attack = cocoGt.loadRes('runs/converted_results.json')
    
    
    #==========================================================================
    
    # AP computing
    #==========================================================================
    cocoEval_benign = COCOeval(cocoGt, cocoDt_benign, 'bbox')
    cocoEval_attack = COCOeval(cocoGt, cocoDt_attack, 'bbox')
    
    cocoEval_benign.params.imgIds = cocoGt.getImgIds()
    cocoEval_attack.params.imgIds = cocoGt.getImgIds()
    
    print()
    print('benign AP is: -----------------------------------------------------')
    cocoEval_benign.evaluate()
    cocoEval_benign.accumulate()
    cocoEval_benign.summarize()
    print()
    print('attack AP is: -----------------------------------------------------')
    cocoEval_attack.evaluate()
    cocoEval_attack.accumulate()
    cocoEval_attack.summarize()
    #==========================================================================
    
    conf_threshold = opt.conf_thres
    IoU_threshold = opt.iou_thres
    gt_number, tp_number_benign = calculate_tp(cocoGt, cocoDt_benign, conf_threshold, IoU_threshold)
    gt_number, tp_number_attack = calculate_tp(cocoGt, cocoDt_attack, conf_threshold, IoU_threshold)
    
    asr = (tp_number_benign - tp_number_attack) / tp_number_benign
    
    print()
    print('===================================================')
    print("Number of gt labels:", gt_number)
    print("Number of tp benign:", tp_number_benign)
    print("Number of tp attack:", tp_number_attack)
    print()
    print('ASR is ---> ', asr)
    print('===================================================')
    
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
