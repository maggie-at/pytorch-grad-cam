import os
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_iou(boxA, boxB):
    interArea = max(0, min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])) * max(0, min(boxA[3], boxB[3]) - max(boxA[1], boxB[1]))
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def read_truth_annotation(file):
    # 读取标注
    with open(file, 'r') as f:
        boxes = []
        lines = f.readlines()
        for line in lines:
            box = list(map(float, line.strip().split(' ')[1:]))
            boxes.append(box)
        return boxes


def read_pred_annotation(file):
    # 读取标注
    with open(file, 'r') as f:
        boxes = []
        lines = f.readlines()
        for line in lines:
            box = list(map(float, line.strip().split(' ')[0:]))
            boxes.append(box)
        return boxes

def calculate_metrics(pred_dir, label_dir, iou_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0
    ious = []
    miou = 0
    for cls_folder in os.listdir(pred_dir):
        for filename in os.listdir(os.path.join(pred_dir, cls_folder, "label/")):
            pred_file = os.path.join(pred_dir, cls_folder, "label/", filename)
            label_file = os.path.join(label_dir, filename)
            if os.path.isfile(pred_file) and os.path.isfile(label_file):
                pred_boxes = read_pred_annotation(pred_file)
                label_boxes = read_truth_annotation(label_file)
                if pred_boxes and label_boxes:
                    ious_individual = []
                    for pred_box in pred_boxes:
                        ious_individual.append(max([compute_iou(pred_box, label_box) for label_box in label_boxes]))

                    ious.extend(ious_individual)
                    miou += np.mean(ious_individual)
                    TP += sum(iou >= iou_threshold for iou in ious_individual)
                    FP += sum(iou < iou_threshold for iou in ious_individual)
                else:
                    FN += len(label_boxes)  # 每个未检测到的目标框都增加 FN
        if len(ious) > 0:
            miou = miou / len(ious)
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    return miou, recall, precision

predictions_dir = "/home/ubuntu/workspace/hy/dataset/NEU-DET/predict_grad/"
ground_truth_dir = "/home/ubuntu/workspace/hy/dataset/NEU-DET/label/"

avg_metrics = calculate_metrics(predictions_dir, ground_truth_dir)
print(avg_metrics)