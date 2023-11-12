import os
import numpy as np

def read_label_file(file_path):
    """
    Read a label file and return the coordinates as a list.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        labels = [list(map(int, line.strip().split(' '))) for line in lines]
    return labels

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Calculate intersection coordinates
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    # Calculate area of intersection
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Calculate each box area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area

    return iou


def evaluate_predictions(label_dir, prediction_dir, iou_threshold=0.5):
    files = os.listdir(label_dir)
    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []  # Store all IoUs for calculating mean IoU

    for file in files:
        label_path = os.path.join(label_dir, file)
        prediction_path = os.path.join(prediction_dir, file)

        if not os.path.exists(prediction_path):
            continue

        true_box = read_label_file(label_path)[0]  # Assuming only one true box per image
        predicted_boxes = read_label_file(prediction_path)

        match_found = False
        for pred_box in predicted_boxes:
            iou = calculate_iou(pred_box, true_box)
            all_ious.append(iou)  # Collect IoU for all predictions

            if iou >= iou_threshold and not match_found:
                total_tp += 1
                match_found = True
            elif iou < iou_threshold:
                total_fp += 1

        if not predicted_boxes:
            total_fn += 1

    # Calculate precision, recall, F1-score
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate mean IoU for all predictions
    mean_iou = np.mean(all_ious) if all_ious else 0

    return precision, recall, f1_score, mean_iou

# Example usage
# pred_dir = 'path/to/prediction/directory'
# label_dir = 'path/to/label/directory'
# precision, recall, f1_score, mean_iou = evaluate_predictions(label_dir, pred_dir)
# print("Precision:", precision, "Recall:", recall, "F1 Score:", f1_score, "Mean IoU:", mean_iou)


# Example usage
pred_dir = '/home/ubuntu/workspace/hy/dataset/DAGM/result/predict_gpp_234/label/'
label_dir = '/home/ubuntu/workspace/hy/dataset/DAGM/label/val/'
precision, recall, f1_score, mean_iou = evaluate_predictions(label_dir, pred_dir)

# 将它们四舍五入到四位小数
precision = round(precision, 4)
recall = round(recall, 4)
f1_score = round(f1_score, 4)
mean_iou = round(mean_iou, 4)

print("miou:", mean_iou, "\nR:", recall, "\nP:", precision, "\nF1 Score:", f1_score)
