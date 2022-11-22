import numpy as np

#Classes relabelled {255,0,1,...,15}.
#Predictions will all be in the set {0,1,...,15}

CLASS_LABELS = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 
    'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]
UNKNOWN_ID = 255
N_CLASSES = len(CLASS_LABELS)

def confusion_matrix(pred_ids, gt_ids):
    
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = (gt_ids < 100)

    return np.bincount(pred_ids[idxs] * 16 + gt_ids[idxs], minlength=256).reshape((16, 16)).astype(np.int32)

def get_iou(label_id, confusion):
    
    # true positives
    tp = np.int32(confusion[label_id, label_id])
    # false positives
    fp = np.int32(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.int32(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return (float(tp) / denom, tp, denom)

def evaluate(pred_ids=None, gt_ids=None, confusion=None):
    if confusion is None:
        print('evaluating', gt_ids.size, 'points...')
        confusion = confusion_matrix(pred_ids, gt_ids)
    class_ious = {}
    mean_iou = 0

    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        class_ious[label_name] = get_iou(i, confusion)
        mean_iou += class_ious[label_name][0] / 16

    print('classes          IoU')
    print('----------------------------')
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
    print('mean IOU', mean_iou)
