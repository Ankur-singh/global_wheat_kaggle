import re
import ast
import numpy as np
import pandas as pd

import numba
from numba import jit

from typing import List, Union, Tuple
from torch.utils.data import DataLoader

from utils import DatasetRetriever

## Don't touch
@jit(nopython=True)
def calculate_iou(gt, pr, form='coco') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    if dx < 0: return 0.0

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1
    if dy < 0: return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = ((gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
                  (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
                  overlap_area)
    return overlap_area / union_area

@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

@jit(nopython=True)
def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision

def get_data_loader(opt):
  # reading from the git repo
  df_folds = pd.read_csv(opt.folds)  
  marking = pd.read_csv(opt.train)

  # formating the bboxes
  bboxs = np.stack(marking['bbox'].apply(lambda x: ast.literal_eval(x)))
  for i, column in enumerate(['x', 'y', 'w', 'h']):
      marking[column] = bboxs[:,i]
  marking.drop(columns=['bbox'], inplace=True)

  dataset = DatasetRetriever(image_ids=df_folds[df_folds['fold'] == opt.fold].image_id.values, 
                             path=opt.path, marking=marking, transforms=get_transforms(img_sz))

  data_loader = DataLoader(dataset, batch_size=2,
                          pin_memory=False, shuffle=False,
                          num_workers=2, collate_fn=collate_fn)  
  return data_loader

def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1
    config.image_size=img_sz
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if 'model_state_dict' in checkpoint.keys():
        net.load_state_dict(checkpoint['model_state_dict']) # model 2 & 3
    else:
        net.load_state_dict(checkpoint)  # model 0 & 1

    del checkpoint
    gc.collect()

    net = DetBenchEval(net, config)
    net.eval()
    return net.cuda()

def make_predictions(net, images):
    images = torch.stack(images).cuda().float()
    predictions = []
    with torch.no_grad():
        det = net(images, torch.tensor([1]*images.shape[0]).float().cuda())
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            boxes = (boxes*(1024/img_sz)).astype(np.int32).clip(min=0, max=1023)
            predictions.append({'boxes': boxes, 'scores': scores})
    return predictions  

def run_wbf(boxes, scores, image_size=img_sz, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    labels = [np.ones(len(score)).tolist() for score in scores]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def get_all(net, data_loader):
    pd_bboxes = []
    pd_scores = []
    gt_bboxes = []
    for images, targets, image_ids in data_loader:
        predictions = make_predictions(net, images)
        for pred in predictions:
            pd_bboxes.append(pred['boxes'])
            pd_scores.append(pred['scores'])
        for targ in targets:
            gt_bboxes.append(targ['boxes'].numpy())

    pd_bboxes, pd_scores, gt_bboxes = map(np.array, [pd_bboxes, pd_scores, gt_bboxes])
    return pd_bboxes, pd_scores, gt_bboxes

def evalutate(net, data_loader, th=0.25):
    validation_image_precisions = []
    iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]

    pd_bboxes, pd_scores, gt_bboxes = get_all(net, data_loader)
    assert len(pd_bboxes) == len(pd_scores) == len(gt_bboxes), "You surely did something wrong!"

    for bbox, score, gt_bbox in zip(pd_bboxes, pd_scores, gt_bboxes):
    # TODO: apply thresholding
    index = score > th
    
    preds    = bbox[index]    # shape: (#predicted box, 4)
    scores   = score[index]   # shape: (#predicted box, )
    gt_boxes = gt_bbox # shape: (#ground-truth box, 4)

    preds_sorted_idx = np.argsort(scores)[::-1]
    preds_sorted = preds[preds_sorted_idx]

    for idx, image in enumerate(images):
        image_precision = calculate_image_precision(preds_sorted, gt_boxes, thresholds=iou_thresholds, form='coco')
        validation_image_precisions.append(image_precision)

    print(f"Validation IOU (for {th}): {np.mean(validation_image_precisions):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='fold to evaluate')
    parser.add_argument('--path', type=str, default='.', help='base_directory path where you have all data downloaded from kaggle')
    parser.add_argument('--train' , type=str, default='data/train.csv', help='train.csv path')
    parser.add_argument('--folds' , type=str, default='train_folds.csv', help='folds.csv path')
    parser.add_argument('--weights', type=str, help='checkpoint.pt path')
    opt = parser.parse_args()
    
    data_loader = get_data_loader(opt)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_net(opt.weights)
    
    evalutate(net, data_loader)