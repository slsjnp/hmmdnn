import torch
import torch.nn.functional as F
from .misc import flatten
from skimage.metrics import peak_signal_noise_ratio


# segmentation metrics
def seg_metrics(y_pred, y_true, smooth=1e-7, isTrain=True, threshold=0.5):
    # comment out if your model contains a sigmoid or equivalent activation layer
    if isTrain:
        y_pred = torch.sigmoid(y_pred)

    if not isTrain:
        # y_pred = torch.where(y_pred < threshold, y_pred, torch.ones(1).cuda())
        # y_pred = torch.where(y_pred >= threshold, y_pred, torch.zeros(1).cuda())
        y_pred = torch.where(y_pred < threshold, y_pred, torch.ones(1).cuda())
        y_pred = torch.where(y_pred >= threshold, y_pred, torch.zeros(1).cuda())
    # flatten label and prediction tensors
    y_pred = flatten(y_pred)
    y_true = flatten(y_true)

    tp = (y_true * y_pred).sum(-1)
    fp = ((1 - y_true) * y_pred).sum(-1)
    fn = (y_true * (1 - y_pred)).sum(-1)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    iou = (tp + smooth) / (tp + fn + fp + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    mean_precision = precision.mean()
    mean_recall = recall.mean()
    mean_iou = iou.mean()
    mean_f1 = f1.mean()

    # for training, ouput mean metrics
    if isTrain:
        return mean_precision.item(), mean_recall.item(), mean_iou.item(), mean_f1.item()
    # for testing, output metrics array by class with threshold
    else:
        return precision.detach().cpu().numpy(), recall.detach().cpu().numpy(), iou.detach().cpu().numpy(), f1.detach().cpu().numpy()