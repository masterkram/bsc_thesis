import torch
from torchvision.ops.boxes import _box_inter_union


def giou_loss(input_boxes, target_boxes, eps=1e-7):
    """
    Args:
        input_boxes: Tensor of shape (N, 4) or (4,).
        target_boxes: Tensor of shape (N, 4) or (4,).
        eps (float): small number to prevent division by zero
    """
    inter, union = _box_inter_union(input_boxes, target_boxes)
    iou = inter / union

    # area of the smallest enclosing box
    min_box = torch.min(input_boxes, target_boxes)
    max_box = torch.max(input_boxes, target_boxes)
    area_c = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])

    giou = iou - ((area_c - union) / (area_c + eps))

    loss = 1 - giou

    return loss.sum()
