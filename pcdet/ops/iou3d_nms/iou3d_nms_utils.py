"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch

from ...utils import common_utils
from . import iou3d_nms_cuda


def boxes_bev_iou_cpu(boxes_a, boxes_b, nusc=False):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_a, is_numpy = common_utils.check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = common_utils.check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    if nusc:
        assert boxes_a.shape[1] == boxes_b.shape[1] == 9
    else:
        assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    
    # if nusc:
    #     boxes_a = boxes_a.contiguous()[:,[0,1,2,3,4,5,8]]
    #     boxes_b = boxes_b.contiguous()[:,[0,1,2,3,4,5,8]]
    # else:
    boxes_a = boxes_a.contiguous()
    boxes_b = boxes_b.contiguous()

    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    if nusc:
        iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a, boxes_b, ans_iou)
    else:
        iou3d_nms_cuda.boxes_iou_bev_cpu_nusc(boxes_a, boxes_b, ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou



def boxes_iou_bev(boxes_a, boxes_b, nusc=False):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    if nusc:
        assert boxes_a.shape[1] == boxes_b.shape[1] == 9
    else:
        assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    
    # if nusc:
    #     boxes_a = boxes_a.contiguous()[:,[0,1,2,3,4,5,8]]
    #     boxes_b = boxes_b.contiguous()[:,[0,1,2,3,4,5,8]]
    # else:
    boxes_a = boxes_a.contiguous()
    boxes_b = boxes_b.contiguous()

    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a, boxes_b, ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b, nusc=False):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """

    if nusc:
        assert boxes_a.shape[1] == boxes_b.shape[1] == 9
    else:
        assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # if nusc:
    #     boxes_a = boxes_a.contiguous()[:,[0,1,2,3,4,5,8]]
    #     boxes_b = boxes_b.contiguous()[:,[0,1,2,3,4,5,8]]
    # else:
    boxes_a = boxes_a.contiguous()
    boxes_b = boxes_b.contiguous()

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a, boxes_b, overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, nusc=False, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    if nusc:
        assert boxes.shape[1] == 9
    else:
        assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    # if nusc:
    #     boxes = boxes[:,[0,1,2,3,4,5,8]]
    # else:
    #     boxes = boxes.contiguous()

    keep = torch.LongTensor(boxes.size(0))
    # if nusc:
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    # else:
    #     num_out = iou3d_nms_cuda.nms_gpu_nusc(boxes, keep, thresh)

    return order[keep[:num_out].cuda()].contiguous(), None


def nms_normal_gpu(boxes, scores, thresh, nusc=False, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    if nusc:
        assert boxes.shape[1] == 9
    else:
        assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()
    # if nusc:
    #     boxes = boxes[:,[0,1,2,3,4,5,8]]
    # else:
    #     boxes = boxes.contiguous()

    keep = torch.LongTensor(boxes.size(0))

    # if nusc:
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    # else:
    #     num_out = iou3d_nms_cuda.nms_normal_gpu_nusc(boxes, keep, thresh)

    return order[keep[:num_out].cuda()].contiguous(), None