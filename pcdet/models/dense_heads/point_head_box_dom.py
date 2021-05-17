import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate
import torch.nn as nn

class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)

# Squeeze Excitation 
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input, batch_size):
        # bt_points, dim = input.size() # 4096(4*1024), 640
        dim = input.size()[-1]
        # input = input.view(batch_size, -1, dim) # 4, 1024, 640
        input = input.view(batch_size, dim, -1) # 4, 640, 1024
        x = input.mean(dim=-1) # 4 * 640
        # print('re',x.shape)
        # y = self.avg_pool(x).view(dim, 1) # 640 * 1
        y = self.fc(x) # 4, 640
        # print('y1', y.shape) # 4, 640
        y = y.view(batch_size, dim, 1)  # 4, 640, 1
        ret = input * y.expand_as(input) # 4, 640, 1024
        # print("ret", ret)
        ret = ret.view(-1, dim)
        return ret

class PointHeadBoxDom(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )
        self.num_keypoints = self.model_cfg.NUM_KEYPOINTS
        self.domain_classifier = nn.Sequential(nn.Linear(self.num_keypoints, 1024), 
                                               nn.ReLU(True), nn.Dropout(),
                                               nn.Linear(1024, 1024), nn.ReLU(True),
                                               nn.Dropout(), nn.Linear(1024, 1))

        self.se = self.model_cfg.get('SE', False)
        # self.domain_classifier_conv = nn.Conv1d(nn.Linear(self.num_keypoints, 1024), 
        #                                        nn.ReLU(True), nn.Dropout(),
        #                                        nn.Linear(1024, 1024), nn.ReLU(True),
        #                                        nn.Dropout(), nn.Linear(1024, 1))


        # self.domain_classifier_point = nn.Sequential(nn.Linear(128, 1024), 
        #                                        nn.ReLU(True), nn.Dropout(),
        #                                        nn.Linear(1024, 1024), nn.ReLU(True),
        #                                        nn.Dropout(), nn.Linear(1024, 1))

    def assign_targets(self, input_dict, dom_src=None):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True, dom_src=dom_src
        )

        return targets_dict

    def get_loss(self, tb_dict={}):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return point_loss, tb_dict

    def get_dom_loss(self, tb_dict={}):
        tb_dict = {} if tb_dict is None else tb_dict
        dom_loss, tb_dict_1 = self.get_dom_layer_loss()

        tb_dict.update(tb_dict_1)
        return dom_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        t_mode = batch_dict['t_mode']
        l = batch_dict['l']

        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size)

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)

        ret_dict = {'point_cls_preds': point_cls_preds,
                    'point_box_preds': point_box_preds}
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            )
            batch_dict['batch_cls_preds_point'] = point_cls_preds
            batch_dict['batch_box_preds_point'] = point_box_preds
            batch_dict['batch_index_point'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized_point'] = False

        self.forward_ret_dict = ret_dict

        if 'dom_img' in t_mode:
            if t_mode == 'dom_img_src':
                dom_src = True
            elif t_mode == 'dom_img_tgt':
                dom_src = False
            # else:
            #     dom_src = None

            # print("point_features", point_features.shape)
            point_features_avg = torch.mean(point_features, -1)
            # print("point_features_avg", point_features_avg.shape)
            batch_point_features = point_features_avg.view(-1, self.num_keypoints)
            # print("batch_point_features", batch_point_features.shape)

            # x_pool = self.domain_pool(batch_point_features).view(point_features.size(0), -1)
            # print("x_pool", x_pool.shape)

            x_reverse = grad_reverse(batch_point_features, l*-1)
            dom_point_preds = self.domain_classifier(x_reverse).squeeze(-1)

            self.forward_ret_dict['dom_point_preds'] = dom_point_preds            
            

            if self.training:
                targets_dict_dom = self.assign_targets(batch_dict, dom_src=dom_src)

                self.forward_ret_dict.update(targets_dict_dom)


        return batch_dict
