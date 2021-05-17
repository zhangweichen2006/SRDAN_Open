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


class PointHeadBoxDomRange(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        
        self.num_keypoints_range = self.model_cfg.get('NUM_KEYPOINTS_RANGE', {})
        self.range_keys = self.num_keypoints_range.keys()

        target_cfg = self.model_cfg.TARGET_CONFIG

        self.cls_layers_range = nn.ModuleDict()
        self.box_layers_range = nn.ModuleDict()
        self.domain_classifier_range = nn.ModuleDict()
        self.range_inv_classifier = nn.ModuleDict()

        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        for i in self.range_keys:
            
            self.cls_layers_range[i] = self.make_fc_layers(
                fc_cfg=self.model_cfg.CLS_FC,
                input_channels=input_channels,
                output_channels=num_class
            )
            self.box_layers_range[i] = self.make_fc_layers(
                fc_cfg=self.model_cfg.REG_FC,
                input_channels=input_channels,
                output_channels=self.box_coder.code_size
            )
            # self.num_keypoints_range[i] = self.model_cfg.NUM_KEYPOINTS_RANGE[i]
            # self.= nn.AvgPool1d(kernel_size=input_channels)
            if self.range_dom_inv:
                self.domain_classifier_range[i] = nn.Sequential(nn.Linear(self.num_keypoints_range[i], 1024), 
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(1024, 1024), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(1024, 1))

            if self.range_inv and i in ['short', 'mid']:
                # 12, 23, 31 (skip)
                self.range_inv_classifier[i] = nn.Sequential(nn.Linear(self.num_keypoints_range[i], 1024), 
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(1024, 1024), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(1024, 1))
                            

                # self.range_inv_classifier2[i] = nn.Sequential(nn.Linear(self.num_keypoints_range[i], 1024), 
                #                                     nn.ReLU(True), nn.Dropout(),
                #                                     nn.Linear(1024, 1024), nn.ReLU(True),
                #                                     nn.Dropout(), nn.Linear(1024, 1))

    def assign_targets(self, input_dict, dom_src=None, range_key=None, range_inv=False):
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
        if range_key is not None:
            suffix = f'_{range_key}' 
        else:
            suffix = ''
        
        # if range_key2 is not None:
        #     suffix2 = f'_{range_key2}' 
        #     point_coords2 = input_dict[f'point_coords{suffix2}']
        # else:
        #     suffix2 = ''

        # for i in self.range_keys:
        point_coords = input_dict[f'point_coords{suffix}']
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
            ret_part_labels=False, ret_box_labels=True, dom_src=dom_src, range_key=range_key, range_inv=range_inv
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
    
    def get_range_loss(self, tb_dict={}):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_range_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_range_box_layer_loss()

        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)

        if self.range_inv:
            point_loss_range, tb_dict_3 = self.get_range_inv_layer_loss()
            tb_dict.update(tb_dict_3)

            point_loss = point_loss_cls + point_loss_box + point_loss_range
        
            return point_loss, tb_dict

        point_loss = point_loss_cls + point_loss_box

        return point_loss, tb_dict

    def get_dom_range_loss(self, tb_dict={}):
        tb_dict = {} if tb_dict is None else tb_dict
        dom_loss, tb_dict_1 = self.get_range_dom_layer_loss()

        tb_dict.update(tb_dict_1)
        return dom_loss, tb_dict

    def get_range_inv_loss(self, tb_dict={}):
        tb_dict = {} if tb_dict is None else tb_dict

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

        if t_mode == 'dom_img_src':
            dom_src = True
        elif t_mode == 'dom_img_tgt':
            dom_src = False
        else:
            dom_src = None

        ret_dict = {}
        self.forward_ret_dict = {}

        self.range_inv_dict = {}
        
        for i in self.range_keys:
            if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
                point_features = batch_dict[f'point_features_before_fusion_{i}']
            else:
                point_features = batch_dict[f'point_features_{i}']

            point_cls_preds = self.cls_layers_range[i](point_features)  # (total_points, num_class)
            point_box_preds = self.box_layers_range[i](point_features)  # (total_points, box_code_size)

            point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
            batch_dict[f'point_cls_scores_{i}'] = torch.sigmoid(point_cls_preds_max)

            ret_dict[f'point_cls_preds_{i}'] = point_cls_preds
            ret_dict[f'point_box_preds_{i}'] = point_box_preds
                            
            if self.training:
                targets_dict = self.assign_targets(batch_dict, range_key=i)
                ret_dict[f'point_cls_labels_{i}'] = targets_dict[f'point_cls_labels_{i}']
                ret_dict[f'point_box_labels_{i}'] = targets_dict[f'point_box_labels_{i}']

            # if not self.training or self.predict_boxes_when_training:
            #     point_cls_preds, point_box_preds = self.generate_predicted_boxes(
            #         points=batch_dict[f'point_coords_{i}'][:, 1:4],
            #         point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            #     )
            #     batch_dict[f'batch_cls_preds_point_{i}'] = point_cls_preds
            #     batch_dict[f'batch_box_preds_point_{i}'] = point_box_preds
            #     batch_dict[f'batch_index_point_{i}'] = batch_dict[f'point_coords_{i}'][:, 0]
            #     batch_dict[f'cls_preds_normalized_point_{i}'] = False

            self.forward_ret_dict.update(ret_dict)

            point_features_avg = torch.mean(point_features, -1)
            # print("point_features_avg", point_features_avg.shape)
            batch_point_features = point_features_avg.view(-1, self.num_keypoints_range[i])

            self.range_inv_dict[i] = batch_point_features

            if 'dom_img' in t_mode:

                x_reverse = grad_reverse(batch_point_features, l*-1)
                dom_point_preds = self.domain_classifier_range[i](x_reverse).squeeze(-1)

                self.forward_ret_dict[f'dom_point_preds_{i}'] = dom_point_preds            
                
                if self.training:
                    targets_dict_dom = self.assign_targets(batch_dict, dom_src=dom_src, range_key=i)

                    self.forward_ret_dict.update(targets_dict_dom)

            ################# range ####################
        if self.range_inv:
            targets_dict_range_all = {}

            for i in range(len(self.range_keys)-1): #[1,2]#,3

                key1 = list(self.range_keys)[i]
                key2 = list(self.range_keys)[i+1]

                x_reverse1 = grad_reverse(self.range_inv_dict[key1], l*-1)
                x_reverse2 = grad_reverse(self.range_inv_dict[key2], l*-1)

                range_point_preds1 = self.range_inv_classifier[key1](x_reverse1).squeeze(-1)
                range_point_preds2 = self.range_inv_classifier[key1](x_reverse2).squeeze(-1)

                self.forward_ret_dict[f'range_point_preds1_{key1}'] = range_point_preds1   
                self.forward_ret_dict[f'range_point_preds2_{key1}'] = range_point_preds2            
                
                if self.training:
                    targets_dict_range = self.assign_targets(batch_dict, range_key=key1, range_inv=self.range_inv)
                    targets_dict_range_all.update(targets_dict_range)
                    
            self.forward_ret_dict.update(targets_dict_range_all)

        return batch_dict
