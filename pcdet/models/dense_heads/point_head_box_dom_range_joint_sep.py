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

# Channel Attention
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(4096)

    def forward(self, x):
        y = self.conv_du(x)
        y = x * y + x
        y = y.view(y.shape[0], -1)
        y = self.bn(y)

        return y

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
        
class PointHeadBoxDomRangeJointSep(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, voxel_size=0, point_cloud_range=[], **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        
        self.num_keypoints_range = self.model_cfg.get('NUM_KEYPOINTS_RANGE', {})
        self.range_keys = self.num_keypoints_range.keys()

        target_cfg = self.model_cfg.TARGET_CONFIG

        self.domain_classifier_range = nn.ModuleDict()
        self.range_inv_classifier = nn.ModuleDict()

        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        self.se = self.model_cfg.get('SE', False)
        
        # print("input_channels", input_channels)

        # input_channels_joint = input_channels * len(self.range_keys)
        # print("input_channels_joint", input_channels_joint)

        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )

        self.selayer_range = nn.ModuleDict()
        self.emd_layer = nn.ModuleDict()

        for i in self.range_keys:
            if self.range_dom_inv:
                self.domain_classifier_range[i] = nn.Sequential(nn.Linear(self.num_keypoints_range[i], 1024), 
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(1024, 1024), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(1024, 1))

            if self.se:
                self.selayer_range[i] = SELayer(channel=640) # 640 dim

            
            # if self.range_inv:
            #     # 12, 23, 31 (skip)
            #     # if self.emd:
            #     #     self.emd_layer[i] = nn.Sequential(nn.Linear(640, 128), 
            #     #                                         nn.ReLU(True), nn.Dropout(),
            #     #                                         nn.Dropout(), nn.Linear(128, 8))
            #     if i in ['short', 'mid']:
            #         self.range_inv_classifier[i] = nn.Sequential(nn.Linear(self.num_keypoints_range[i], 1024), 
            #                                             nn.ReLU(True), nn.Dropout(),
            #                                             nn.Linear(1024, 1024), nn.ReLU(True),
            #                                             nn.Dropout(), nn.Linear(1024, 1))

            
                            

                # self.range_inv_classifier2[i] = nn.Sequential(nn.Linear(self.num_keypoints_range[i], 1024), 
                #                                     nn.ReLU(True), nn.Dropout(),
                #                                     nn.Linear(1024, 1024), nn.ReLU(True),
                #                                     nn.Dropout(), nn.Linear(1024, 1))

    def assign_targets(self, input_dict, dom_src=None, range_key=None, range_inv=False, joint=False):
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
        elif joint:
            suffix = '_joint'
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
            ret_part_labels=False, ret_box_labels=True, dom_src=dom_src, range_key=range_key, range_inv=range_inv, joint=joint
        )

        # print("targets_dict", targets_dict)

        return targets_dict

    def get_loss(self, tb_dict={}):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss(joint=True)
        point_loss_box, tb_dict_2 = self.get_box_layer_loss(joint=True)

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
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss(joint=True)
        point_loss_box, tb_dict_2 = self.get_box_layer_loss(joint=True)

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

        # interpolate_to_bev_features

        ret_dict = {}
        self.forward_ret_dict = {}

        self.emd_feat_dict = {}

        self.range_inv_dict = {}
        
        point_features_joint = []
        point_features_joint_comp = []
        point_coords_joint = []

        for i in self.range_keys:
            if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
                point_features = batch_dict[f'point_features_before_fusion_{i}']
            else:
                point_features = batch_dict[f'point_features_{i}']

            point_coords_i = batch_dict[f'point_coords_{i}']
            # print("point_coords_i", point_coords_i.shape)
            # point features attention

            if self.se:
                point_features = self.selayer_range[i](point_features, batch_dict['batch_size'])
            # print("point_features", point_features.shape)

            # self.emd_feat_dict[i] = point_features.view(batch_dict['batch_size'], self.num_keypoints_range[i], -1)
            # self.emd_layer[i](point_features).view(batch_dict['batch_size'], self.num_keypoints_range[i], -1)
            # print("emd_features", self.emd_feat_dict[i].shape)

            point_features_joint.append(point_features)
            point_features_joint_comp.append(batch_dict[f'point_features_{i}'])
            point_coords_joint.append(point_coords_i)

            # print(f"point_features_{i}", point_features.shape) # 1024* 640


            point_features_avg = torch.mean(point_features, -1)
            # print("point_features_avg", point_features_avg.shape)
            batch_point_features = point_features_avg.view(-1, self.num_keypoints_range[i])
        
            self.range_inv_dict[i] = batch_point_features
            self.emd_feat_dict[i] = batch_point_features

            if 'dom_img' in t_mode and self.range_dom_inv:

                x_reverse = grad_reverse(batch_point_features, l*-1)
                dom_point_preds = self.domain_classifier_range[i](x_reverse).squeeze(-1)

                self.forward_ret_dict[f'dom_point_preds_{i}'] = dom_point_preds            
                
                if self.training:
                    targets_dict_dom = self.assign_targets(batch_dict, dom_src=dom_src, range_key=i)

                    self.forward_ret_dict.update(targets_dict_dom)

        point_features_joint = torch.cat(point_features_joint)
        point_features_joint_small = torch.cat(point_features_joint_comp)
        batch_dict[f'point_coords_joint'] = torch.cat(point_coords_joint)  # (BxN, 4)
        batch_dict[f'point_features_joint'] = point_features_joint_small

        # print("point_features_joint", point_features_joint.shape)
    
        point_cls_preds = self.cls_layers(point_features_joint)  # (total_points, num_class)
        point_box_preds = self.box_layers(point_features_joint)  # (total_points, box_code_size)

        # print("point_cls_preds", point_cls_preds.shape)
        # print("point_box_preds", point_box_preds.shape)

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict[f'point_cls_scores_joint'] = torch.sigmoid(point_cls_preds_max)

        ret_dict[f'point_cls_preds_joint'] = point_cls_preds
        ret_dict[f'point_box_preds_joint'] = point_box_preds
                        

        if self.training:
            targets_dict = self.assign_targets(batch_dict, joint=True)
            # print("targets_dict out ", targets_dict)
            ret_dict[f'point_cls_labels'] = targets_dict[f'point_cls_labels_joint']
            ret_dict[f'point_box_labels'] = targets_dict[f'point_box_labels_joint']

            self.forward_ret_dict.update(ret_dict)

            ################# range ####################
        
        if self.range_inv:
            if self.emd:

                for i in range(len(self.range_keys)-1): #[1,2]#,3

                    key1 = list(self.range_keys)[i]
                    key2 = list(self.range_keys)[i+1]

                    self.forward_ret_dict[f'emd_feat1_{key1}'] = self.emd_feat_dict[key1]   
                    self.forward_ret_dict[f'emd_feat2_{key1}'] = self.emd_feat_dict[key2]           

            else:
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
