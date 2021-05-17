import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


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
        input = input.view(batch_size, -1, dim) # 4, 1024, 640
        # input = input.view(batch_size, dim, -1) # 4, 640, 1024
        input = input.permute(0, 2, 1)#.contiguous() # 4, 640, 1024
        x = input.mean(dim=-1) # 4 * 640
        # print('re',x.shape)
        # y = self.avg_pool(x).view(dim, 1) # 640 * 1
        y = self.fc(x) # 4, 640
        # print('y1', y.shape) # 4, 640
        # y = y.view(batch_size, dim, 1)  # 4, 640, 1
        y = y.unsqueeze(-1)
        ret = input * y.expand_as(input) # 2, 640, 1024
        ret = ret.permute(0, 2, 1)
        ret = ret.view(-1, dim)
        return ret

class PointSELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(PointSELayer, self).__init__()
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
        input = input.view(batch_size, -1, dim) # 4, 1024, 640
        # input = torch.transpose(input, -1, -2)
        x = input.mean(dim=-1) # 4 * 1024
        # print('re',x.shape)
        y = self.fc(x) # 4, 1024->4, 1024
        y = y.unsqueeze(-1)  # 4, 1024, 1
        ret = input * y.expand_as(input) # 4, 1024, 640
        # print("ret", ret.shape)
        ret = ret.view(-1, dim) # 4*1024, 640
        return ret

class PointHeadBoxLocalDom(PointHeadTemplate):
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

        self.dom_cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.LOCAL_DOM_FC,
            input_channels=input_channels,
            output_channels=1
        )

        self.domain_classifier = nn.Sequential(nn.Linear(self.num_keypoints, 1024), 
                                               nn.ReLU(True), nn.Dropout(),
                                               nn.Linear(1024, 256), nn.ReLU(True),
                                               nn.Dropout(), nn.Linear(256, 1))

        self.attention_point_feature_fusion = nn.Sequential(
            nn.Linear(self.model_cfg.get('NUM_IN_FEATURES', 640), self.model_cfg.get('NUM_FUSION_FEATURES', 128), bias=False),
            nn.BatchNorm1d(self.model_cfg.get('NUM_FUSION_FEATURES', 128)),
            nn.ReLU(),
        )

        self.se = self.model_cfg.get('SE', False)
        self.point_se = self.model_cfg.get('POINT_SE', False)

        if self.se:
            self.selayer = SELayer(channel=640) # 640 dim
        if self.point_se:
            self.point_selayer = PointSELayer(channel=2048) # 640 dim

        # self.domain_classifier_conv = nn.Conv1d(nn.Linear(self.num_keypoints, 1024), 
        #                                        nn.ReLU(True), nn.Dropout(),
        #                                        nn.Linear(1024, 1024), nn.ReLU(True),
        #                                        nn.Dropout(), nn.Linear(1024, 1))


        # self.domain_classifier_point = nn.Sequential(nn.Linear(128, 1024), 
        #                                        nn.ReLU(True), nn.Dropout(),
        #                                        nn.Linear(1024, 1024), nn.ReLU(True),
        #                                        nn.Dropout(), nn.Linear(1024, 1))

        
    def assign_targets(self, input_dict, dom_src=None, localdom=True):
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
            ret_part_labels=False, ret_box_labels=True, dom_src=dom_src, localdom=localdom
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

    def entropy(self, pred, lamda=1.0):

        # out_t1 = F1(feat, reverse=True, eta=-eta)
        # print("pred", pred)
        prob = torch.sigmoid(pred)
        # print("prob",prob)
        # print("prob *torch.log(prob + 1e-5)", prob * torch.log(prob + 1e-5))
        weight_ent = -lamda * prob *(torch.log(prob + 1e-5))
        # print("weight_ent", weight_ent)
        return weight_ent.unsqueeze(-1)

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
        self.forward_ret_dict = {}

        t_mode = batch_dict['t_mode']
        l = batch_dict['l']

        if t_mode == 'dom_img_src':
            dom_src = True
        elif t_mode == 'dom_img_tgt':
            dom_src = False
        else:
            dom_src = None

        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        # else:
        #     point_features = batch_dict['point_features']

        if self.se and self.point_se:
            point_features_se = self.selayer(point_features, batch_dict['batch_size']) # bt*2048, 640
            point_features_point_se = self.point_selayer(point_features, batch_dict['batch_size']) # bt*2048, 640
            point_features = point_features_se + point_features_point_se
        elif self.se:
            point_features = self.selayer(point_features, batch_dict['batch_size']) # bt, 2048, 640
        elif self.point_se:
            point_features = self.point_selayer(point_features, batch_dict['batch_size'])


        # print("point_features", point_features.shape)
        # point_features_attention = self.attention_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))
        # batch_dict['point_features'] = point_features_attention
            
        ###### dom part ############
        # print("tmode", t_mode)
        if 'dom_img' in t_mode:
            if self.pos_da:
                x_reverse = grad_reverse(point_features, l)
            else:
                x_reverse = grad_reverse(point_features, l*-1)

            dom_point_context = self.dom_cls_layers[:-2](x_reverse).squeeze(-1)
            # print("dom_point_context", dom_point_context.shape)
            # if 'dom_img_det' in t_mode:
            dom_point_context_avg = dom_point_context.view(batch_dict['batch_size'], -1, dom_point_context.shape[-1])
            
            # print("dom_point_context_avg", dom_point_context_avg.shape)
            dom_point_context_avg = dom_point_context_avg.mean(dim=1)
            # print("dom_point_context_avg", dom_point_context_avg.shape)

            batch_dict['dom_point_context'] = dom_point_context_avg #localdom
            # 1, 256
            dom_point_preds = self.dom_cls_layers[-2:](dom_point_context).squeeze(-1)
            # [1]
            # print("dom_point_preds", dom_point_preds.shape)

            self.forward_ret_dict['dom_point_preds'] = dom_point_preds            

            if self.training:
                targets_dict_dom = self.assign_targets(batch_dict, dom_src=dom_src, localdom=True)

                self.forward_ret_dict.update(targets_dict_dom)

            if self.dom_attention:
                weight_dom_uncertainty = self.entropy(dom_point_preds)
                # print("point_features ori", point_features.shape) 
                point_features = point_features * (1+weight_dom_uncertainty)

                point_features = point_features.permute(0,1)

                # print("point_features aft", point_features)
            elif t_mode != 'dom_img_det':
                return batch_dict

        ###### det part ############

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

        self.forward_ret_dict.update(ret_dict)

        return batch_dict
