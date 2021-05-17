import numpy as np
import torch
import torch.nn as nn
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
import torch.nn.functional as F
from ...utils import box_coder_utils, loss_utils, common_utils


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training, nusc=False, num_fpn_up=0, num_fpn_down=0, num_fpn_downup=0, fpn_layers=[], fpn_only=False, voxel_size=[0.1, 0.1, 0.2]):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTI_HEAD', False)
        self.nusc = nusc
        self.num_fpn_up = num_fpn_up
        self.num_fpn_down = num_fpn_down
        self.num_fpn_downup = num_fpn_downup
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        # print("gridz_size", grid_size)
        # print("voxel_size", voxel_size)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6), nusc=self.nusc
        )
        self.fpn_only = self.model_cfg.get('FPN_ONLY', False)
        self.dom_squeeze = self.model_cfg.get('DOM_SQUEEZE', True)

        self.cross_scale = self.model_cfg.get('CROSS_SCALE', False)
        self.cross_two_scale = self.model_cfg.get('CROSS_TWO_SCALE', False)

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG

        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range, nusc=self.nusc
        )
        self.anchors = [x.cuda() for x in anchors]
        self.mcd = self.model_cfg.get('MCD', False)
        self.mcd_mid = self.model_cfg.get('MCD_MID', False)
        self.mcd_context = self.model_cfg.get('MCD_CONTEXT', False)
        self.point_interpolation = self.model_cfg.get('POINT_INTERPOLATION', False)
        self.fast_interpolation = self.model_cfg.get('FAST_INTERPOLATION', True)
        self.point_features_dim = self.model_cfg.get('POINT_FEATURES_DIM', 128)
        self.patch_da = self.model_cfg.get('PATCH_DA', False)
        self.dom_cat_point = self.model_cfg.get('DOM_CAT_POINT', False)
        self.two_dom_weight = self.model_cfg.get('TWO_DOM_WEIGHT', False)

        self.debug = self.model_cfg.get('DEBUG', False)

        self.range_da = self.model_cfg.get('RANGE_DA', 0)
        self.interval_da = self.model_cfg.get('INTERVAL_DA', 0)

        self.remove_near_range = self.model_cfg.get('REMOVE_NEAR_RANGE', 0)
        self.remove_far_range = self.model_cfg.get('REMOVE_FAR_RANGE', 0)
        self.remove_near_range2 = self.model_cfg.get('REMOVE_NEAR_RANGE2', 0)
        self.remove_far_range2 = self.model_cfg.get('REMOVE_FAR_RANGE2', 0)

        self.cross_scale = self.model_cfg.get('CROSS_SCALE', False)
        self.cross_two_scale = self.model_cfg.get('CROSS_TWO_SCALE', False)

        self.range_guidance = self.model_cfg.get('RANGE_GUIDANCE',False)
        self.index_range_guidance = self.model_cfg.get('INDEX_RANGE_GUIDANCE', False)
        self.range_guidance_dom_only = self.model_cfg.get('RANGE_GUIDANCE_DOM_ONLY', True)

        self.range_guidance_conv_dom = self.model_cfg.get('RANGE_GUIDANCE_CONV_DOM', False)
        self.range_guidance_new_conv_dom = self.model_cfg.get('RANGE_GUIDANCE_NEW_CONV_DOM', False)
        self.range_guidance_new_conv_dom_attention = self.model_cfg.get('RANGE_GUIDANCE_NEW_CONV_DOM_ATTENTION', False)
        self.range_guidance_new_conv_dom_context = self.model_cfg.get('RANGE_GUIDANCE_NEW_CONV_DOM_CONTEXT', False)
        self.range_guidance_double_dom = self.model_cfg.get('RANGE_GUIDANCE_DOUBLE_DOM', False)
        self.sep_two_dom = self.model_cfg.get('SEP_TWO_DOM', False)

        self.joint_two_dom = self.model_cfg.get('JOINT_TWO_DOM', False)

        self.diff_dom_opt = self.model_cfg.get('DIFF_DOM_OPT', False)

        self.dom_context = self.model_cfg.get('DOM_CONTEXT', False)
        self.point_feat_in_voxel_dom = self.model_cfg.get('POINT_FEAT_IN_VOXEL_DOM', False)

        self.multi_range_interpolate = self.model_cfg.get('MULTI_RANGE_INTERPOLATE', False)

        self.two_dom_reg = self.model_cfg.get('TWO_DOM_REG', False)
        self.rev_reg = self.model_cfg.get('REV_REG', False)

        self.sep_fpn_dom_context = self.model_cfg.get('SEP_FPN_DOM_CONTEXT', False)
        self.patch_unplug_context = self.model_cfg.get('PATCH_UNPLUG_CONTEXT', False)
        self.two_attention_max = self.model_cfg.get('TWO_ATTENTION_MAX', False)

        self.double_pma = self.model_cfg.get('DOUBLE_PMA', False)
        self.joint_pma = self.model_cfg.get('JOINT_PMA', False)

        self.range_guidance_dist = self.model_cfg.get('RANGE_GUIDANCE_DIST', False)

        self.fov = self.model_cfg.get('FOV', False)

        # print("self.num_anchors_per_location", self.num_anchors_per_location) [2]
        # print("self.num_fpn_up + self.num_fpn_down + self.num_fpn_downup", self.num_fpn_up + self.num_fpn_down + self.num_fpn_downup)
        if self.num_fpn_up + self.num_fpn_down + self.num_fpn_downup > 0:
            self.FPN = True
            self.fpn_layers = fpn_layers
            self.num_anchors_per_location_fpn = {}
            self.anchors_fpn = {}

            for layer in self.fpn_layers:
                anchors, self.num_anchors_per_location_fpn[layer] = self.generate_anchors(
                    anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range, nusc=self.nusc,
                    fpn_layer=layer
                )
                self.anchors_fpn[layer] = [x.cuda() for x in anchors]

                # print("self.num_anchors_per_location_fpn[layer] ", self.num_anchors_per_location_fpn[layer] )
        else:
            self.FPN = False
            self.fpn_layers = []

        self.target_assigner = self.get_target_assigner(anchor_target_cfg, anchor_generator_cfg, fpn_layers=self.fpn_layers)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

        if self.nusc:
            self.dir_idx = 8
        else:
            self.dir_idx = 6


    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, nusc=False, fpn_layer=None):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg,
            nusc = nusc,
            fpn_layer=fpn_layer
        )
        # print("grid_size", grid_size) # 1008, 1008, 40?
        # print("fpn_layer", fpn_layer)
        if fpn_layer is None:
            feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
            # print("feature_map_size", feature_map_size) # 126
        else:
            feature_map_size = [grid_size[:2] // config['feature_map_stride_fpn'][fpn_layer] for config in anchor_generator_cfg]# 3: 4, 5:16
            # print("feature_map_size fpn", feature_map_size) # 252? 63?
        # print("nusc", nusc)
        if nusc:
            # anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)
            # print('-------------------')
            anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors_range(feature_map_size)
        else:
            anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)
        # print("anchors_list", anchors_list)
        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg, anchor_generator_cfg, fpn_layers=[]):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                anchor_target_cfg=anchor_target_cfg,
                anchor_generator_cfg=anchor_generator_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT,
                nusc=self.nusc,
                fpn_layers=fpn_layers
            )
        else:
            raise NotImplementedError
        return target_assigner

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    @staticmethod
    def make_conv_layers(conv_cfg, input_channels, output_channels):
        conv_layers = []
        c_in = input_channels
        for k in range(0, conv_cfg.__len__()):
            conv_layers.extend([
                nn.Conv2d(c_in, conv_cfg[k], 1, 1, bias=False),
                nn.ReLU(True),
            ])
            c_in = conv_cfg[k]
        conv_layers.append(nn.Conv2d(c_in, output_channels, 1, 1, bias=True))
        # nn.ReLU(inplace=False)
        return nn.Sequential(*conv_layers)

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )
        self.add_module(
            'mmd_loss_func',
            loss_utils.RBFMMDLoss(sigma_list=[0.01, 0.1, 1, 10, 100])
        )

    def assign_targets(self, gt_boxes=None, dom_src=None, pseudo=False, pseudo_weights=[], fpn_layer=None, patch_tensor=None, dom_squeeze=True):#, localdom=False):
        """
        Args:
            gt_boxes: (B, M, 8) #10 dom
        Returns:

        """
        # if dom_src is None:
        #     targets_dict = self.target_assigner.assign_targets(
        #         self.anchors, gt_boxes, self.use_multihead, pseudo=pseudo
        #     )
        # else:
        # print("fpn_layer", fpn_layer)
        if fpn_layer is not None:
            anchors = self.anchors_fpn[fpn_layer]
        else:
            anchors = self.anchors

        # print("self.dom_squeeze",self.dom_squeeze)
        targets_dict = self.target_assigner.assign_targets(
            anchors, gt_boxes, self.use_multihead, dom_src=dom_src, pseudo=pseudo, pseudo_weights=pseudo_weights, fpn_layer=fpn_layer, fpn_only=self.fpn_only, patch_tensor=patch_tensor, dom_squeeze=self.dom_squeeze)#, localdom=localdom)
        return targets_dict

    def get_cls_layer_loss(self, tb_dict={}, mcd_id=None):
        if mcd_id is not None:
            suffix = f'_{mcd_id}'
        else:
            suffix = ''

        # print("self.forward_ret_dict[f'cls_preds{suffix}']", self.forward_ret_dict[f'cls_preds{suffix}'])
        cls_preds = self.forward_ret_dict[f'cls_preds{suffix}']
        # print('cls_preds', cls_preds.shape) # 188, 188, 2
        box_cls_labels = self.forward_ret_dict[f'box_cls_labels']

        # print("box_cls_labels", box_cls_labels)
         # 0 0 -1 1 0 0 1 0 0 0 0
        # print('box_cls_labels', box_cls_labels.shape) # 69938
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors] # filter -1 boxes
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        # print("one_hot_targets", one_hot_targets.shape)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        # print("one_hot_targets early", one_hot_targets.shape)
        # print("batch_size", batch_size)
        # print("self.num_class", self.num_class)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        # print("cls_preds", cls_preds.shape)
        # print("one_hot_targets", one_hot_targets.shape)
        # print("cls_weights", cls_weights.shape)
        # print("one_hot_targets late", one_hot_targets.shape)
        # import pdb
        # pdb.set_trace()
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

        if tb_dict is None:
            tb_dict = {}
        else:
            tb_dict.update({
                'rpn_loss_cls': cls_loss.item()
            })

        return cls_loss, tb_dict

    def get_fpn_cls_layer_loss(self, tb_dict={}, mcd_id=None):
        if mcd_id is not None:
            suffix = f'_{mcd_id}'
        else:
            suffix = ''

        cls_loss_fpn = []
        for layers in self.fpn_layers:
            cls_preds = self.forward_ret_dict[f'cls_preds{suffix}_fpn{layers}']
            box_cls_labels = self.forward_ret_dict[f'box_cls_labels_fpn{layers}'] # 0 0 -1 1 0 0 1 0 0 0 0
            batch_size = int(cls_preds.shape[0])
            cared = box_cls_labels >= 0  # [N, num_anchors] # filter -1 boxes
            positives = box_cls_labels > 0
            negatives = box_cls_labels == 0
            negative_cls_weights = negatives * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            reg_weights = positives.float()
            if self.num_class == 1:
                # class agnostic
                box_cls_labels[positives] = 1

            pos_normalizer = positives.sum(1, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
            cls_targets = cls_targets.unsqueeze(dim=-1)

            cls_targets = cls_targets.squeeze(dim=-1)
            one_hot_targets = torch.zeros(
                *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
            )
            # print("one_hot_targets", one_hot_targets.shape)
            one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
            # print("one_hot_targets early", one_hot_targets.shape)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)
            one_hot_targets = one_hot_targets[..., 1:]

            cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
            cls_loss = cls_loss_src.sum() / batch_size

            cls_loss_fpn.append(cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight'])

        cls_loss_fpn_sum = sum(cls_loss_fpn)

        if tb_dict is None:
            tb_dict = {}
        else:
            tb_dict.update({
                'rpn_loss_cls_fpn': cls_loss_fpn_sum.item()
            })
        return cls_loss_fpn_sum, tb_dict


    def consistency_loss(self, local_logits, global_logits):
        # print("local_logits1", local_logits)
        # print("global_logits1", global_logits)
        # zero = torch.zeros(global_logits.size())[0].float().cuda()

        # # local_logits = torch.sum(local_logits)/list(local_logits.size())[0]
        # # local_logits = torch.ones(global_logits.size()).cuda() * local_logits
        # #local_logits = local_logits.view(1, list(local_logits.size())[0])
        # #global_logits = global_logits.view(1, list(global_logits.size())[0])
        # loss = nn.L1Loss()
        # return loss(local_logits- global_logits, zero)

        #consistency_prob = torch.max(F.softmax(base_score, dim=1),dim=1)[0]
        # consistency_prob = F.softmax(global_logits, dim=1)[:,1,:,:]
        # consistency_prob=torch.mean(global_logits)
        global_logits=global_logits.repeat(local_logits.size())
        loss = nn.MSELoss(size_average=False)

        # print("local_logits", local_logits)
        # print("global_logits", global_logits)

        return loss(local_logits, global_logits)#, zero)

    def get_dom_layer_loss(self, dom_ins=False, tb_dict={}, diffdom=False):
        dom_preds = self.forward_ret_dict['dom_img_preds']
        dom_labels = self.forward_ret_dict['dom_img_labels']

        ########## LOCAL LOSS ##########
        if self.dom_squeeze and not (self.range_guidance_conv_dom or self.range_guidance_new_conv_dom):
            dom_preds = dom_preds.view(-1)

        if self.range_guidance_conv_dom or self.range_guidance_new_conv_dom:
            dom_labels = dom_labels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        dom_preds_sig = torch.sigmoid(dom_preds)
        dom_labels = dom_labels.expand_as(dom_preds_sig)

        dom_loss = F.binary_cross_entropy(dom_preds_sig.float(), dom_labels.float())

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        dom_img_loss = dom_loss * loss_weights_dict['dom_weight']

        ########## GLOBAL LOSS ##########
        if self.range_guidance_double_dom or self.sep_two_dom or self.joint_two_dom or self.joint_pma:
            dom_preds2 = self.forward_ret_dict['dom_img_preds2']
            dom_labels2 = self.forward_ret_dict['dom_img_labels']

            if self.dom_squeeze:
                dom_preds2 = dom_preds2.view(-1)

            dom_preds_sig2 = torch.sigmoid(dom_preds2)
            dom_labels2 = dom_labels2.expand_as(dom_preds_sig2)

            dom_loss2 = F.binary_cross_entropy(dom_preds_sig2.float(), dom_labels2.float())

            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            if self.two_dom_weight:
                dom_img_loss2 = dom_loss2 * loss_weights_dict['dom_weight2']
            else:
                dom_img_loss2 = dom_loss2 * loss_weights_dict['dom_weight']

            if self.two_dom_reg:
                dom_reg_loss = self.consistency_loss(dom_preds_sig, dom_preds_sig2) *  loss_weights_dict['consist_weight']

                if self.diff_dom_opt:
                    if diffdom:
                        dom_loss = dom_img_loss2 + dom_reg_loss
                    else:
                        dom_loss = dom_img_loss
                else:
                    dom_loss = dom_img_loss + dom_img_loss2 + dom_reg_loss
            else:
                if self.diff_dom_opt:
                    if diffdom:
                        dom_loss = dom_img_loss2
                    else:
                        dom_loss = dom_img_loss
                else:
                    dom_loss = dom_img_loss + dom_img_loss2
        else:
            dom_loss = dom_img_loss
        # if dom_ins:
        #     ...
        if tb_dict is None:
            tb_dict = {}
        else:
            # if self.diff_dom_opt:
            #     tb_dict.update({'img_dom_loss1': dom_loss[0].item(), 'img_dom_loss2': dom_loss[1].item()})
            # else:
            tb_dict.update({'img_dom_loss': dom_loss.item()})
        return dom_loss, tb_dict

    def get_dom_range_layer_loss(self, dom_ins=False, tb_dict={}):
        # domain_criterion = torch.nn.BCEWithLogitsLoss().cuda()
        # dom_loss2 = domain_criterion(dom_preds, dom_labels) dom_src

        dom_loss_range = []
        for n in range(0+self.remove_near_range, self.range_da-self.remove_far_range):

            dom_preds = self.forward_ret_dict[f'dom_img_preds_range{n}']
            dom_labels = self.forward_ret_dict['dom_img_labels']#.expand_as(dom_preds).float() #box_

            # print("dom_preds", dom_preds)
            # print("dom_labels", dom_labels)

            if self.dom_squeeze:
                dom_preds = dom_preds.view(-1)
            # print("dom_preds", dom_preds)
            # print("dom_labels", dom_labels)
            dom_preds_sig = torch.sigmoid(dom_preds)
            dom_labels = dom_labels.expand_as(dom_preds_sig)

            # dom_loss = F.binary_cross_entropy(dom_preds_sig, dom_labels.float())
            try:
                # print("correct dom_labels", dom_labels)
                # print("correct dom_preds_sig", dom_preds_sig)
                dom_loss = F.binary_cross_entropy(dom_preds_sig.float(), dom_labels)
            except:
                # print("dom_img_labels error!!", dom_labels)
                # print("dom_preds_sig error!!", dom_preds_sig)
                dom_loss = F.binary_cross_entropy(dom_preds_sig.float(), dom_labels)

            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            dom_img_loss = dom_loss * loss_weights_dict['dom_weight']

            # print("dom_img_loss", dom_img_loss)

            dom_loss_range.append(dom_img_loss)

            if self.keep_xy:
                dom_preds = self.forward_ret_dict[f'dom_img_preds_range{n}_2']
                dom_labels = self.forward_ret_dict['dom_img_labels']#.expand_as(dom_preds).float() #box_

                if self.dom_squeeze:
                    dom_preds = dom_preds.view(-1)
                dom_preds_sig = torch.sigmoid(dom_preds)
                dom_labels = dom_labels.expand_as(dom_preds_sig)

                # dom_loss = F.binary_cross_entropy(dom_preds_sig, dom_labels.float())
                try:
                    # print("correct dom_labels", dom_labels)
                    # print("correct dom_preds_sig", dom_preds_sig)
                    dom_loss = F.binary_cross_entropy(dom_preds_sig.float(), dom_labels)
                except:
                    # print("dom_img_labels error !!", dom_labels)
                    # print("dom_preds_sig error !!", dom_preds_sig)
                    dom_loss = F.binary_cross_entropy(dom_preds_sig.float(), dom_labels)

                loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
                dom_img_loss = dom_loss * loss_weights_dict['dom_weight']
                # print("dom_img_loss", dom_img_loss)

                dom_loss_range.append(dom_img_loss)
            # if dom_ins:
            #     ...
        dom_loss_all = sum(dom_loss_range)

        # if dom_ins:
        #     ...
        if tb_dict is None:
            tb_dict = {}
        else:
            tb_dict.update({'img_dom_loss': dom_loss_all.item()})
        return dom_loss, tb_dict

    def get_dom_interval_layer_loss(self, dom_ins=False, tb_dict={}):
        # domain_criterion = torch.nn.BCEWithLogitsLoss().cuda()
        # dom_loss2 = domain_criterion(dom_preds, dom_labels) dom_src

        dom_loss_interval = []
        for n in range(self.interval_da):

            dom_labels = self.forward_ret_dict['dom_img_labels'] #box_
            dom_preds = self.forward_ret_dict[f'dom_img_preds_interval{n}']

            if self.dom_squeeze:
                dom_preds = dom_preds.view(-1)
            dom_preds_sig = torch.sigmoid(dom_preds)
            dom_labels = dom_labels.expand_as(dom_preds_sig)

            # dom_loss = F.binary_cross_entropy(dom_preds_sig, dom_labels.float())
            try:
                dom_loss = F.binary_cross_entropy(dom_preds_sig.float(), dom_labels.float())
            except:
                # print("dom_img_labels error !!", dom_labels)
                # print("dom_preds_sig error !!", dom_preds_sig)
                dom_loss = F.binary_cross_entropy(dom_preds_sig.float(), dom_labels.expand_as(dom_preds_sig).float())

            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            dom_img_loss = dom_loss * loss_weights_dict['dom_weight']

            dom_loss_interval.append(dom_img_loss)
            # if dom_ins:
            #     ...
        dom_loss_all = sum(dom_loss_interval)

        # if dom_ins:
        #     ...
        if tb_dict is None:
            tb_dict = {}
        else:
            tb_dict.update({'img_dom_loss': dom_loss_all.item()})
        return dom_loss, tb_dict


    def get_fpn_dom_layer_loss(self, dom_ins=False, tb_dict={}, diffdom=False):
        # domain_criterion = torch.nn.BCEWithLogitsLoss().cuda()
        # dom_loss2 = domain_criterion(dom_preds, dom_labels)

        dom_loss_fpn = []
        for layers in self.fpn_layers:
            ##### local ########
            dom_labels = self.forward_ret_dict[f'dom_img_labels_fpn{layers}'] #box_
            dom_preds = self.forward_ret_dict[f'dom_img_preds_fpn{layers}']

            if self.dom_squeeze and not (self.range_guidance_conv_dom or self.range_guidance_new_conv_dom):
                dom_preds = dom_preds.view(-1)

            # print("dom_preds", dom_preds.shape)
            # print("dom_labels", dom_labels.shape)
            if self.range_guidance_conv_dom or self.range_guidance_new_conv_dom:
                dom_labels = dom_labels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            dom_preds_sig = torch.sigmoid(dom_preds)
            dom_labels = dom_labels.expand_as(dom_preds_sig)

            # dom_loss = F.binary_cross_entropy(dom_preds_sig, dom_labels.float())
            try:
                dom_loss = F.binary_cross_entropy(dom_preds_sig, dom_labels.float())
            except:
                # print("dom_img_labels error !!", dom_labels)
                # print("dom_preds_sig error !!", dom_preds_sig)
                dom_loss = F.binary_cross_entropy(dom_preds_sig, dom_labels.expand_as(dom_preds_sig).float())

            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            dom_img_loss = dom_loss * loss_weights_dict['dom_weight']

            # print("dom_img_loss", dom_img_loss)

            ########## GLOBAL LOSS ##########
            if self.range_guidance_double_dom or self.sep_two_dom or self.joint_two_dom or self.joint_pma:
                dom_preds2 = self.forward_ret_dict[f'dom_img_preds2_fpn{layers}']
                dom_labels2 = self.forward_ret_dict[f'dom_img_labels_fpn{layers}']

                if self.dom_squeeze:
                    dom_preds2 = dom_preds2.view(-1)

                dom_preds_sig2 = torch.sigmoid(dom_preds2)
                dom_labels2 = dom_labels2.expand_as(dom_preds_sig2)

                dom_loss2 = F.binary_cross_entropy(dom_preds_sig2.float(), dom_labels2.float())

                loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
                if self.two_dom_weight:
                    dom_img_loss2 = dom_loss2 * loss_weights_dict['dom_weight2']
                else:
                    dom_img_loss2 = dom_loss2 * loss_weights_dict['dom_weight']

                if self.two_dom_reg:
                    dom_reg_loss = self.consistency_loss(dom_preds_sig, dom_preds_sig2) * loss_weights_dict['consist_weight']

                    if self.diff_dom_opt:
                        if diffdom:
                            dom_loss_layer = dom_img_loss2 + dom_reg_loss
                            # print("dom_loss_layer = dom_img_loss2 + dom_reg_loss", dom_loss_layer)
                        else:
                            dom_loss_layer = dom_img_loss
                            # print("dom_loss_layer = dom_img_loss", dom_loss_layer)
                    else:
                        dom_loss_layer = dom_img_loss + dom_img_loss2 + dom_reg_loss
                        # print("dom_loss_layer = dom_img_loss + dom_img_loss2 + dom_reg_loss", dom_loss_layer)
                else:
                    if self.diff_dom_opt:
                        if diffdom:
                            dom_loss_layer = dom_img_loss2
                            # print("dom_loss_layer = dom_img_loss2", dom_loss_layer)
                        else:
                            dom_loss_layer = dom_img_loss
                            # print("dom_loss_layer = dom_img_loss", dom_loss_layer)
                    else:
                        dom_loss_layer = dom_img_loss + dom_img_loss2
                        # print("dom_loss_layer = dom_img_loss + dom_img_loss2", dom_loss_layer)
            else:
                dom_loss_layer = dom_img_loss

            dom_loss_fpn.append(dom_loss_layer)
            # if dom_ins:
            #     ...

        if self.cross_scale:
            scale_labels = self.forward_ret_dict[f'scale_labels1'] #box_
            scale_preds = self.forward_ret_dict[f'scale_preds1']

            scale_labels2 = self.forward_ret_dict[f'scale_labels2'] #box_
            scale_preds2 = self.forward_ret_dict[f'scale_preds2']

            scale_preds_sig = torch.sigmoid(scale_preds)
            scale_labels = scale_labels.expand_as(scale_preds_sig)

            scale_preds_sig2 = torch.sigmoid(scale_preds2)
            scale_labels2 = scale_labels2.expand_as(scale_preds_sig2)


            try:
                scale_loss = F.binary_cross_entropy(scale_preds_sig, scale_labels.float())
                scale_loss2 = F.binary_cross_entropy(scale_preds_sig2, scale_labels2.float())
            except:
                # print("scale_img_labels error !!", scale_labels)
                # print("scale_preds_sig error !!", scale_preds_sig)
                scale_loss = F.binary_cross_entropy(scale_preds_sig, scale_labels.expand_as(scale_preds_sig).float())
                scale_loss2 = F.binary_cross_entropy(scale_preds_sig2, scale_labels2.expand_as(scale_preds_sig2).float())

            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            scale_img_loss = scale_loss * loss_weights_dict['dom_weight']
            scale_img_loss2 = scale_loss2 * loss_weights_dict['dom_weight']

            dom_loss_fpn.append(scale_img_loss)
            dom_loss_fpn.append(scale_img_loss2)

            if self.cross_two_scale:
                scale_labels2_2 = self.forward_ret_dict[f'scale_labels2_2'] #box_
                scale_preds2_2 = self.forward_ret_dict[f'scale_preds2_2']

                scale_labels2_3 = self.forward_ret_dict[f'scale_labels2_3'] #box_
                scale_preds2_3 = self.forward_ret_dict[f'scale_preds2_3']

                scale_preds_sig2_2 = torch.sigmoid(scale_preds2_2)
                scale_labels2_2 = scale_labels.expand_as(scale_preds_sig2_2)

                scale_preds_sig2_3 = torch.sigmoid(scale_preds2_3)
                scale_labels2_3 = scale_labels2.expand_as(scale_preds_sig2_3)


                try:
                    scale_loss2_2 = F.binary_cross_entropy(scale_preds_sig2_2, scale_labels2_2.float())
                    scale_loss2_3 = F.binary_cross_entropy(scale_preds_sig2_3, scale_labels2_3.float())
                except:
                    # print("scale_img_labels error !!", scale_labels)
                    # print("scale_preds_sig error !!", scale_preds_sig)
                    scale_loss2_2 = F.binary_cross_entropy(scale_preds_sig2_2, scale_labels2_2.expand_as(scale_preds_sig2_2).float())
                    scale_loss2_3 = F.binary_cross_entropy(scale_preds_sig2_3, scale_labels2_3.expand_as(scale_preds_sig2_3).float())

                loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
                scale_img_loss2_2 = scale_loss2_2 * loss_weights_dict['dom_weight']
                scale_img_loss2_3 = scale_loss2_3 * loss_weights_dict['dom_weight']

                dom_loss_fpn.append(scale_img_loss2_2)
                dom_loss_fpn.append(scale_img_loss2_3)

        # dom_loss = F.binary_cross_entropy(dom_preds_sig, dom_labels.float())
        # print("dom_loss_fpn", dom_loss_fpn)
        dom_loss_all = sum(dom_loss_fpn)
        # print("dom_loss_all", dom_loss_all)

        if tb_dict is None:
            tb_dict = {}
        else:
            tb_dict.update({'img_fpn_dom_loss': dom_loss_all.item()})

        return dom_loss_all, tb_dict

    def get_mcd_src_loss(self, tb_dict={}):
        # print('mcd_src')
        cls_loss, tb_dict = self.get_cls_layer_loss(tb_dict, mcd_id=1)
        box_loss, tb_dict = self.get_box_reg_layer_loss(tb_dict, mcd_id=1)

        cls_loss2, tb_dict = self.get_cls_layer_loss(tb_dict, mcd_id=2)
        box_loss2, tb_dict = self.get_box_reg_layer_loss(tb_dict, mcd_id=2)
        # tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss + cls_loss2 + box_loss2

        if tb_dict is None:
            tb_dict = {'rpn_loss':rpn_loss.item()}
        else:
            tb_dict.update({'rpn_loss':rpn_loss.item()})

        return rpn_loss, tb_dict

    def get_mcd_src_fpn_loss(self, tb_dict={}):
        cls_loss, tb_dict = self.get_fpn_cls_layer_loss(tb_dict, mcd_id=1)
        box_loss, tb_dict = self.get_fpn_box_reg_layer_loss(tb_dict, mcd_id=1)

        cls_loss2, tb_dict = self.get_fpn_cls_layer_loss(tb_dict, mcd_id=2)
        box_loss2, tb_dict = self.get_fpn_box_reg_layer_loss(tb_dict, mcd_id=2)

        # tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss + cls_loss2 + box_loss2

        if tb_dict is None:
            tb_dict = {'rpn_loss_fpn':rpn_loss.item()}
        else:
            tb_dict.update({'rpn_loss_fpn':rpn_loss.item()})

        return rpn_loss, tb_dict

    def get_mcd_tgt_loss(self, tb_dict={}):

        # print('mcd_tgt')
        mcd_loss_cls, tb_dict_1 = self.get_mcd_cls_layer_loss()
        mcd_loss_box, tb_dict_2 = self.get_mcd_box_layer_loss()

        loss = mcd_loss_cls + mcd_loss_box

        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)

        return loss, tb_dict

    def get_mcd_tgt_fpn_loss(self, tb_dict={}):

        # print('mcd_tgt')
        mcd_loss_cls, tb_dict_1 = self.get_mcd_fpn_cls_layer_loss()
        mcd_loss_box, tb_dict_2 = self.get_mcd_fpn_box_layer_loss()

        loss = mcd_loss_cls + mcd_loss_box

        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)

        return loss, tb_dict

    def mcd_discrepancy(self, out1, out2):
        """discrepancy loss"""
        out = torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))
        return out


    def get_mcd_cls_layer_loss(self, tb_dict={}):

        pred_t1 = self.forward_ret_dict[f'mcd_cls_preds_1']
        pred_t2 = self.forward_ret_dict[f'mcd_cls_preds_2']
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS

        mcd_loss_adv = -1 * self.mcd_discrepancy(pred_t1, pred_t2) * loss_weights_dict['disc_cls_weight']

        tb_dict.update({'mcd_cls_loss': mcd_loss_adv.item()})

        return mcd_loss_adv, tb_dict

    def get_mcd_box_layer_loss(self, tb_dict={}):

        pred_t1 = self.forward_ret_dict[f'mcd_box_preds_1']
        pred_t2 = self.forward_ret_dict[f'mcd_box_preds_2']
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS

        mcd_loss_adv = -1 * self.mcd_discrepancy(pred_t1, pred_t2) * loss_weights_dict['disc_loc_weight']

        tb_dict.update({'mcd_box_loss': mcd_loss_adv.item()})

        return mcd_loss_adv, tb_dict

    def get_mcd_fpn_cls_layer_loss(self, tb_dict={}):

        mcd_loss_adv_fpn = []
        for layers in self.fpn_layers:
            pred_t1 = self.forward_ret_dict[f'mcd_cls_preds_1_fpn{layers}']
            pred_t2 = self.forward_ret_dict[f'mcd_cls_preds_2_fpn{layers}']
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS

            mcd_loss_adv = -1 * self.mcd_discrepancy(pred_t1, pred_t2) * loss_weights_dict['disc_loc_weight']

            mcd_loss_adv_fpn.append(mcd_loss_adv)

        mcd_loss_adv_fpn_sum = sum(mcd_loss_adv_fpn)

        tb_dict.update({'mcd_loss_adv_fpn_loss': mcd_loss_adv_fpn_sum.item()})

        return mcd_loss_adv_fpn_sum, tb_dict

    def get_mcd_fpn_box_layer_loss(self, tb_dict={}):

        mcd_loss_adv_fpn = []
        for layers in self.fpn_layers:
            pred_t1 = self.forward_ret_dict[f'mcd_cls_preds_1_fpn{layers}']
            pred_t2 = self.forward_ret_dict[f'mcd_cls_preds_2_fpn{layers}']
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS

            mcd_loss_adv = -1 * self.mcd_discrepancy(pred_t1, pred_t2) * loss_weights_dict['disc_loc_weight']

            mcd_loss_adv_fpn.append(mcd_loss_adv)

        mcd_loss_adv_fpn_sum = sum(mcd_loss_adv_fpn)

        tb_dict.update({'mcd_loss_adv_fpn_loss': mcd_loss_adv_fpn_sum.item()})

        return mcd_loss_adv_fpn_sum, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def add_sin_difference_nusc(boxes1, boxes2, dim=8):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target_nusc(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 8] + anchors[..., 8]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_range_inv_loss(self, tb_dict={}):

        range_pred_near = self.forward_ret_dict['range_pred_near'] #box_
        range_pred_far = self.forward_ret_dict['range_pred_far']

        # print("range_pred_near", range_pred_near.shape)
        # print("range_pred_far", range_pred_far.shape)

        range_labels_near = self.forward_ret_dict['range_labels_near'].view(-1) #box_
        range_labels_far = self.forward_ret_dict['range_labels_far'].view(-1)
        # print("range_labels_near", range_labels_near)
        # print("range_labels_far", range_labels_far.shape)
        # print("range_labels_near", range_labels_near.shape)

        range_pred_near_flat = range_pred_near.view(-1)
        range_pred_near_flat_sig = torch.sigmoid(range_pred_near_flat)
        # print("range_pred_near_flat_sig", range_pred_near_flat_sig.shape)

        range_pred_far_flat = range_pred_far.view(-1)
        range_pred_far_flat_sig = torch.sigmoid(range_pred_far_flat)

        range_loss_near = F.binary_cross_entropy(range_pred_near_flat_sig, range_labels_near.float())

        range_loss_far = F.binary_cross_entropy(range_pred_far_flat_sig, range_labels_far.float())

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        rangeinv_loss_all = range_loss_near * loss_weights_dict['range_inv_weight'] + range_loss_far * loss_weights_dict['range_inv_weight']

        tb_dict.update({
                    'rangeinv_loss_all': rangeinv_loss_all.item()
                })

        return rangeinv_loss_all, tb_dict


    def get_box_reg_layer_loss(self, tb_dict={}, mcd_id=None):
        if mcd_id is not None:
            suffix = f'_{mcd_id}'
        else:
            suffix = ''
        box_preds = self.forward_ret_dict[f'box_preds{suffix}']
        box_dir_cls_preds = self.forward_ret_dict.get(f'dir_cls_preds{suffix}', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        if self.nusc:
            box_preds_sin, reg_targets_sin = self.add_sin_difference_nusc(box_preds, box_reg_targets)
        else:
            box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        if tb_dict is None:
            tb_dict = {'rpn_loss_loc': loc_loss.item()}
        else:
            tb_dict.update({
                'rpn_loss_loc': loc_loss.item()
            })

        if box_dir_cls_preds is not None:
            if self.nusc:
                dir_targets = self.get_direction_target_nusc(
                    anchors, box_reg_targets,
                    dir_offset=self.model_cfg.DIR_OFFSET,
                    num_bins=self.model_cfg.NUM_DIR_BINS
                )
            else:
                dir_targets = self.get_direction_target(
                    anchors, box_reg_targets,
                    dir_offset=self.model_cfg.DIR_OFFSET,
                    num_bins=self.model_cfg.NUM_DIR_BINS
                )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()


        if self.FPN:
            box_loss_fpn = []
            for layer in self.fpn_layers:
                box_preds = self.forward_ret_dict[f'box_preds_fpn{layer}']
                box_dir_cls_preds = self.forward_ret_dict.get(f'dir_cls_preds_fpn{layer}', None)
                box_reg_targets = self.forward_ret_dict[f'box_reg_targets_fpn{layer}']
                box_cls_labels = self.forward_ret_dict[f'box_cls_labels_fpn{layer}']
                batch_size = int(box_preds.shape[0])

                positives = box_cls_labels > 0
                reg_weights = positives.float()
                pos_normalizer = positives.sum(1, keepdim=True).float()
                reg_weights /= torch.clamp(pos_normalizer, min=1.0)

                if isinstance(self.anchors_fpn[layer], list):
                    if self.use_multihead:
                        anchors = torch.cat(
                            [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                            self.anchors_fpn[layer]], dim=0)
                    else:
                        anchors = torch.cat(self.anchors_fpn[layer], dim=-3)
                else:
                    anchors = self.anchors_fpn[layer]
                anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
                box_preds = box_preds.view(batch_size, -1,
                                        box_preds.shape[-1] // self.num_anchors_per_location_fpn[layer] if not self.use_multihead else
                                        box_preds.shape[-1])
                # sin(a - b) = sinacosb-cosasinb
                if self.nusc:
                    box_preds_sin, reg_targets_sin = self.add_sin_difference_nusc(box_preds, box_reg_targets)
                else:
                    box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
                loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
                loc_loss = loc_loss_src.sum() / batch_size

                loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                box_loss_fpn.append(loc_loss)

            box_loss_fpn_all = sum(box_loss_fpn)

            if tb_dict is None:
                tb_dict = {'rpn_loss_loc_fpn': box_loss_fpn_all.item()}
            else:
                tb_dict.update({
                    'rpn_loss_loc_fpn': box_loss_fpn_all.item()
                })

            if box_dir_cls_preds is not None:
                box_loss_dir_fpn = []
                for layer in self.fpn_layers:
                    if self.nusc:
                        dir_targets = self.get_direction_target_nusc(
                        anchors, box_reg_targets,
                        dir_offset=self.model_cfg.DIR_OFFSET,
                        num_bins=self.model_cfg.NUM_DIR_BINS
                        )
                    else:
                        dir_targets = self.get_direction_target(
                            anchors, box_reg_targets,
                            dir_offset=self.model_cfg.DIR_OFFSET,
                            num_bins=self.model_cfg.NUM_DIR_BINS
                        )

                    dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
                    weights = positives.type_as(dir_logits)
                    weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                    dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
                    dir_loss = dir_loss.sum() / batch_size
                    dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
                    box_loss_dir_fpn.append(dir_loss)

                rpn_loss_dir_fpn = sum(box_loss_dir_fpn)

            if self.FPN:
                tb_dict.update({
                    'rpn_loss_dir_fpn': rpn_loss_dir_fpn.item()
                })
        if self.FPN:
            box_loss_all = box_loss + box_loss_fpn_all + rpn_loss_dir_fpn
        else:
            box_loss_all = box_loss

        return box_loss_all, tb_dict

    def get_fpn_box_reg_layer_loss(self, tb_dict={}, mcd_id=None):
        if mcd_id is not None:
            suffix = f'_{mcd_id}'
        else:
            suffix = ''

        if self.FPN:
            box_loss_fpn = []
            for layer in self.fpn_layers:
                box_preds = self.forward_ret_dict[f'box_preds{suffix}_fpn{layer}']
                box_dir_cls_preds = self.forward_ret_dict.get(f'dir_cls_preds{suffix}_fpn{layer}', None)
                box_reg_targets = self.forward_ret_dict[f'box_reg_targets_fpn{layer}']
                box_cls_labels = self.forward_ret_dict[f'box_cls_labels_fpn{layer}']
                batch_size = int(box_preds.shape[0])

                positives = box_cls_labels > 0
                reg_weights = positives.float()
                pos_normalizer = positives.sum(1, keepdim=True).float()
                reg_weights /= torch.clamp(pos_normalizer, min=1.0)

                if isinstance(self.anchors_fpn[layer], list):
                    if self.use_multihead:
                        anchors = torch.cat(
                            [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                            self.anchors_fpn[layer]], dim=0)
                    else:
                        anchors = torch.cat(self.anchors_fpn[layer], dim=-3)
                else:
                    anchors = self.anchors_fpn[layer]
                anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
                box_preds = box_preds.view(batch_size, -1,
                                        box_preds.shape[-1] // self.num_anchors_per_location_fpn[layer] if not self.use_multihead else
                                        box_preds.shape[-1])
                # sin(a - b) = sinacosb-cosasinb
                if self.nusc:
                    box_preds_sin, reg_targets_sin = self.add_sin_difference_nusc(box_preds, box_reg_targets)
                else:
                    box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
                loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
                loc_loss = loc_loss_src.sum() / batch_size

                loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                box_loss_fpn.append(loc_loss)

            box_loss_fpn_all = sum(box_loss_fpn)

            if tb_dict is None:
                tb_dict = {'rpn_loss_loc_fpn': box_loss_fpn_all.item()}
            else:
                tb_dict.update({
                    'rpn_loss_loc_fpn': box_loss_fpn_all.item()
                })

            if box_dir_cls_preds is not None:
                box_loss_dir_fpn = []
                for layer in self.fpn_layers:
                    if self.nusc:
                        dir_targets = self.get_direction_target_nusc(
                        anchors, box_reg_targets,
                        dir_offset=self.model_cfg.DIR_OFFSET,
                        num_bins=self.model_cfg.NUM_DIR_BINS
                        )
                    else:
                        dir_targets = self.get_direction_target(
                            anchors, box_reg_targets,
                            dir_offset=self.model_cfg.DIR_OFFSET,
                            num_bins=self.model_cfg.NUM_DIR_BINS
                        )

                    dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
                    weights = positives.type_as(dir_logits)
                    weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                    dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
                    dir_loss = dir_loss.sum() / batch_size
                    dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
                    box_loss_dir_fpn.append(dir_loss)

                rpn_loss_dir_fpn = sum(box_loss_dir_fpn)

            if self.FPN:
                tb_dict.update({
                    'rpn_loss_dir_fpn': rpn_loss_dir_fpn.item()
                })
        box_loss_all = box_loss_fpn_all + rpn_loss_dir_fpn

        return box_loss_all, tb_dict

    def get_loss(self, tb_dict={}):
        cls_loss, tb_dict = self.get_cls_layer_loss(tb_dict)
        box_loss, tb_dict = self.get_box_reg_layer_loss(tb_dict)
        # tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        if tb_dict is None:
            tb_dict = {'rpn_loss':rpn_loss.item()}
        else:
            tb_dict.update({'rpn_loss':rpn_loss.item()})
        return rpn_loss, tb_dict

    def get_fpn_loss(self, tb_dict={}):
        cls_loss, tb_dict = self.get_fpn_cls_layer_loss(tb_dict)
        box_loss, tb_dict = self.get_fpn_box_reg_layer_loss(tb_dict)
        # tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        if tb_dict is None:
            tb_dict = {'rpn_loss_fpn':rpn_loss.item()}
        else:
            tb_dict.update({'rpn_loss_fpn':rpn_loss.item()})
        return rpn_loss, tb_dict

    def get_dom_loss(self, dom_ins=False, tb_dict={}, diffdom=False):
        if self.range_da > 0:
            dom_loss, tb_dict = self.get_dom_range_layer_loss(dom_ins=dom_ins, tb_dict={})
        elif self.interval_da > 0:
            dom_loss, tb_dict = self.get_dom_interval_layer_loss(dom_ins=dom_ins, tb_dict={})
        else:
            dom_loss, tb_dict = self.get_dom_layer_loss(dom_ins=dom_ins, tb_dict={}, diffdom=diffdom)

        return dom_loss, tb_dict

    def get_fpn_dom_loss(self, dom_ins=False, tb_dict={}, diffdom=False):
        dom_loss, tb_dict = self.get_fpn_dom_layer_loss(dom_ins=dom_ins, tb_dict={}, diffdom=False)

        return dom_loss, tb_dict

    def mean_interpolate_point_torch(self, point_feat, x_idxs, y_idxs, bev_features_map_shape):
        """
        Args:
            im: (P, C) [y, x]
            x: (N)
            y: (N)

        Returns:

        """

        # print("bev_features_map", bev_features_map)
        batch_size, _, x, y = bev_features_map_shape
        # print("point_feat",point_feat.shape)
        point_feat_dim = point_feat.shape[1]

        # print("interpolated_bev_features", interpolated_bev_features.shape) # 1, 128, 126, 126
        interpolated_bev_features = interpolated_bev_features.view(batch_size, point_feat_dim, -1) # 1, 128, 126*126=15876

        # for k in range(batch_size):
        xy_idxs = (x_idxs +1) * (y_idxs +1) -1
        # print("cur_xy_idxs", cur_xy_idxs)

        interpolated_bev_features[:, :, xy_idxs] = point_feat[:, :]
        # print("point_feat[:, :]", point_feat[:, :])

        # print("interpolated_bev_features",interpolated_bev_features)

        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]

        wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
        wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
        wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
        wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
        ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
        return ans

    def interpolate_to_bev_features_fast(self, keypoints_points, keypoints_features, bev_features_map_shape, bev_stride):

        batch_size, _, bev_x, bev_y = bev_features_map_shape
        _, point_feat_dim = keypoints_features.shape

        batch_keypoints_features = keypoints_features.view(batch_size, -1, point_feat_dim)
        batch_keypoints_points = keypoints_points.view(batch_size, -1, keypoints_points.shape[-1])

        # print("batch_keypoints_features", batch_keypoints_features.shape) # 2, 1024, 128
        # print("batch_keypoints_points", batch_keypoints_points.shape) # 2, 1024, 3

        # print("x_idxs max min0", torch.max(batch_keypoints_points[:,:,0]), torch.min(batch_keypoints_points[:,:,0]))
        # print("y_idxs max min0", torch.max(batch_keypoints_points[:,:,1]), torch.min(batch_keypoints_points[:,:,1]))

        x_idxs = (batch_keypoints_points[:,:,0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (batch_keypoints_points[:,:,1] - self.point_cloud_range[1]) / self.voxel_size[1]
        # print("x_idxs max min", torch.max(x_idxs), torch.min(x_idxs))
        # print("y_idxs max min", torch.max(y_idxs), torch.min(y_idxs))
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        voxel_idx_x_max = bev_x - 1
        # voxel_idx_x_max = bev_x - 1
        voxel_idx_y_max = bev_y - 1

        idxs = batch_keypoints_points.shape[1]

        # print("x_idxs", x_idxs.shape) # 2, 1024(2048)
        # print("y_idxs", y_idxs.shape) # 2, 1024(2048)

        point_bev_features_list = []
        interpolated_add_bev_features = torch.zeros((batch_size, point_feat_dim, bev_x*bev_y)).cuda()

        interpolated_bev_features = torch.zeros((batch_size, point_feat_dim, bev_x, bev_y)).cuda()

        #  = torch.zeros((batch_size, 1, bev_x*bev_y)).cuda()
        for k in range(batch_size):
            cur_x_idxs = torch.round(x_idxs[k]).long()
            cur_y_idxs = torch.round(y_idxs[k]).long()
            cur_x_idxs[cur_x_idxs > voxel_idx_x_max] = voxel_idx_x_max
            cur_y_idxs[cur_y_idxs > voxel_idx_y_max] = voxel_idx_y_max

            # print("cur_x_idxs min", torch.min(cur_x_idxs)) # 2, 1024

                    # 2048 (126*126)
            xy_idxs = (cur_x_idxs +1) * (cur_y_idxs +1) -1
            # print("xy_idxs", xy_idxs.shape) # 1024
            # batch_keypoints_features 1024*128
            # interpolated_add_bev_features[k,:,:] 1, 128, 126*126(xy_idxs 1024)
            unique_xy, xy_counts = torch.unique(xy_idxs,return_counts=True)

            # print("unique_xy", unique_xy.shape) # 1260
            # print("xy_counts", xy_counts.shape)

            # print("interpolated_add_feat", interpolated_add_feat.shape) # 1024, 128
            interpolated_add_feat = batch_keypoints_features[k,:,:].transpose(-1,-2)
            # print("interpolated_add_feat", interpolated_add_feat.shape) # 128, 1024

            # print("interpolated_add_bev_features[k,:,:]", interpolated_add_bev_features[k,:,:].shape)
            # print("xy_idxs", xy_idxs.shape) # 1024
            # print("interpolated_add_feat", interpolated_add_feat.shape)

            interpolated_add_bev_features[k,:,:].index_add_(-1,xy_idxs,interpolated_add_feat) #
            # print("interpolated_add_bev_features add", interpolated_add_bev_features.shape, interpolated_add_bev_features)

            interpolated_add_bev_features[k,:,unique_xy] /= xy_counts.float()
            # print("interpolated_add_bev_features", interpolated_add_bev_features.shape)

            # interpolated_add_bev_features[k,:,xy_idxs] = batch_keypoints_features[k,:,:].transpose(-1,-2)

            # print("interpolated_add_bev_features", interpolated_add_bev_features.shape)
            # print("batch_keypoints_features", batch_keypoints_features.shape)

            interpolated_bev_features[k,:,:,:] = interpolated_add_bev_features[k,:,:].reshape(point_feat_dim, bev_x, bev_y)
            # print("interpolated_bev_features", interpolated_bev_features.shape)
            # cur_x_idxs = torch.round(x_idxs[k]).long()
            # cur_y_idxs = torch.round(y_idxs[k]).long()
            # # cur_x_idxs[cur_x_idxs > voxel_idx_x_max] = voxel_idx_x_max
            # # cur_y_idxs[cur_y_idxs > voxel_idx_y_max] = voxel_idx_y_max
            # cur_x_idxs[cur_x_idxs > voxel_idx_x_max] = voxel_idx_x_max
            # cur_y_idxs[cur_y_idxs > voxel_idx_x_max] = voxel_idx_x_max

            # _, interpolated_bev_counts = torch.unique(xy_idxs, return_counts=True)

            # print("xy_idxs", xy_idxs.shape)
            # print("xy_idxs", xy_idxs)

        # print("non zero", torch.nonzero(interpolated_bev_features).shape)
        # a.index_add_(-1,index,val)

        return interpolated_bev_features

    def interpolate_to_bev_features(self, keypoints_points, keypoints_features, bev_features_map_shape, bev_stride):
        # print("keypoints_points", keypoints_points.shape) # 2048, 3
        # print("keypoints_features", keypoints_features.shape) # 3072, 640 # 1024*2, 640
        # print("batch_size", batch_size) # 1
        # print("bev_features_map_shape", bev_features_map_shape) # 2, 512, 126, 126
        # print("bev_stride", bev_stride) # 8
        batch_size, _, bev_x, bev_y = bev_features_map_shape
        _, point_feat_dim = keypoints_features.shape

        batch_keypoints_features = keypoints_features.view(batch_size, -1, point_feat_dim)
        batch_keypoints_points = keypoints_points.view(batch_size, -1, keypoints_points.shape[-1])

        # print("batch_keypoints_features", batch_keypoints_features.shape) # 2, 1024, 128
        # print("batch_keypoints_points", batch_keypoints_points.shape) # 2, 1024, 3

        # print("x_idxs max min0", torch.max(batch_keypoints_points[:,:,0]), torch.min(batch_keypoints_points[:,:,0]))
        # print("y_idxs max min0", torch.max(batch_keypoints_points[:,:,1]), torch.min(batch_keypoints_points[:,:,1]))

        x_idxs = (batch_keypoints_points[:,:,0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (batch_keypoints_points[:,:,1] - self.point_cloud_range[1]) / self.voxel_size[1]
        # print("x_idxs max min", torch.max(x_idxs), torch.min(x_idxs))
        # print("y_idxs max min", torch.max(y_idxs), torch.min(y_idxs))
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        voxel_idx_x_max = bev_x - 1
        # voxel_idx_x_max = bev_x - 1
        # voxel_idx_y_max = bev_y - 1

        idxs = batch_keypoints_points.shape[1]

        # print("x_idxs", x_idxs.shape) # 2, 1024
        # print("y_idxs", y_idxs.shape) # 2, 1024

        point_bev_features_list = []
        interpolated_bev_features = torch.zeros((batch_size, point_feat_dim, bev_x, bev_y)).cuda()
        interpolated_bev_counts = torch.zeros((batch_size, 1, bev_x, bev_y)).cuda()
        for k in range(batch_size):
            cur_x_idxs = torch.round(x_idxs[k]).long()
            cur_y_idxs = torch.round(y_idxs[k]).long()
            # cur_x_idxs[cur_x_idxs > voxel_idx_x_max] = voxel_idx_x_max
            # cur_y_idxs[cur_y_idxs > voxel_idx_y_max] = voxel_idx_y_max
            cur_x_idxs[cur_x_idxs > voxel_idx_x_max] = voxel_idx_x_max
            cur_y_idxs[cur_y_idxs > voxel_idx_x_max] = voxel_idx_x_max

            # print("cur_x_idxs min", torch.min(cur_x_idxs)) # 2, 1024
            # print("cur_y_idxs min", torch.min(cur_y_idxs)) # 2, 1024
            # print("cur_x_idxs max", torch.max(cur_x_idxs)) # 2, 1024
            # print("cur_y_idxs max", torch.max(cur_y_idxs)) # 2, 1024

            xy_idxs = (cur_x_idxs +1) * (cur_y_idxs +1) -1

            for idx in range(idxs):
                # print(idx)
                # print("cur_x_idxs[idx]", cur_x_idxs[idx])
                # print("cur_y_idxs[idx]", cur_y_idxs[idx])
                cur_batch_point_features = batch_keypoints_features[k,idx,:] #1, #1024, 128 # 128
                # print("cur_batch_point_features", cur_batch_point_features.shape)
                # print("cur_batch_point_features", cur_batch_point_features.shape)
                interpolated_bev_features[k,:,cur_x_idxs[idx],cur_y_idxs[idx]] += cur_batch_point_features #
                # print("interpolated_bev_features[k,:,cur_x_idxs[idx],cur_y_idxs[idx]]", interpolated_bev_features[k,:,cur_x_idxs[idx],cur_y_idxs[idx]].shape)
                interpolated_bev_counts[k,:,cur_x_idxs[idx],cur_y_idxs[idx]] += 1
                # interpolated_bev_features[k,:,cur_x_idxs,cur_y_idxs] = self.mean_interpolate_point_torch(cur_batch_point_features, cur_x_idxs, cur_y_idxs, bev_features_map_shape)
                # point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))
                # print("point_bev_features_list", point_bev_features.unsqueeze(dim=0).shape)
        # print("interpolated_bev_counts", torch.nonzero(interpolated_bev_counts))
        interpolated_bev_counts[interpolated_bev_counts==0] = 1.
        interpolated_bev_features /= interpolated_bev_counts
        # print("interpolated_bev_features", interpolated_bev_features.shape)

        # point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        # print("interpolated_bev_features", interpolated_bev_features.shape)
        # print("interpolated_bev_features", torch.nonzero(interpolated_bev_features))
        # print("interpolated_bev_features", interpolated_bev_features)
        # print("non zero slow", torch.nonzero(interpolated_bev_features).shape)
        return interpolated_bev_features

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None, fpn_layer=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C) dim-3 is  Nusc:9+c

        """
        if fpn_layer is not None:
            anchors = self.anchors_fpn[fpn_layer]
        else:
            anchors = self.anchors

        if isinstance(anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in anchors], dim=0)
            else:
                anchors = torch.cat(anchors, dim=-3)
        else:
            anchors = anchors

        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float()
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1)
        if self.nusc:
            batch_box_preds = self.box_coder.decode_torch_velo(batch_box_preds, batch_anchors)
        else:
            batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., self.dir_idx] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., self.dir_idx] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., self.dir_idx] = common_utils.limit_period(
                -(batch_box_preds[..., self.dir_idx] + np.pi / 2), offset=0.5, period=np.pi * 2
            )
        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
