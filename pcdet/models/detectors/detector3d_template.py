import os

import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils


class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset, nusc=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.nusc = nusc
        self.fpn_only = model_cfg.get('FPN_ONLY', False)
        self.fpn_fuse_res = model_cfg.get('FPN_FUSE_RES', True)

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'point_head', 'dense_head', 'roi_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        self.num_fpn_up=self.model_cfg.get('FPN_UP_LAYERS',0)
        self.num_fpn_down=self.model_cfg.get('FPN_DOWN_LAYERS',0)
        self.num_fpn_downup=self.model_cfg.get('FPN_DOWNUP_LAYERS',0)

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_fpn_up=self.num_fpn_up,
            num_fpn_down=self.num_fpn_down,
            num_fpn_downup=self.num_fpn_downup
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features

        self.num_fpn_up=self.model_cfg.get('FPN_UP_LAYERS',0)
        self.num_fpn_down=self.model_cfg.get('FPN_DOWN_LAYERS',0)
        self.num_fpn_downup=self.model_cfg.get('FPN_DOWNUP_LAYERS',0)

        self.FPN = self.num_fpn_up + self.num_fpn_downup + self.num_fpn_down > 0

        if self.FPN:
            # print("backbone_2d_module.num_bev_features_fpn", backbone_2d_module.num_bev_features_fpn)
            model_info_dict['num_bev_features_fpn'] = {}
            for l in range(self.num_fpn_up):
                layer = str(3 - l)
                model_info_dict['num_bev_features_fpn'][layer] = backbone_2d_module.num_bev_features_fpn[layer]

            for l in range(self.num_fpn_down):
                layer = str(4 + 1 + l)
                model_info_dict['num_bev_features_fpn'][layer] = backbone_2d_module.num_bev_features_fpn[layer]

            if self.num_fpn_downup > 0:
                for l in range(self.num_fpn_downup+1):
                    layer = str(4 + l)
                    model_info_dict['num_bev_features_fpn'][layer] = backbone_2d_module.num_bev_features_fpn[layer]


        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features'],
            nusc=self.nusc
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        # print("model_info_dict['grid_size']",model_info_dict['grid_size'])
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.get('ROI_HEAD', False) == False:
            predict_boxes_when_training = False
        else:
            predict_boxes_when_training = True

            # self.FPN = self.num_fpn_up + self.num_fpn_downup > 0

        if self.FPN:
            # print("model_info_dict['num_bev_features_fpn']", model_info_dict['num_bev_features_fpn'])
            self.input_channels_fpn=model_info_dict['num_bev_features_fpn']

            fpn_layers = []

            for l in range(self.num_fpn_up):
                layer = str(3 - l)
                fpn_layers.append(layer)

            for l in range(self.num_fpn_down):
                layer = str(4 + 1 - l)
                fpn_layers.append(layer)

            if self.num_fpn_downup > 0:
                for l in range(self.num_fpn_downup+1):
                    layer = str(4 + l)
                    fpn_layers.append(layer)

            dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
                model_cfg=self.model_cfg.DENSE_HEAD,
                input_channels=model_info_dict['num_bev_features'],
                num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
                class_names=self.class_names,
                grid_size=model_info_dict['grid_size'],
                point_cloud_range=model_info_dict['point_cloud_range'],
                predict_boxes_when_training=predict_boxes_when_training,
                nusc=self.nusc,
                input_channels_fpn=self.input_channels_fpn,
                num_fpn_up=self.num_fpn_up,
                num_fpn_down=self.num_fpn_down,
                num_fpn_downup=self.num_fpn_downup,
                fpn_layers=fpn_layers,
                voxel_size=model_info_dict['voxel_size'],
            )

        else:
            self.input_channels_fpn=None

            # print(" build_dense_head voxel_size", model_info_dict['voxel_size'])
            dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
                model_cfg=self.model_cfg.DENSE_HEAD,
                input_channels=model_info_dict['num_bev_features'],
                num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
                class_names=self.class_names,
                grid_size=model_info_dict['grid_size'],
                point_cloud_range=model_info_dict['point_cloud_range'],
                predict_boxes_when_training=predict_boxes_when_training,
                nusc=self.nusc,
                voxel_size=model_info_dict['voxel_size'],
            )

        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            nusc = self.nusc,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            nusc=self.nusc,
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1
        )
            # nusc=self.nusc,

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        # print("batch_dict in ", batch_dict)
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            # print("batch_dict['batch_box_preds']", batch_dict['batch_box_preds'])
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            # print("box_preds", box_preds)
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            # print("cls_preds", cls_preds.shape)

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)

            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)

                # print("label_preds", label_preds.shape)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH,
                    nusc=self.nusc
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST,
                nusc=self.nusc
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

            # print("pred_dicts", pred_dicts)

        # print("cls ori", torch.sort(cls_preds)[0]) #31752
        return pred_dicts, recall_dict

    def post_processing_FPN(self, batch_dict, fpn_layers=[]):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        pred_dicts = {}
        recall_dict = {}
        for fpn_layer in fpn_layers:
            pred_dicts[fpn_layer] = []
            recall_dict[fpn_layer] = {}
        ### dict expand ###

        for fpn_layer in fpn_layers:

            # print("batch_dict in ", batch_dict)
            post_process_cfg = self.model_cfg.POST_PROCESSING
            batch_size = batch_dict['batch_size']


            for index in range(batch_size):
                if batch_dict.get('batch_index', None) is not None:
                    assert batch_dict[f'batch_box_preds_fpn{fpn_layer}'].shape.__len__() == 2
                    batch_mask = (batch_dict['batch_index'] == index)
                else:
                    assert batch_dict[f'batch_box_preds_fpn{fpn_layer}'].shape.__len__() == 3
                    batch_mask = index

                box_preds = batch_dict[f'batch_box_preds_fpn{fpn_layer}'][batch_mask]
                src_box_preds = box_preds

                if not isinstance(batch_dict[f'batch_cls_preds_fpn{fpn_layer}'], list):
                    cls_preds = batch_dict[f'batch_cls_preds_fpn{fpn_layer}'][batch_mask]

                    src_cls_preds = cls_preds
                    assert cls_preds.shape[1] in [1, self.num_class]

                    if not batch_dict[f'cls_preds_normalized_fpn{fpn_layer}']:
                        cls_preds = torch.sigmoid(cls_preds)
                else:
                    cls_preds = [x[batch_mask] for x in batch_dict[f'batch_cls_preds_fpn{fpn_layer}']]
                    src_cls_preds = cls_preds
                    if not batch_dict[f'cls_preds_normalized_fpn{fpn_layer}']:
                        cls_preds = [torch.sigmoid(x) for x in cls_preds]

                if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                    if not isinstance(cls_preds, list):
                        cls_preds = [cls_preds]
                        multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                    else:
                        multihead_label_mapping = batch_dict['multihead_label_mapping']

                    cur_start_idx = 0
                    pred_scores, pred_labels, pred_boxes = [], [], []
                    for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                        assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                        cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                        cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                            cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                            nms_config=post_process_cfg.NMS_CONFIG,
                            score_thresh=post_process_cfg.SCORE_THRESH
                        )
                        cur_pred_labels = cur_label_mapping[cur_pred_labels]
                        pred_scores.append(cur_pred_scores)
                        pred_labels.append(cur_pred_labels)
                        pred_boxes.append(cur_pred_boxes)
                        cur_start_idx += cur_cls_preds.shape[0]

                    final_scores = torch.cat(pred_scores, dim=0)
                    final_labels = torch.cat(pred_labels, dim=0)
                    final_boxes = torch.cat(pred_boxes, dim=0)
                else:
                    cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                    # print("label_preds fpn", label_preds.shape)
                    # print("cls_preds fpn", cls_preds.shape) 127008
                    if batch_dict.get('has_class_labels', False):
                        label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                        label_preds = batch_dict[label_key][index]
                    else:
                        label_preds = label_preds + 1
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=cls_preds, box_preds=box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH,
                        nusc=self.nusc
                    )
                    if post_process_cfg.OUTPUT_RAW_SCORE:
                        max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                        selected_scores = max_cls_preds[selected]

                    final_scores = selected_scores
                    final_labels = label_preds[selected]
                    final_boxes = box_preds[selected]

                recall_dict[fpn_layer] = self.generate_recall_record(
                    box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                    recall_dict=recall_dict[fpn_layer], batch_index=index, data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST,
                    nusc=self.nusc
                )

                record_dict = {
                    f'pred_boxes_fpn{fpn_layer}': final_boxes,
                    f'pred_scores_fpn{fpn_layer}': final_scores,
                    f'pred_labels_fpn{fpn_layer}': final_labels
                }
                pred_dicts[fpn_layer].append(record_dict)

            # print("pred_dicts[fpn_layer]", pred_dicts[fpn_layer])

        # print("cls fpn", torch.sort(cls_preds)[0])
        # print('---------------------------------')
        return pred_dicts, recall_dict

    def post_processing_fuse(self, batch_dict, fpn_layers=[]):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        pred_dicts_fuse = []
        recall_dict_fuse = {}

        # print("batch_dict in ", batch_dict)
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']

        for index in range(batch_size):

            ######## fpn det ########

            cls_preds = {}
            box_preds = {}
            src_cls_preds = {}
            src_box_preds = {}

            for fpn_layer in fpn_layers:
                if batch_dict.get('batch_index', None) is not None:
                    assert batch_dict[f'batch_box_preds_fpn{fpn_layer}'].shape.__len__() == 2
                    batch_mask = (batch_dict['batch_index'] == index)
                else:
                    assert batch_dict[f'batch_box_preds_fpn{fpn_layer}'].shape.__len__() == 3
                    batch_mask = index

                box_preds[fpn_layer] = batch_dict[f'batch_box_preds_fpn{fpn_layer}'][batch_mask]
                src_box_preds[fpn_layer] = box_preds[fpn_layer]

                if not isinstance(batch_dict[f'batch_cls_preds_fpn{fpn_layer}'], list):
                    cls_preds[fpn_layer] = batch_dict[f'batch_cls_preds_fpn{fpn_layer}'][batch_mask]

                    src_cls_preds[fpn_layer] = cls_preds[fpn_layer]
                    assert cls_preds[fpn_layer].shape[1] in [1, self.num_class]

                    if not batch_dict[f'cls_preds_normalized_fpn{fpn_layer}']:
                        cls_preds[fpn_layer] = torch.sigmoid(cls_preds[fpn_layer])
                else:
                    cls_preds[fpn_layer] = [x[batch_mask] for x in batch_dict[f'batch_cls_preds_fpn{fpn_layer}']]
                    src_cls_preds[fpn_layer] = cls_preds[fpn_layer]
                    if not batch_dict[f'cls_preds_normalized_fpn{fpn_layer}']:
                        cls_preds[fpn_layer] = [torch.sigmoid(x) for x in cls_preds[fpn_layer]]

            cls_preds_fuse = torch.cat([v for _, v in cls_preds.items()], dim=0)
            box_preds_fuse = torch.cat([v for _, v in box_preds.items()], dim=0)
            src_cls_preds_fuse = torch.cat([v for _, v in src_cls_preds.items()], dim=0)
            src_box_preds_fuse = torch.cat([v for _, v in src_box_preds.items()], dim=0)

            ######## original det ########
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            # print("batch_dict['batch_box_preds']", batch_dict['batch_box_preds'])
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            cls_preds_fuse = torch.cat((cls_preds_fuse, cls_preds), dim=0)
            box_preds_fuse = torch.cat((box_preds_fuse, box_preds), dim=0)
            src_cls_preds_fuse = torch.cat((src_cls_preds_fuse, src_cls_preds), dim=0)
            src_box_preds_fuse = torch.cat((src_box_preds_fuse, src_box_preds), dim=0)

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds_fuse, list):
                    cls_preds_fuse = [cls_preds_fuse]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds_fuse[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores_fuse, pred_labels_fuse, pred_boxes_fuse = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds_fuse, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds_fuse[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores_fuse.append(cur_pred_scores)
                    pred_labels_fuse.append(cur_pred_labels)
                    pred_boxes_fuse.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores_fuse = torch.cat(pred_scores_fuse, dim=0)
                final_labels_fuse = torch.cat(pred_labels_fuse, dim=0)
                final_boxes_fuse = torch.cat(pred_boxes_fuse, dim=0)
            else:
                cls_preds_fuse, label_preds_fuse = torch.max(cls_preds_fuse, dim=-1)
                # print("label_preds fpn", label_preds.shape)
                # print("cls_preds fpn", cls_preds.shape) 127008
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds_fuse = batch_dict[label_key][index]
                else:
                    label_preds_fuse = label_preds_fuse + 1

                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds_fuse, box_preds=box_preds_fuse,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH,
                    nusc=self.nusc
                )
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds_fuse, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores_fuse = selected_scores
                final_labels_fuse = label_preds_fuse[selected]
                final_boxes_fuse = box_preds_fuse[selected]

            recall_dict_fuse = self.generate_recall_record(
                box_preds=final_boxes_fuse if 'rois' not in batch_dict else src_box_preds_fuse,
                recall_dict=recall_dict_fuse, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST,
                nusc=self.nusc
            )

            record_dict = {
                f'pred_boxes_fuse': final_boxes_fuse,
                f'pred_scores_fuse': final_scores_fuse,
                f'pred_labels_fuse': final_labels_fuse
            }
            pred_dicts_fuse.append(record_dict)

        return pred_dicts_fuse, recall_dict_fuse

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None, nusc=False):
        if nusc:
            box_size = 9
        else:
            box_size = 7

        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds, cur_gt[:, 0:box_size], nusc=nusc)
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois, cur_gt[:, 0:box_size], nusc=nusc)

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None, dom_optimizer=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        print("filename", filename)
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        # print("det", self)
        # print("checkpoint['model_state']", checkpoint['model_state'])

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if dom_optimizer is not None:
            if 'dom_optimizer_state' in checkpoint and checkpoint['dom_optimizer_state'] is not None:
                logger.info('==> Loading dom_optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                dom_optimizer.load_state_dict(checkpoint['dom_optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                dom_optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(dom_optimizer_filename):
                    dom_optimizer_ckpt = torch.load(dom_optimizer_filename, map_location=loc_type)
                    dom_optimizer.load_state_dict(dom_optimizer_ckpt['dom_optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
