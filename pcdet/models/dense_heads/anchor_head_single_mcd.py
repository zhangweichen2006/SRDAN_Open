import numpy as np
import torch
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate

class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)

class AnchorHeadSingleMCD(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, nusc=False,  fpn_layers=[], **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training, nusc=nusc, fpn_layers=fpn_layers
        )
        
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        if self.mcd:
            self.conv_cls2 = nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.num_class,
                kernel_size=1
            )

            self.conv_box2 = nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.box_coder.code_size,
                kernel_size=1
            )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
            if self.mcd:
                self.conv_dir_cls2 = nn.Conv2d(
                    input_channels,
                    self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                    kernel_size=1
                )
        else:
            self.conv_dir_cls = None


        if self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None) is not None:
            self.domain_pool = nn.AdaptiveAvgPool2d(1)
            self.domain_classifier = nn.Sequential(nn.Linear(input_channels, 1024), 
                                                nn.ReLU(True), nn.Dropout(),
                                                nn.Linear(1024, 1024), nn.ReLU(True),
                                                nn.Dropout(), nn.Linear(1024, 1))

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        # print("spatial_features_2d", spatial_features_2d.shape) 126
        t_mode = data_dict['t_mode']
        l = data_dict['l']

        if 'pseudo' in t_mode:
            pseudo = True
        else:
            pseudo = False
        
        if t_mode == 'dom_img_src':
            dom_src = True
        elif t_mode == 'dom_img_tgt':
            dom_src = False
            
        if self.mcd:
            if t_mode == 'mcd_tgt':
                prefix = 'mcd_'
                spatial_features_2d = grad_reverse(spatial_features_2d, l*-1)
            else:
                prefix = ''

            cls_preds = self.conv_cls(spatial_features_2d)
            box_preds = self.conv_box(spatial_features_2d)

            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

            self.forward_ret_dict[f'{prefix}cls_preds_1'] = cls_preds
            self.forward_ret_dict[f'{prefix}box_preds_1'] = box_preds

            if self.conv_dir_cls is not None:
                dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
                self.forward_ret_dict['dir_cls_preds_1'] = dir_cls_preds
            else:
                dir_cls_preds = None
            
            cls_preds2 = self.conv_cls2(spatial_features_2d)
            box_preds2 = self.conv_box2(spatial_features_2d)

            cls_preds2 = cls_preds2.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
            box_preds2 = box_preds2.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

            self.forward_ret_dict[f'{prefix}cls_preds_2'] = cls_preds2
            self.forward_ret_dict[f'{prefix}box_preds_2'] = box_preds2

            if self.conv_dir_cls is not None:
                dir_cls_preds2 = self.conv_dir_cls2(spatial_features_2d)
                dir_cls_preds2 = dir_cls_preds2.permute(0, 2, 3, 1).contiguous()
                self.forward_ret_dict['dir_cls_preds_2'] = dir_cls_preds2
            else:
                dir_cls_preds2 = None

        if self.training:
            if pseudo:
                pseudo_weights = data_dict['pseudo_weights']
            else:
                pseudo_weights = None

            # print("gt_classes", data_dict['gt_classes'].shape)
            # print("gt_classes", data_dict['gt_classes'])
            # print("pseudo_weights", pseudo_weights)

            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'],
                pseudo=pseudo,
                pseudo_weights=pseudo_weights
            )
           
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        if 'dom_img' in t_mode and not self.mcd:

            x_pool = self.domain_pool(spatial_features_2d).view(spatial_features_2d.size(0), -1)
            x_reverse = grad_reverse(x_pool, l*-1)
            dom_img_preds = self.domain_classifier(x_reverse).squeeze(-1)

            self.forward_ret_dict['dom_img_preds'] = dom_img_preds            
            
            if self.training:
                targets_dict_dom = self.assign_targets(
                        gt_boxes=data_dict['gt_boxes'],
                        dom_src=dom_src,
                        pseudo=pseudo
                )
                self.forward_ret_dict.update(targets_dict_dom)

        return data_dict
