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

class AnchorHeadSingleFPN(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, nusc=False, input_channels_fpn=None, num_fpn_up=0, num_fpn_downup=0, fpn_layers=[]):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training, nusc=nusc,
            num_fpn_up=num_fpn_up, num_fpn_downup=num_fpn_downup, fpn_layers=fpn_layers
        )
        # print("fpn_layers", fpn_layers)
        # print("input_channels", input_channels) 512
        self.num_anchors_per_location = sum(self.num_anchors_per_location)


        #####################
        if not self.fpn_only:
            self.conv_cls = nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.num_class,
                kernel_size=1
            )
            self.conv_box = nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.box_coder.code_size,
                kernel_size=1
            )

        ######## FPN #########
        self.input_channels_fpn = input_channels_fpn
        
        self.conv_cls_fpn = nn.ModuleDict()
        self.conv_box_fpn = nn.ModuleDict()

        for layer in self.fpn_layers:

            self.num_anchors_per_location_fpn[layer] = sum(self.num_anchors_per_location_fpn[layer]) # 2, 7

            self.conv_cls_fpn[layer] = nn.Conv2d(
                self.input_channels_fpn[layer], self.num_anchors_per_location_fpn[layer] * self.num_class,
                kernel_size=1
            )# 512 -> 2
            self.conv_box_fpn[layer] = nn.Conv2d(
                self.input_channels_fpn[layer], self.num_anchors_per_location_fpn[layer] * self.box_coder.code_size,
                kernel_size=1
            )# 512 -> 14

        ######### dir cls #########
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            if not self.fpn_only:
                self.conv_dir_cls = nn.Conv2d(
                    input_channels,
                    self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                    kernel_size=1
                )

            self.conv_dir_cls_fpn = nn.ModuleDict()

            for layer in self.fpn_layers:
                self.conv_dir_cls_fpn[layer] = nn.Conv2d(
                    self.input_channels_fpn[layer],
                    self.num_anchors_per_location_fpn[layer] * self.model_cfg.NUM_DIR_BINS,
                    kernel_size=1
                )

        else:
            self.conv_dir_cls = None
            for layer in self.fpn_layers:
                self.conv_dir_cls_fpn[layer] = None

        # print("USE_DOMAIN_CLASSIFIER", self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None))
        if self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None):
            if not self.fpn_only:
                self.domain_pool = nn.AdaptiveAvgPool2d(1)
                self.domain_classifier = nn.Sequential(nn.Linear(input_channels, 1024), 
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(1024, 1024), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(1024, 1))
                
            self.domain_pool_fpn = nn.ModuleDict()
            self.domain_classifier_fpn = nn.ModuleDict()

            for layer in self.fpn_layers:
                self.domain_pool_fpn[layer] = nn.AdaptiveAvgPool2d(1)
                self.domain_classifier_fpn[layer] = nn.Sequential(nn.Linear(self.input_channels_fpn[layer], 1024), 
                                                nn.ReLU(True), nn.Dropout(),
                                                nn.Linear(1024, 1024), nn.ReLU(True),
                                                nn.Dropout(), nn.Linear(1024, 1))


        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

        for layer in self.fpn_layers:
            nn.init.constant_(self.conv_cls_fpn[layer].bias, -np.log((1 - pi) / pi))
            nn.init.normal_(self.conv_box_fpn[layer].weight, mean=0, std=0.001)
    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        # print("spatial_features_2d", spatial_features_2d.shape) 126
        t_mode = data_dict['t_mode']
        l = data_dict['l']

        if 'pseudo' in t_mode:
            pseudo = True
        else:
            pseudo = False
        
        if not self.fpn_only:
            cls_preds = self.conv_cls(spatial_features_2d)
            box_preds = self.conv_box(spatial_features_2d)

            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

            # print("cls_preds", cls_preds.shape) # 126, 126, 2
            # print("box_preds", box_preds.shape) # 126, 126, 14

            self.forward_ret_dict['cls_preds'] = cls_preds
            self.forward_ret_dict['box_preds'] = box_preds

            if self.conv_dir_cls is not None:
                dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
                self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
            else:
                dir_cls_preds = None

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
        
                # print("batch_cls_preds", batch_cls_preds)
                # print("batch_box_preds", batch_box_preds)
                # print("data_dict", data_dict['batch_cls_preds'])
            
            ##########################################
            if 'dom_img' in t_mode:

                if t_mode == 'dom_img_src':
                    dom_src = True
                elif t_mode == 'dom_img_tgt':
                    dom_src = False
                else:
                    dom_src = None

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

            
        ##################################################

        if self.num_fpn_up + self.num_fpn_downup > 0:
            # print("fpn")
            for layer in self.fpn_layers:
                spatial_features_2d = data_dict[f'spatial_features_2d_fpn{layer}']
                
                cls_preds = self.conv_cls_fpn[layer](spatial_features_2d)
                box_preds = self.conv_box_fpn[layer](spatial_features_2d)

                cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
                box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

                # print("cls_preds2", cls_preds.shape) # 1, 252, 252, 2
                # print("box_preds2", box_preds.shape) # 1, 252, 252, 14

                self.forward_ret_dict[f'cls_preds_fpn{layer}'] = cls_preds
                self.forward_ret_dict[f'box_preds_fpn{layer}'] = box_preds

                if self.conv_dir_cls is not None:
                    dir_cls_preds = self.conv_dir_cls_fpn[layer](spatial_features_2d)
                    dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
                    self.forward_ret_dict[f'dir_cls_preds_fpn{layer}'] = dir_cls_preds
                else:
                    dir_cls_preds = None

                if self.training:
                    if pseudo:
                        pseudo_weights = data_dict['pseudo_weights']
                    else:
                        pseudo_weights = None

                    targets_dict_fpn = self.assign_targets(
                        gt_boxes=data_dict['gt_boxes'],
                        pseudo=pseudo,
                        pseudo_weights=pseudo_weights,
                        fpn_layer=layer
                    )
                
                    self.forward_ret_dict.update(targets_dict_fpn)

                if not self.training or self.predict_boxes_when_training:
                    batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                        batch_size=data_dict['batch_size'],
                        cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds,
                        fpn_layer=layer
                    )
                    data_dict[f'batch_cls_preds_fpn{layer}'] = batch_cls_preds
                    data_dict[f'batch_box_preds_fpn{layer}'] = batch_box_preds
                    data_dict[f'cls_preds_normalized_fpn{layer}'] = False

                    # print("data_dict fpn", data_dict[f'batch_cls_preds_fpn{layer}'])

                ##########################################
                if 'dom_img' in t_mode:

                    x_pool = self.domain_pool_fpn[layer](spatial_features_2d).view(spatial_features_2d.size(0), -1)
                    x_reverse = grad_reverse(x_pool, l*-1)
                    dom_img_preds = self.domain_classifier_fpn[layer](x_reverse).squeeze(-1)

                    self.forward_ret_dict[f'dom_img_preds_fpn{layer}'] = dom_img_preds            
                    
                    if self.training:
                        targets_dict_dom = self.assign_targets(
                                gt_boxes=data_dict['gt_boxes'],
                                dom_src=dom_src,
                                pseudo=pseudo,
                                fpn_layer=layer
                        )
                        self.forward_ret_dict.update(targets_dict_dom)

        
        
        # print("self.forward_ret_dict", self.forward_ret_dict)
        return data_dict
