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

class AnchorHeadFuseRangeFPN(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, nusc=False, input_channels_fpn=None, num_fpn_up=0, num_fpn_downup=0, fpn_layers=[], voxel_size=[0.1, 0.1, 0.2], **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training,nusc=nusc, num_fpn_up=num_fpn_up, num_fpn_downup=num_fpn_downup, fpn_layers=fpn_layers, voxel_size=voxel_size, **kwargs
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.input_channels_fpn = input_channels_fpn

        self.context_num = self.num_fpn_up + self.num_fpn_downup
        
        if self.num_fpn_downup > 0:
            self.context_num += 1
        #256 512
        self.context_num += 2 # point context 256*2=512
        
        
        self.num_keypoints_range = self.model_cfg.get('NUM_KEYPOINTS_RANGE', {})
        self.range_keys = self.num_keypoints_range.keys()

        ###################
        if not self.fpn_only:
            if not self.point_interpolation:
                self.conv_cls = nn.Conv2d(
                    input_channels, self.num_anchors_per_location * self.num_class,
                    kernel_size=1
                )

                self.conv_box = nn.Conv2d(
                    input_channels, self.num_anchors_per_location * self.box_coder.code_size,
                    kernel_size=1
                )
                if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
                    self.conv_dir_cls = nn.Conv2d(
                        input_channels,
                        self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                        kernel_size=1
                    )
                else:
                    self.conv_dir_cls = None
                    
            else:
                self.conv_cls = nn.Conv2d(
                    input_channels+self.point_features_dim, self.num_anchors_per_location * self.num_class,
                    kernel_size=1
                )

                self.conv_box = nn.Conv2d(
                    input_channels+self.point_features_dim, self.num_anchors_per_location * self.box_coder.code_size,
                    kernel_size=1
                )


                if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
                    self.conv_dir_cls = nn.Conv2d(
                        input_channels+self.point_features_dim,
                        self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                        kernel_size=1
                    )
                else:
                    self.conv_dir_cls = None

        ######## FPN #########
        
        self.conv_cls_fpn = nn.ModuleDict()
        self.conv_box_fpn = nn.ModuleDict()
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls_fpn = nn.ModuleDict()

        for layer in self.fpn_layers:
            self.num_anchors_per_location_fpn[layer] = sum(self.num_anchors_per_location_fpn[layer]) # 2, 7
            self.conv_cls_fpn[layer] = nn.Conv2d(
                self.input_channels_fpn[layer]+self.point_features_dim, self.num_anchors_per_location_fpn[layer] * self.num_class,
                kernel_size=1
            )# 512 -> 2
            self.conv_box_fpn[layer] = nn.Conv2d(
                self.input_channels_fpn[layer]+self.point_features_dim, self.num_anchors_per_location_fpn[layer] * self.box_coder.code_size,
                kernel_size=1
            )# 512 -> 14

            if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
                self.conv_dir_cls_fpn[layer] = nn.Conv2d(
                    self.input_channels_fpn[layer]+self.point_features_dim,
                    self.num_anchors_per_location_fpn[layer] * self.model_cfg.NUM_DIR_BINS,
                    kernel_size=1
                )#+self.context_num*256

        self.domain_pool = nn.AdaptiveAvgPool2d(1)

        self.point_fc_range = nn.ModuleDict()
        # self.domain_classifier_range = nn.ModuleDict()

        for i in self.range_keys:
            self.point_fc_range[i] = nn.Sequential(nn.Linear(self.num_keypoints_range[i], input_channels), nn.ReLU(True), nn.Dropout())

        if self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None) is not None:
            self.domain_classifier = nn.Sequential(nn.Linear(input_channels*(1+len(self.range_keys)), 1024), 
                                                nn.ReLU(True), nn.Dropout(),
                                                nn.Linear(1024, 1024), nn.ReLU(True),
                                                nn.Dropout(), nn.Linear(1024, 1))

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        
        # print("conv_cls", self.conv_cls)
        # print("num_anchors_per_location before", self.num_anchors_per_location)
        # print("fuse data_dict", data_dict)
        # print("point_features_joint", point_features_joint)

        spatial_features_2d = data_dict['spatial_features_2d']

        range_fpn_dict = {'short': '3', 'mid': '4', 'long': '5'}

        if self.point_interpolation:
            batch_size, _, bev_x, bev_y = spatial_features_2d.shape 
            point_feat_dim = data_dict[f'point_features_mid'].shape[-1]

            # interpolated_bev_features_joint = torch.zeros((batch_size, point_feat_dim, bev_x, bev_y)).cuda()

            bev_feat_range = {}

            for i in self.range_keys:
                # print("data_dict[f'point_coords_{i}'][:, 1:4]", data_dict[f'point_coords_{i}'][:, 1:4])
                # print("data_dict[f'point_features_{i}']", data_dict[f'point_features_{i}'].shape)
                bev_feat_range[i] = self.interpolate_to_bev_features(data_dict[f'point_coords_{i}'][:, 1:4], data_dict[f'point_features_{i}'], data_dict[f'spatial_features_2d_fpn{range_fpn_dict[i]}'].shape, data_dict['spatial_features_stride'])

                data_dict[f'spatial_features_2d_fpn{range_fpn_dict[i]}'] = torch.cat((data_dict[f'spatial_features_2d_fpn{range_fpn_dict[i]}'], bev_feat_range[i]), dim=1)

                # print("i", i)
                # print("bev_feat_range[i]", bev_feat_range[i].shape)
            
            interpolated_bev_features_joint = self.interpolate_to_bev_features(data_dict[f'point_coords_joint'][:, 1:4], data_dict[f'point_features_joint'], data_dict[f'spatial_features_2d'].shape, data_dict['spatial_features_stride'])


            # print("spatial_features_2d", spatial_features_2d.shape)
            spatial_features_2d = torch.cat((spatial_features_2d, interpolated_bev_features_joint), dim=1)
            # print("spatial_features_2d aft", spatial_features_2d.shape)

        # print("bev_feat_range", bev_feat_range)
        # print("spatial_features_2d", spatial_features_2d.shape) # 2,512,126,126
        # print("batch_point_features", batch_point_features.shape) # 2,2048

        t_mode = data_dict['t_mode']
        l = data_dict['l']

        if t_mode == 'dom_img_src':
            dom_src = True
        elif t_mode == 'dom_img_tgt':
            dom_src = False
        else:
            dom_src = None

        if 'pseudo' in t_mode:
            pseudo = True
        else:
            pseudo = False

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        
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



        ########################################
        if 'dom_img' in t_mode:
            x_pool = self.domain_pool(spatial_features_2d).view(spatial_features_2d.size(0), -1)

            for i in self.range_keys:
                point_features_2d = data_dict[f'point_features_{i}']
                point_features_avg = torch.mean(point_features_2d, -1)
                batch_point_features = point_features_avg.view(-1, self.num_keypoints_range[i])

                x_pool_point = self.point_fc_range[i](batch_point_features)

                x_pool = torch.cat((x_pool, x_pool_point),dim=-1)

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

