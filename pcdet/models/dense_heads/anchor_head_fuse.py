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

class AnchorHeadFuse(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, nusc=False, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training, nusc=nusc, **kwargs
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)


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


        # if self.model_cfg.get('PREDBBX_FUSE_TWO_FEAT', None):
        #     self.pred_fuse = True:
        # else:
        #     self.pred_fuse = False

        self.num_keypoints = self.model_cfg.NUM_KEYPOINTS
        self.domain_pool = nn.AdaptiveAvgPool2d(1)
        # if self.point_interpolation:
        #     self.point_fc = nn.Sequential(nn.Linear(self.num_keypoints, input_channels+self.point_features_dim), nn.ReLU(True), nn.Dropout())
        # else:
        self.point_fc = nn.Sequential(nn.Linear(self.num_keypoints, input_channels+self.point_features_dim), nn.ReLU(True), nn.Dropout())

        # print("input_channels", input_channels)
        

        if self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None) is not None:
            # self.domain_classifier = nn.Sequential(nn.Linear(input_channels*2, 1024), 
            #                                     nn.ReLU(True), nn.Dropout(),
            #                                     nn.Linear(1024, 1024), nn.ReLU(True),
            #                                     nn.Dropout(), nn.Linear(1024, 1))

            if not self.point_interpolation:
                self.domain_classifier = nn.Sequential(nn.Linear(input_channels, 1024), 
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(1024, 1024), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(1024, 1))
            else:

                self.domain_classifier = nn.Sequential(nn.Linear(input_channels+self.point_features_dim, 1024), 
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(1024, 1024), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(1024, 1))

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
    
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
        else:
            dom_src = None
            
        # print("conv_cls", self.conv_cls)
        # print("num_anchors_per_location before", self.num_anchors_per_location)
        # print("fuse data_dict", data_dict)
        spatial_features_2d = data_dict['spatial_features_2d']
        if self.point_interpolation:
            batch_size, _, bev_x, bev_y = spatial_features_2d.shape 
            point_feat_dim = data_dict[f'point_features'].shape[-1]

            # 
            if self.fast_interpolation:
                interpolated_bev_features = self.interpolate_to_bev_features_fast(data_dict[f'point_coords'][:, 1:4], data_dict[f'point_features'], spatial_features_2d.shape, data_dict['spatial_features_stride'])
            else:
                interpolated_bev_features = self.interpolate_to_bev_features(data_dict[f'point_coords'][:, 1:4], data_dict[f'point_features'], spatial_features_2d.shape, data_dict['spatial_features_stride'])

            spatial_features_2d = torch.cat((spatial_features_2d, interpolated_bev_features), dim=1)

        if self.dom_cat_point:
            point_features_2d = data_dict['point_features']

            point_features_avg = torch.mean(point_features_2d, -1)
            # print("point_features_avg", point_features_avg.shape)
            batch_point_features = point_features_avg.view(-1, self.num_keypoints)

        # print("spatial_features_2d", spatial_features_2d.shape) # 2,512,126,126
        # print("batch_point_features", batch_point_features.shape) # 2,2048

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

        if 'dom_img' in t_mode:

            # print("spatial_features_2d", spatial_features_2d.shape)

            x_pool = self.domain_pool(spatial_features_2d).squeeze(-1).squeeze(-1)
            # 2, 640, 126, 126
            # print("x_pool", x_pool.shape)

            if self.dom_cat_point:
                x_pool_point = self.point_fc(batch_point_features)
                x_pool = torch.cat((x_pool, x_pool_point),dim=-1) #_joint

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

