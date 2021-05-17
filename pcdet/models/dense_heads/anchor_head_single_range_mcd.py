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

class AnchorHeadSingleRangeMCD(AnchorHeadTemplate):
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

        self.rangeinv = self.model_cfg.get('RANGE_INV', False)
        self.keep_x = self.model_cfg.get('KEEP_X', False)
        self.keep_y = self.model_cfg.get('KEEP_Y', False)
        self.rm_thresh = self.model_cfg.get('RM_THRESH', 0)

        if self.rangeinv:
            self.conv_range = nn.Conv2d(
                input_channels, 1,
                kernel_size=1
            )
            #nn.Sequential(



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


        # if self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None) is not None:
    
        if self.range_da > 0:
            self.domain_pool = nn.AdaptiveAvgPool2d(1)
            self.domain_classifier_range = nn.ModuleDict()
            for n in range(self.range_da):
                self.domain_classifier_range[str(n)] = nn.Sequential(nn.Linear(input_channels, 1024), 
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(1024, 1024), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(1024, 1))

        elif self.interval_da > 0:
            self.domain_pool = nn.AdaptiveAvgPool2d(1)
            self.domain_classifier_interval = nn.ModuleDict()
            for n in range(self.interval_da):
                self.domain_classifier_interval[str(n)] = nn.Sequential(nn.Linear(input_channels, 1024), 
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(1024, 1024), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(1024, 1))

        else:
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
        
        if self.mcd:
            if t_mode == 'dom_img_tgt':
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
                self.forward_ret_dict[f'{prefix}dir_cls_preds_1'] = dir_cls_preds
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
                self.forward_ret_dict[f'{prefix}dir_cls_preds_2'] = dir_cls_preds2
            else:
                dir_cls_preds2 = None

        else:

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

        if self.rangeinv:
            # print("spatial_features_2d", spatial_features_2d.shape) #512,128,128
            thresh = self.rm_thresh

            start_dim = int(spatial_features_2d.shape[-1]/4.)
            mid_dim = int(spatial_features_2d.shape[-1]/2.) 
            end_dim = start_dim+int(spatial_features_2d.shape[-1]/2.)

            near_idx = torch.LongTensor([i for i in range(start_dim, mid_dim-thresh)]+[i for i in range(mid_dim+thresh, end_dim)])
            far_idx = torch.LongTensor([i for i in range(start_dim)]+[i for i in range(end_dim, spatial_features_2d.shape[-1])])

            if self.keep_x:
                near_feat_2d = spatial_features_2d[:,:,:,near_idx]
                far_feat_2d = spatial_features_2d[:,:,:, far_idx]
            elif self.keep_y:
                near_feat_2d = spatial_features_2d[:,:,near_idx,:]
                far_feat_2d = spatial_features_2d[:,:,far_idx,:]

            near_feat_2d_reverse = grad_reverse(near_feat_2d, l*-1)
            range_pred_near = self.conv_range(near_feat_2d_reverse)
            # print("near_range_pred", near_range_pred.shape)
            far_feat_2d_reverse = grad_reverse(far_feat_2d, l*-1)
            range_pred_far = self.conv_range(far_feat_2d_reverse)
            # print("far_range_pred", far_range_pred.shape)

            range_labels_near = torch.ones((range_pred_near.shape), dtype=torch.float32, device=spatial_features_2d.device)

            range_labels_far = torch.zeros((range_pred_far.shape), dtype=torch.float32, device=spatial_features_2d.device)

            targets_dict_range = {
                'range_pred_near': range_pred_near,
                'range_pred_far': range_pred_far,
                'range_labels_near': range_labels_near,
                'range_labels_far': range_labels_far,
            }
            self.forward_ret_dict.update(targets_dict_range)

        if 'dom_img' in t_mode:

            if t_mode == 'dom_img_src':
                dom_src = True
            elif t_mode == 'dom_img_tgt':
                dom_src = False
            else:
                dom_src = None

            # 
            if self.range_da > 0:
                mid_dim = int(spatial_features_2d.shape[-1]/2.) 
                range_interval = int(spatial_features_2d.shape[-1]/(2*self.range_da))

                start_dim = {}
                mid1_dim = {}
                mid2_dim = {}
                end_dim = {}
                interval_idx = {}
                interval_feat = {}

                # for each range 0,1,2,3 (4)
                
                for n in range(self.range_da): # 0,1
                    start_dim[n] = mid_dim - range_interval*(n+1) #  2-1=1, 2-2=0
                    mid1_dim[n] = mid_dim - range_interval*n # 2-0=2 2-1=1 #int(spatial_features_2d.shape[-1]/2.) 
                    mid2_dim[n] = mid_dim + range_interval*n # 2+0=2 2+1=3
                    end_dim[n] = mid_dim + range_interval*(n+1) # 2+1=3 2+2=4

                    print("range n", n)
                    print("start_dim[n]", start_dim[n])
                    print("mid1_dim[n]", mid1_dim[n])
                    print("mid2_dim[n]", mid2_dim[n])
                    print("end_dim[n]", end_dim[n])

                    interval_idx[n] = torch.LongTensor([i for i in range(start_dim[n], mid1_dim[n])]+[i for i in range(mid2_dim[n], end_dim[n])])
            
                    if self.keep_x:
                        interval_feat[n] = spatial_features_2d[:,:,:,interval_idx[n]]
                    elif self.keep_y:
                        interval_feat[n] = spatial_features_2d[:,:,interval_idx[n],:]

                    
                    x_pool = self.domain_pool(interval_feat[n]).view(interval_feat[n].size(0), -1)
                    x_reverse = grad_reverse(x_pool, l*-1)
                    dom_img_preds = self.domain_classifier_range[str(n)](x_reverse).squeeze(-1)

                    self.forward_ret_dict[f'dom_img_preds_range{n}'] = dom_img_preds            
                    
                if self.training:
                    targets_dict_dom = self.assign_targets(
                            gt_boxes=data_dict['gt_boxes'],
                            dom_src=dom_src,
                            pseudo=pseudo
                    )
                    self.forward_ret_dict.update(targets_dict_dom)

            elif self.interval_da > 0:

                # mid_dim = int(spatial_features_2d.shape[-1]/2.) 
                range_interval = int(spatial_features_2d.shape[-1]/self.interval_da)

                start_dim = {}
                # mid1_dim = {}
                # mid2_dim = {}
                end_dim = {}
                interval_idx = {}
                interval_feat = {}

                # for each range 0,1,2,3 (4)
                
                for n in range(self.interval_da): # 0,1
                    start_dim[n] = range_interval*n #  2-1=1, 2-2=0
                    # mid1_dim[n] = mid_dim - range_interval*n # 2-0=2 2-1=1 #int(spatial_features_2d.shape[-1]/2.) 
                    # mid2_dim[n] = mid_dim + range_interval*n # 2+0=2 2+1=3
                    end_dim[n] = range_interval*(n+1) # 2+1=3 2+2=4

                    interval_idx[n] = torch.LongTensor([i for i in range(start_dim[n], end_dim[n])])

                    # print("spatial_features_2d", spatial_features_2d.shape)
                    if self.keep_x:
                        interval_feat[n] = spatial_features_2d[:,:,:,interval_idx[n]]
                    elif self.keep_y:
                        interval_feat[n] = spatial_features_2d[:,:,interval_idx[n],:]

                    
                    # print("interval_feat[n]", interval_feat[n].shape)
                    x_pool = self.domain_pool(interval_feat[n]).view(interval_feat[n].size(0), -1)
                    # print("x_pool[n]", x_pool.shape)
                    x_reverse = grad_reverse(x_pool, l*-1)
                    dom_img_preds = self.domain_classifier_interval[str(n)](x_reverse).squeeze(-1)

                    self.forward_ret_dict[f'dom_img_preds_interval{n}'] = dom_img_preds            
                    
                if self.training:
                    targets_dict_dom = self.assign_targets(
                            gt_boxes=data_dict['gt_boxes'],
                            dom_src=dom_src,
                            pseudo=pseudo
                    )
                    self.forward_ret_dict.update(targets_dict_dom)


            else:
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