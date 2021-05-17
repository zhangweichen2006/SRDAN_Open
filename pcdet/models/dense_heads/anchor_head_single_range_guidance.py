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

class AnchorHeadSingleRangeGuidance(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, nusc=False,  fpn_layers=[], **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training, nusc=nusc, fpn_layers=fpn_layers
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        if self.range_guidance:
            if self.range_guidance_dom_only:
                input_channels_dom = input_channels + 2
            else:
                input_channels = input_channels + 2 + 256
                input_channels_dom = input_channels - 256
        else:
            input_channels_dom = input_channels

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        self.rangeinv = self.model_cfg.get('RANGE_INV', False)
        self.keep_x = self.model_cfg.get('KEEP_X', False)
        self.keep_y = self.model_cfg.get('KEEP_Y', False)
        self.keep_xy = self.model_cfg.get('KEEP_XY', False)
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
        else:
            self.conv_dir_cls = None


        # if self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None) is not None:

        if self.range_da > 0:
            self.domain_pool = nn.AdaptiveAvgPool2d(1)
            self.domain_classifier_range = nn.ModuleDict()
            for n in range(0+self.remove_near_range, self.range_da-self.remove_far_range):
                self.domain_classifier_range[str(n)] = nn.Sequential(nn.Linear(input_channels, 1024),
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(1024, 256), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(256, 1))
            if self.keep_xy:
                self.domain_classifier_range2 = nn.ModuleDict()
                for n in range(0+self.remove_near_range2, self.range_da-self.remove_far_range2):
                    self.domain_classifier_range2[str(n)] = nn.Sequential(nn.Linear(input_channels, 1024),
                                                        nn.ReLU(True), nn.Dropout(),
                                                        nn.Linear(1024, 256), nn.ReLU(True),
                                                        nn.Dropout(), nn.Linear(256, 1))

        elif self.interval_da > 0:
            self.domain_pool = nn.AdaptiveAvgPool2d(1)
            self.domain_classifier_interval = nn.ModuleDict()
            for n in range(self.interval_da):
                self.domain_classifier_interval[str(n)] = nn.Sequential(nn.Linear(input_channels, 1024),
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(1024, 256), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(256, 1))

        else:
            self.domain_pool = nn.AdaptiveAvgPool2d(1)
            self.domain_classifier = nn.Sequential(nn.Linear(input_channels_dom, 1024),
                                                nn.ReLU(True), nn.Dropout(),
                                                nn.Linear(1024, 256), nn.ReLU(True),
                                                nn.Dropout(), nn.Linear(256, 1))

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        t_mode = data_dict['t_mode']
        l = data_dict['l']
        # print("t_mode", t_mode)

        if 'pseudo' in t_mode:
            pseudo = True
        else:
            pseudo = False

        spatial_features_2d = data_dict['spatial_features_2d']

        # print("spatial_features_2d", spatial_features_2d.shape) 126
        # print('range ctx',self.range_guidance)

        if t_mode == 'tsne':
            if self.range_da > 0:
                mid_dim = int(spatial_features_2d.shape[-1]/2.)
                range_interval = int(spatial_features_2d.shape[-1]/(2*self.range_da))

                start_dim = {}
                mid1_dim = {}
                mid2_dim = {}
                end_dim = {}
                interval_idx = {}
                interval_feat = {}
                if self.keep_xy:
                    interval_feat2 = {}

                # for each range 0,1,2,3 (4)

                for n in range(0+self.remove_near_range, self.range_da-self.remove_far_range): # no0,1
                    start_dim[n] = mid_dim - range_interval*(n+1) #  2-1=1, 2-2=0
                    mid1_dim[n] = mid_dim - range_interval*n # 2-0=2 2-1=1 #int(spatial_features_2d.shape[-1]/2.)
                    mid2_dim[n] = mid_dim + range_interval*n # 2+0=2 2+1=3
                    end_dim[n] = mid_dim + range_interval*(n+1) # 2+1=3 2+2=4

                    interval_idx[n] = torch.LongTensor([i for i in range(start_dim[n], mid1_dim[n])]+[i for i in range(mid2_dim[n], end_dim[n])])

                    feat1 = spatial_features_2d[:,:,:,interval_idx[n]]
                    feat1 = self.domain_pool(feat1).view(feat1.size(0), -1)
                    data_dict[f'spatial_features_2d_x_{n}'] = feat1

                    feat2 = spatial_features_2d[:,:,interval_idx[n],:]
                    feat2 = self.domain_pool(feat2).view(feat2.size(0), -1)
                    data_dict[f'spatial_features_2d_y_{n}'] = feat2


        if self.range_guidance and not self.range_guidance_dom_only:
            total_range = spatial_features_2d.shape[-1]
            half_range = int(spatial_features_2d.shape[-1] * 0.5)

            # x_range = torch.zeros((total_range, total_range)).cuda()
            # y_range = torch.zeros((total_range, total_range)).cuda()
            # for i in range(-half_range, half_range):
            #     for j in range(-half_range, half_range):
            #         x_range[i+half_range,j+half_range] = abs(i+0.5)
            #         y_range[i+half_range,j+half_range] = abs(j+0.5)
            x_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1, total_range, 1).cuda()
            # print("x_range", x_range)
            y_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1,1,total_range).cuda()
            # print('x_range',x_range[0,-1])
            # print('y_range',y_range[0,-1])
            # print("spatial_features_2d 0", spatial_features_2d.shape)
            # x_range = x_range.unsqueeze(0).unsqueeze(0).repeat((spatial_features_2d.shape[0],1,1,1))
            # y_range = y_range.unsqueeze(0).unsqueeze(0).repeat((spatial_features_2d.shape[0],1,1,1))
            # print('x_range',x_range.shape)
            # print('y_range',y_range.shape)
            spatial_features_2d = torch.cat((spatial_features_2d, x_range, y_range), dim=1)
            # print("spatial_features_2d", spatial_features_2d.shape)

        # print("t_mode", t_mode)
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
                if self.keep_xy:
                    interval_feat2 = {}

                # for each range 0,1,2,3 (4)

                for n in range(0+self.remove_near_range, self.range_da-self.remove_far_range): # no0,1
                    start_dim[n] = mid_dim - range_interval*(n+1) #  2-1=1, 2-2=0
                    mid1_dim[n] = mid_dim - range_interval*n # 2-0=2 2-1=1 #int(spatial_features_2d.shape[-1]/2.)
                    mid2_dim[n] = mid_dim + range_interval*n # 2+0=2 2+1=3
                    end_dim[n] = mid_dim + range_interval*(n+1) # 2+1=3 2+2=4

                    # print("range n", n)
                    # print("start_dim[n]", start_dim[n])
                    # print("mid1_dim[n]", mid1_dim[n])
                    # print("mid2_dim[n]", mid2_dim[n])
                    # print("end_dim[n]", end_dim[n])

                    interval_idx[n] = torch.LongTensor([i for i in range(start_dim[n], mid1_dim[n])]+[i for i in range(mid2_dim[n], end_dim[n])])

                    if self.keep_x:
                        interval_feat[n] = spatial_features_2d[:,:,:,interval_idx[n]]
                        # self.forward_ret_dict[f'spatial_features_2d_x_{n}'] = interval_feat[n]
                    elif self.keep_y:
                        interval_feat[n] = spatial_features_2d[:,:,interval_idx[n],:]
                        # self.forward_ret_dict[f'spatial_features_2d_y_{n}'] = interval_feat[n]
                    elif self.keep_xy:
                        interval_feat[n] = spatial_features_2d[:,:,:,interval_idx[n]]
                        # self.forward_ret_dict[f'spatial_features_2d_x_{n}'] = interval_feat[n]


                    x_pool = self.domain_pool(interval_feat[n]).view(interval_feat[n].size(0), -1)
                    x_reverse = grad_reverse(x_pool, l*-1)
                    # dom_img_preds = self.domain_classifier_range[str(n)](x_reverse).squeeze(-1)
                    dom_head_context = self.domain_classifier_range[str(n)][:-2](x_reverse)#.squeeze(-1)
                    if 'dom_img_det' in t_mode:
                        data_dict['dom_head_context'] = dom_head_context

                    dom_img_preds = self.domain_classifier_range[str(n)][-2:](dom_head_context)#.squeeze(-1)

                    self.forward_ret_dict[f'dom_img_preds_range{n}'] = dom_img_preds

                    if self.keep_xy:
                        interval_feat2[n] = spatial_features_2d[:,:,interval_idx[n],:]
                        self.forward_ret_dict[f'spatial_features_2d_y_{n}'] = interval_feat2[n]

                        x_pool2 = self.domain_pool(interval_feat2[n]).view(interval_feat2[n].size(0), -1)
                        x_reverse2 = grad_reverse(x_pool2, l*-1)
                        # dom_img_preds2 = self.domain_classifier_range2[str(n)](x_reverse2).squeeze(-1)

                        dom_head_context2 = self.domain_classifier_range2[str(n)][:-2](x_reverse2)#.squeeze(-1)
                        if 'dom_img_det' in t_mode:
                            data_dict['dom_head_context2'] = dom_head_context2

                        dom_img_preds2 = self.domain_classifier_range2[str(n)][-2:](dom_head_context2)#.squeeze(-1)

                        self.forward_ret_dict[f'dom_img_preds_range{n}_2'] = dom_img_preds2

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
                    # dom_img_preds = self.domain_classifier_interval[str(n)](x_reverse).squeeze(-1)

                    dom_head_context = self.domain_classifier_interval[str(n)][:-2](x_reverse)#.squeeze(-1)
                    if 'dom_img_det' in t_mode:
                        data_dict['dom_head_context'] = dom_head_context

                    dom_img_preds = self.domain_classifier_interval[str(n)][-2:](dom_head_context)#.squeeze(-1)



                    self.forward_ret_dict[f'dom_img_preds_interval{n}'] = dom_img_preds

                if self.training:
                    targets_dict_dom = self.assign_targets(
                            gt_boxes=data_dict['gt_boxes'],
                            dom_src=dom_src,
                            pseudo=pseudo
                    )
                    self.forward_ret_dict.update(targets_dict_dom)


            else:

                if self.range_guidance and self.range_guidance_dom_only:
                    total_range = spatial_features_2d.shape[-1]
                    half_range = int(spatial_features_2d.shape[-1] * 0.5)
                    x_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1, total_range, 1).cuda()
                    y_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1,1,total_range).cuda()
                    spatial_features_2d = torch.cat((spatial_features_2d, x_range, y_range), dim=1)



                x_pool = self.domain_pool(spatial_features_2d).view(spatial_features_2d.size(0), -1)
                x_reverse = grad_reverse(x_pool, l*-1)
                # dom_img_preds = self.domain_classifier(x_reverse).squeeze(-1)

                dom_head_context = self.domain_classifier[:-2](x_reverse)

                if 'dom_img_det' in t_mode:
                    data_dict['dom_head_context'] = dom_head_context

                dom_img_preds = self.domain_classifier[-2:](dom_head_context)

                self.forward_ret_dict['dom_img_preds'] = dom_img_preds

                if self.training:
                    targets_dict_dom = self.assign_targets(
                            gt_boxes=data_dict['gt_boxes'],
                            dom_src=dom_src,
                            pseudo=pseudo
                    )
                    self.forward_ret_dict.update(targets_dict_dom)

            if 'det' not in t_mode:
                return data_dict

        dom_head_context = data_dict[f'dom_head_context']

        dom_head_context_reshape = dom_head_context.unsqueeze(-1).unsqueeze(-1).repeat(1,1,spatial_features_2d.shape[-2],spatial_features_2d.shape[-1])

        spatial_features_2d = torch.cat((spatial_features_2d, dom_head_context_reshape), dim=1)

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


        return data_dict