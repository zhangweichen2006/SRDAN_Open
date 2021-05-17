import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class RangeIntervalAttentionLayer(nn.Module):

    def __init__(self, num_channels, kernel_size, division=6, prior=False):
        super(RangeIntervalAttentionLayer, self).__init__()
        if prior:
            param_list = []
            for i in range(division):
                param_list.append(torch.tensor([[[i * (division-1)]]], dtype=torch.float32))
            param = torch.cat(param_list, dim=-2)
        else:
            param = torch.randn(1, division, 1)

        self.patch_param = nn.Parameter(param, requires_grad=True)

        # self.patch_param = nn.Parameter(torch.randn(1, division, 1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.kernel_size = kernel_size
        self.division = division
        self.elem = int(self.kernel_size / division)

    def forward(self, input_tensor):
        bt, c, h, w = input_tensor.size()

        input_tensor = input_tensor.view(-1, h, w)

        self.patch_matrix = self.patch_param.repeat(1, 1, self.elem).view(1, -1, 1)

        self.patch_matrix = self.patch_matrix.repeat(bt*c, 1, w)

        input_tensor = input_tensor * self.patch_matrix

        input_tensor = self.sigmoid(input_tensor).view(bt, c, h, w)

        return input_tensor

class RoadIntervalAttentionLayer(nn.Module):

    def __init__(self, num_channels, kernel_size, division=6, prior=False):
        super(RoadIntervalAttentionLayer, self).__init__()
        if prior:
            param_list = []
            for i in range(division):
                param_list.append(torch.tensor([[[i * (division-1)]]], dtype=torch.float32))
            param = torch.cat(param_list, dim=-1)
        else:
            param = torch.randn(1, 1, division)

        self.patch_param = nn.Parameter(param, requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.kernel_size = kernel_size
        self.division = division
        self.elem = int(self.kernel_size / division)

    def forward(self, input_tensor):
        bt, c, h, w = input_tensor.size()

        input_tensor = input_tensor.view(-1, h, w)

        self.patch_matrix = self.patch_param.repeat(1, self.elem, 1).permute(0,2,1).contiguous().view(1,-1,1).permute(0,2,1)

        self.patch_matrix = self.patch_matrix.repeat(bt*c, h, 1)

        input_tensor = input_tensor * self.patch_matrix

        input_tensor = self.sigmoid(input_tensor).view(bt, c, h, w)

        return input_tensor

class LocationAttentionLayer(nn.Module):

    def __init__(self, num_channels, kernel_size, prior=False):
        super(LocationAttentionLayer, self).__init__()
        self.patch_matrix = nn.Parameter(torch.randn(1, kernel_size, kernel_size), requires_grad=True)
        # self.patch_conv = nn.Conv2d(num_channels, 1, kernel_size, kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        #2, 512, 126, 126
        bt, c, h, w = input_tensor.size()

        # print("input_tensor", input_tensor.shape)
        # patch_tensor = self.patch_conv(input_tensor)
        # print("patch_tensor", patch_tensor.shape)

        input_tensor = input_tensor.view(-1, h, w)
        # self.patch_matrix = self.patch_matrix.repeat(bt*c, 1, 1)
        # print("self.patch_matrix.repeat(bt*c, 1, 1)x", self.patch_matrix.repeat(bt*c, 1, 1).shape)
        input_tensor = input_tensor * self.patch_matrix.repeat(bt*c, 1, 1)

        input_tensor = self.sigmoid(input_tensor).view(bt, c, h, w)
        # print("input_tensor")
        # print("self.input_tensor", input_tensor.shape)
        return input_tensor

class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        # print("input_tensor.size()", input_tensor.size()) #2, 512, 126, 126

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        # print("out.size()", out.size()) #2, 1, 126, 126

        squeeze_tensor = self.sigmoid(out)
        # print("squeeze_tensor.size()", squeeze_tensor.size()) # 2, 1, 126, 126

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        # print("squeeze_tensor 2.size()", squeeze_tensor.size()) # 2, 1, 126, 126
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        # print("output_tensor 2.size()", output_tensor.size()) #2, 512, 126, 126
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor

class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size() #2, 512, 126, 126
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2) #2, 512, 126*126(1)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class LocalDomainClassifier(nn.Module):
    def __init__(self, input_channels=256, context=False):
        super(LocalDomainClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        self._init_weights()
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
            #m.bias.data.zero_()
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            x = self.conv3(x)
            return F.sigmoid(x),feat
        else:
            x = self.conv3(x)
            return F.sigmoid(x)

class AnchorHeadSingleRangeNewConvDom(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, nusc=False,  fpn_layers=[], **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training, nusc=nusc, fpn_layers=fpn_layers
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)


        self.voxel_det_seconv_attention = self.model_cfg.get('VOXEL_DET_SECONV_ATTENTION', False)
        self.voxel_det_se_attention = self.model_cfg.get('VOXEL_DET_SE_ATTENTION', False)
        self.voxel_det_patch_attention = self.model_cfg.get('VOXEL_DET_PATCH_ATTENTION', False)
        self.voxel_dom_seconv_attention = self.model_cfg.get('VOXEL_DOM_SECONV_ATTENTION', False)
        self.voxel_dom_se_attention = self.model_cfg.get('VOXEL_DOM_SE_ATTENTION', False)
        self.voxel_dom_patch_attention = self.model_cfg.get('VOXEL_DOM_PATCH_ATTENTION', False)
        self.voxel_dom_rangeinterval_attention = self.model_cfg.get('VOXEL_DOM_RANGEINTERVAL_ATTENTION', False)
        self.voxel_dom_roadinterval_attention = self.model_cfg.get('VOXEL_DOM_ROADINTERVAL_ATTENTION', False)
        self.joint_attention = self.model_cfg.get('VOXEL_DETDOM_JOINT_ATTENTION', False)
        self.dom_patch_first = self.model_cfg.get('DOM_PATCH_FIRST', False)

        if self.range_guidance:
            if self.range_guidance_dom_only:
                input_channels_dom = input_channels + 2
            else:
                input_channels = input_channels + 2
                input_channels_dom = input_channels
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
        self.center_xy = self.model_cfg.get('CENTER_XY', False)
        self.zeroone_prior = self.model_cfg.get('ZEROONE_PRIOR', False)

        self.rm_thresh = self.model_cfg.get('RM_THRESH', 0)

        if self.rangeinv:
            self.conv_range = nn.Conv2d(
                input_channels, 1,
                kernel_size=1
            )
            #nn.Sequential(

        if self.voxel_det_seconv_attention and not self.joint_attention:
            self.att_spatial_se_layer_det = SpatialSELayer(512)

        if self.voxel_det_se_attention and not self.joint_attention:
            self.att_se_layer_det = ChannelSELayer(512)

        if self.voxel_det_patch_attention and not self.joint_attention:
            self.att_patch_layer_det = LocationAttentionLayer(512, self.model_cfg.PATCH_SIZE, prior=self.zeroone_prior)


        ###################
        if self.voxel_dom_seconv_attention:
            self.att_spatial_se_layer = SpatialSELayer(512)

        if self.voxel_dom_se_attention:
            self.att_se_layer = ChannelSELayer(512)

        if self.voxel_dom_patch_attention:
            self.att_patch_layer = LocationAttentionLayer(512, self.model_cfg.PATCH_SIZE, prior=self.zeroone_prior)

        if self.voxel_dom_rangeinterval_attention:
            self.att_rangeinterval_layer = RangeIntervalAttentionLayer(512, self.model_cfg.PATCH_SIZE, division=self.model_cfg.get('RANGE_INTERVAL_DIVISION', 6), prior=self.zeroone_prior)

        if self.voxel_dom_roadinterval_attention:
            self.att_roadinterval_layer = RoadIntervalAttentionLayer(512, self.model_cfg.PATCH_SIZE, division=self.model_cfg.get('ROAD_INTERVAL_DIVISION', 6), prior=self.zeroone_prior)

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None

        dom_fc1, dom_fc2 = self.model_cfg.get('DOM_FC', [1024, 1024])
        # print("dom_fc ", dom_fc1, dom_fc2)
        # if self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None) is not None:

        if self.range_da > 0:
            self.domain_pool = nn.AdaptiveAvgPool2d(1)
            self.domain_classifier_range = nn.ModuleDict()
            for n in range(0+self.remove_near_range, self.range_da-self.remove_far_range):
                self.domain_classifier_range[str(n)] = nn.Sequential(nn.Linear(input_channels, dom_fc1),
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(dom_fc1, dom_fc2), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(dom_fc2, 1))
            if self.keep_xy:
                self.domain_classifier_range2 = nn.ModuleDict()
                for n in range(0+self.remove_near_range2, self.range_da-self.remove_far_range2):
                    self.domain_classifier_range2[str(n)] = nn.Sequential(nn.Linear(input_channels, dom_fc1),
                                                        nn.ReLU(True), nn.Dropout(),
                                                        nn.Linear(dom_fc1, dom_fc2), nn.ReLU(True),
                                                        nn.Dropout(), nn.Linear(dom_fc2, 1))

        elif self.interval_da > 0:
            self.domain_pool = nn.AdaptiveAvgPool2d(1)
            self.domain_classifier_interval = nn.ModuleDict()
            for n in range(self.interval_da):
                self.domain_classifier_interval[str(n)] = nn.Sequential(nn.Linear(input_channels, dom_fc1),
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(dom_fc1, dom_fc2), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(dom_fc2, 1))

        elif self.range_guidance_conv_dom:
            self.conv_dom_layers = self.make_conv_layers(
                conv_cfg=self.model_cfg.LOCAL_DOM_FC,
                input_channels=input_channels_dom,
                output_channels=1
            )
            if self.range_guidance_double_dom:
                self.domain_pool = nn.AdaptiveAvgPool2d(1)
                self.domain_classifier = nn.Sequential(nn.Linear(input_channels_dom, dom_fc1),
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(dom_fc1, dom_fc2), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(dom_fc2, 1))
        elif self.range_guidance_new_conv_dom:
            print("input_channels_dom", input_channels_dom)
            self.conv_dom_layers = LocalDomainClassifier(input_channels=input_channels_dom, context=self.range_guidance_new_conv_dom_context)  #

        # elif self.range_guidance_pixelfc_dom:
        #     # for i in range()

        #     self.pixelfc_layers = nn.ModuleList()

        #     # for i in range(self.model_cfg.PATCH_SIZE):
        #     self.make_fc_layers(
        #         conv_cfg=self.model_cfg.LOCAL_DOM_FC,
        #         input_channels=input_channels_dom,
        #         output_channels=1
        #     )

        else:
            self.domain_pool = nn.AdaptiveAvgPool2d(1)
            self.domain_classifier = nn.Sequential(nn.Linear(input_channels_dom, dom_fc1),
                                                nn.ReLU(True), nn.Dropout(),
                                                nn.Linear(dom_fc1, dom_fc2), nn.ReLU(True),
                                                nn.Dropout(), nn.Linear(dom_fc2, 1))

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def local_attention(self, features, d):
        # features.size() =  [1, 256, h, w]
        # d.size() = [1, 1, h, w]  after sigmoid

        d = d.clamp(1e-6, 1)
        H = - ( d * d.log() + (1-d) * (1-d).log() )
        w = 1 - H
        features_new = (1 + w) * features

        return  features_new

    def forward(self, data_dict):
        t_mode = data_dict['t_mode']
        l = data_dict['l']

        if 'pseudo' in t_mode:
            pseudo = True
        else:
            pseudo = False

        spatial_features_2d = data_dict['spatial_features_2d']

        if t_mode == 'tsne':
            self.range_da = 2

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

            return data_dict

        ###########################
        if self.range_guidance and not self.range_guidance_dom_only:
            total_range = spatial_features_2d.shape[-1]
            half_range = int(spatial_features_2d.shape[-1] * 0.5)

            x_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1, total_range, 1).cuda()
            # print("x_range", x_range)
            y_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1,1,total_range).cuda()

            spatial_features_2d = torch.cat((spatial_features_2d, x_range, y_range), dim=1)
            # print("spatial_features_2d", spatial_features_2d.shape)


        if 'dom_img' in t_mode:

            if t_mode == 'dom_img_src':
                dom_src = True
            elif t_mode == 'dom_img_tgt':
                dom_src = False
            else:
                dom_src = None

            #################### PATCH EARLY
            if self.voxel_dom_patch_attention and self.dom_patch_first:
                spatial_features_2d = self.att_patch_layer(spatial_features_2d)

            if self.voxel_dom_rangeinterval_attention and self.dom_patch_first:
                spatial_features_2d = self.att_rangeinterval_layer(spatial_features_2d)

            if self.voxel_dom_roadinterval_attention and self.dom_patch_first:
                spatial_features_2d = self.att_roadinterval_layer(spatial_features_2d)

            #################### PATCH LATE
            if self.voxel_dom_patch_attention and not self.dom_patch_first:
                spatial_features_2d = self.att_patch_layer(spatial_features_2d)

            if self.voxel_dom_rangeinterval_attention and not self.dom_patch_first:
                spatial_features_2d = self.att_rangeinterval_layer(spatial_features_2d)

            if self.voxel_dom_roadinterval_attention and not self.dom_patch_first:
                spatial_features_2d = self.att_roadinterval_layer(spatial_features_2d)

            ####################

            if self.range_guidance and self.range_guidance_dom_only:
                total_range = spatial_features_2d.shape[-1]
                half_range = int(spatial_features_2d.shape[-1] * 0.5)
                x_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1, total_range, 1).cuda()
                y_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1,1,total_range).cuda()
                spatial_features_2d = torch.cat((spatial_features_2d, x_range, y_range), dim=1)


            if self.range_guidance_conv_dom or self.range_guidance_new_conv_dom:
                # x_pool = self.domain_pool().view(spatial_features_2d.size(0), -1)
                # print('t_mode', t_mode)
                # print("l", l)
                if self.range_guidance_new_conv_dom_attention:
                    x_reverse = grad_reverse(spatial_features_2d, l*-1)
                    if self.range_guidance_new_conv_dom_context:
                        dom_img_preds, _ = self.conv_dom_layers(x_reverse)
                        #print(d_pixel)
                        # if not target:
                        _, feat_pixel = self.conv_dom_layers(spatial_features_2d.detach())
                    else:
                        dom_img_preds = self.conv_dom_layers(x_reverse)
                    spatial_features_2d = self.local_attention(spatial_features_2d, dom_img_preds.detach())

                else:
                    x_reverse = grad_reverse(spatial_features_2d, l*-1)
                    dom_img_preds = self.conv_dom_layers(x_reverse)

                if self.range_guidance_double_dom:
                    x_pool2 = self.domain_pool(spatial_features_2d).view(spatial_features_2d.size(0), -1)
                    x_reverse2 = grad_reverse(x_pool2, l*-1)
                    # print("x_reverse2", x_reverse2.shape)
                    dom_img_preds2 = self.domain_classifier(x_reverse2)#.squeeze(-1)

            else:
                x_pool = self.domain_pool(spatial_features_2d).view(spatial_features_2d.size(0), -1)
                x_reverse = grad_reverse(x_pool, l*-1)
                dom_img_preds = self.domain_classifier(x_reverse)#.squeeze(-1)

            # print("dom_img_preds", dom_img_preds.shape)
            if self.dom_squeeze:
                dom_img_preds = dom_img_preds.squeeze(-1)
                if self.range_guidance_double_dom:
                    dom_img_preds2 = dom_img_preds2.squeeze(-1)

            self.forward_ret_dict['dom_img_preds'] = dom_img_preds
            if self.range_guidance_double_dom:
                self.forward_ret_dict['dom_img_preds2'] = dom_img_preds2

            if self.training:
                targets_dict_dom = self.assign_targets(
                        gt_boxes=data_dict['gt_boxes'],
                        dom_src=dom_src,
                        pseudo=pseudo
                )
                # if self.range_guidance_conv_dom:
                #     targets_dict_dom['dom_labels']
                self.forward_ret_dict.update(targets_dict_dom)

            if 'det' not in t_mode:
                return data_dict


        if self.joint_attention:
            if self.voxel_det_seconv_attention and self.voxel_det_se_attention:
                spatial_features_2d = torch.max(self.att_spatial_se_layer(spatial_features_2d), self.att_se_layer(spatial_features_2d))
                # spatial_features_2d_det = spatial_features_2d
            elif self.voxel_det_seconv_attention:
                # print("spatial_features_2d before", spatial_features_2d.shape)
                spatial_features_2d = self.att_spatial_se_layer(spatial_features_2d)
                # spatial_features_2d_det = spatial_features_2d
            elif self.voxel_det_se_attention:
                spatial_features_2d = self.att_se_layer(spatial_features_2d)
                # spatial_features_2d_det = spatial_features_2d

            # else:
            spatial_features_2d_det = spatial_features_2d
        else:
            if self.voxel_det_seconv_attention and self.voxel_det_se_attention:
                spatial_features_2d_out = torch.max(self.att_spatial_se_layer_det(spatial_features_2d), self.att_se_layer_det(spatial_features_2d))
                spatial_features_2d_det = spatial_features_2d_out
            elif self.voxel_det_seconv_attention:
                # print("spatial_features_2d before", spatial_features_2d.shape)
                spatial_features_2d_det = self.att_spatial_se_layer_det(spatial_features_2d)
            elif self.voxel_det_se_attention:
                spatial_features_2d_det = self.att_se_layer_det(spatial_features_2d)
            else:
                spatial_features_2d_det = spatial_features_2d
                # print("spatial_features_2d", spatial_features_2d.shape)

        cls_preds = self.conv_cls(spatial_features_2d_det)
        box_preds = self.conv_box(spatial_features_2d_det)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d_det)
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