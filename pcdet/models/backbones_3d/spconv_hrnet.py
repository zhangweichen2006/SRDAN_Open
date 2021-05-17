from functools import partial

import spconv
import torch
import torch.nn as nn

from ...utils import common_utils
from .spconv_backbone import post_act_block


class CatTensors(spconv.SparseModule):
    def forward(self, x: spconv.SparseConvTensor, x2: spconv.SparseConvTensor):
        # print("x", x.dense().shape)
        # print("x2", x2.dense().shape)

        # print(torch.cat((x.features, x2.features)).shape)
        # print("n", n.dense())

        inds = x.indices
        inds2 = x2.indices

        inds_joint = torch.cat((inds, inds2))

        spatial_shape = [x.batch_size, *x.spatial_shape]
        spatial_stride = [0] * len(spatial_shape)

        spatial_shape2 = [x2.batch_size, *x2.spatial_shape]
        spatial_stride2 = [0] * len(spatial_shape2)

        val = 1
        for i in range(inds.shape[1] - 1, -1, -1):
            spatial_stride[i] = val
            val *= spatial_shape[i]
        
        val2 = 1
        for i in range(inds2.shape[1] - 1, -1, -1):
            spatial_stride2[i] = val2
            val2 *= spatial_shape2[i]

        indices_index = inds[:, -1]
        indices_index2 = inds2[:, -1]

        for i in range(len(spatial_shape) - 1):
            indices_index += spatial_stride[i] * inds[:, i]
        
        for i in range(len(spatial_shape2) - 1):
            indices_index2 += spatial_stride2[i] * inds2[:, i]

        indices_index_joint = torch.cat((indices_index, indices_index2)) # 405314, 60000

        _, unique_inds = torch.unique(indices_index_joint, return_inverse=True) # 465314

        features_cat = torch.cat((x.features, x2.features))

        new_features = features_cat[unique_inds]
        new_inds = inds_joint[unique_inds]

        res = spconv.SparseConvTensor(new_features, new_inds, x.spatial_shape,
                                      x.batch_size, x.grid)
        
        # print("x", x.dense().shape)
        # print("x2", x2.dense().shape)
        # print("res", res.dense().shape)

        # res.indice_dict = {**x.indice_dict, **x2.indice_dict}

        return res

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out


class HRNet(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, num_fpn_up=0, num_fpn_downup=0, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.num_fpn_up = num_fpn_up
        self.num_fpn_downup = num_fpn_downup

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        self.cat_tensors = CatTensors()

        # self.conv1 = spconv.SparseSequential(
        #     block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        # )

        # self.conv2 = spconv.SparseSequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        # )


        # spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
        #                             bias=False, indice_key='spconv_down2'),

        # spconv.SparseSequential(
        #         spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
        #         norm_fn(16),
        #         nn.ReLU(),
        #     )

        self.share_up_module = self.model_cfg.get('SHARE_UP_MODULE', False)

        # self.hrnet_extra = self.model_cfg.get('HRNET_EXTRA', {})

        # layer 1
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.conv1_transition_1_1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv1_trans1', conv_type='spconv'),
        )
        self.conv1_transition_1_2 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv1_down1', conv_type='spconv'),
        )

        # layer 2 (stage)
        self.conv2_1 = spconv.SparseSequential(
            # block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='spconv2_1'),
            block(16, 8, 3, norm_fn=norm_fn, padding=1, indice_key='spconv2_1'),
        )
        self.conv2_2 = spconv.SparseSequential(
            # block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='spconv2_2'),
            block(32, 16, 3, norm_fn=norm_fn, padding=1, indice_key='spconv2_2'),
        )
        self.conv2_transition_down_1 = spconv.SparseSequential(
            block(8, 16, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2_down1', conv_type='spconv', no_relu=True),
        )
        self.conv2_transition_down_2 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2_down2', conv_type='spconv', no_relu=True),
        )
        self.conv2_transition_down_12 = spconv.SparseSequential(
            block(8, 16, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2_down12_1', conv_type='spconv', no_relu=True),
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2_down12_2', conv_type='spconv', no_relu=True),
        )
        self.conv2_transition_up_1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, indice_key='spconv1_down1', conv_type='inverseconv', no_relu=True)
        ) # can use conv and upsample 
        
        # layer 3 (stage)
        self.conv3_1 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            # block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='spconv3_1', conv_type='spconv'),
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='spconv3_1'),
        )
        self.conv3_2 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            # block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='spconv3_2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='spconv3_2'),
        )
        self.conv3_3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            # block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='spconv3_3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='spconv3_3'),
        )
        self.conv3_transition_down_1 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down1', conv_type='spconv', no_relu=True),
        )
        self.conv3_transition_down_2 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down2', conv_type='spconv', no_relu=True),
        )
        self.conv3_transition_down_3 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv3_down3', conv_type='spconv', no_relu=True),
        )
        self.conv3_transition_down_12 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down12_1', conv_type='spconv', no_relu=True),
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down12_2', conv_type='spconv', no_relu=True),
        )
        self.conv3_transition_down_23 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down23_2', conv_type='spconv', no_relu=True),
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv3_down23_3', conv_type='spconv', no_relu=True),
        )
        self.conv3_transition_down_123 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down123_1', conv_type='spconv', no_relu=True),
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down123_2', conv_type='spconv', no_relu=True),
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv3_down123_3', conv_type='spconv', no_relu=True),
        )
        self.conv3_transition_up_1 = spconv.SparseSequential(
            block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2_down1', conv_type='inverseconv', no_relu=True)
        )
        self.conv3_transition_up_2 = spconv.SparseSequential(
            block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv2_down2', conv_type='inverseconv', no_relu=True)
        )  # can use conv and upsample 
        self.conv3_transition_up_12 = spconv.SparseSequential(
            block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv2_down12_2', conv_type='inverseconv', no_relu=True),
            block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2_down12_1', conv_type='inverseconv', no_relu=True)
        )

        # layer 4 (stage)
        self.conv4_1 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            # block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='spconv4_1', conv_type='spconv'),
            block(16, 8, 3, norm_fn=norm_fn, padding=1, indice_key='spconv4_1'),
        )
        self.conv4_2 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            # block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='spconv4_2', conv_type='spconv'),
            block(32, 16, 3, norm_fn=norm_fn, padding=1, indice_key='spconv4_2'),
        )
        self.conv4_3 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            # block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='spconv4_3', conv_type='spconv'),
            block(64, 32, 3, norm_fn=norm_fn, padding=1, indice_key='spconv4_3'),
        )
        self.conv4_4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            # block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='spconv4_4', conv_type='spconv'),
            block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='spconv4_4'),
        )
        
        self.conv4_transition_down_123 = spconv.SparseSequential(
            block(8, 16, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down123_1', conv_type='spconv', no_relu=True),
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down123_2', conv_type='spconv', no_relu=True),
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv3_down123_3', conv_type='spconv', no_relu=True),
        )
        self.conv4_transition_down_23 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down23_2', conv_type='spconv', no_relu=True),
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv3_down23_3', conv_type='spconv', no_relu=True),
        )
        self.conv4_transition_down_3 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv3_down3', conv_type='spconv', no_relu=True),
        )

        
        
        # self.conv3_transition_down_1 = spconv.SparseSequential(
        #     block(8, 16, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down1', conv_type='spconv', no_relu=True),
        # )
        # self.conv3_transition_down_2 = spconv.SparseSequential(
        #     block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down2', conv_type='spconv', no_relu=True),
        # )
        # self.conv3_transition_down_3 = spconv.SparseSequential(
        #     block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv3_down3', conv_type='spconv', no_relu=True),
        # )
        # self.conv3_transition_down_12 = spconv.SparseSequential(
        #     block(8, 16, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down12_1', conv_type='spconv', no_relu=True),
        #     block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down12_2', conv_type='spconv', no_relu=True),
        # )
        # self.conv3_transition_down_23 = spconv.SparseSequential(
        #     block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down23_2', conv_type='spconv', no_relu=True),
        #     block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_down23_3', conv_type='spconv', no_relu=True),
        # )
        # self.conv3_transition_down_123 = spconv.SparseSequential(
        #     block(8, 16, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv3_down123_1', conv_type='spconv', no_relu=True),
        #     block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv3_down123_2', conv_type='spconv', no_relu=True),
        #     block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv3_down123_3', conv_type='spconv', no_relu=True),
        # )
        # self.conv3_transition_up_1 = spconv.SparseSequential(
        #     block(16, 8, 3, norm_fn=norm_fn, indice_key='spconv2_down1', conv_type='inverseconv', no_relu=True)
        # )
        # self.conv3_transition_up_2 = spconv.SparseSequential(
        #     block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2_down2', conv_type='inverseconv', no_relu=True)
        # )  # can use conv and upsample 
        # self.conv3_transition_up_12 = spconv.SparseSequential(
        #     block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2_down12_2', conv_type='inverseconv', no_relu=True),
        #     block(16, 8, 3, norm_fn=norm_fn, indice_key='spconv2_down12_1', conv_type='inverseconv', no_relu=True)
        # )
        
        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
            last_pad = self.model_cfg.get('last_pad', 0)

            self.conv_out = spconv.SparseSequential(
                # conv 4: 1, 64, 5, 126, 126 to 128, 2, 126, 126
                spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                # 1, 128, 2, 126, 126 -> 256, 126, 126
                norm_fn(128),
                nn.ReLU(),
            )
        else:
            self.conv_out = None

        if self.num_fpn_up + self.num_fpn_downup > 0:
            self.FPN = True
            last_pad = self.model_cfg.get('last_pad', 0)
            ######### can change different receptive field #########
            ######### can change different receptive field #########
            ######### can change different receptive field #########
            # fpn conv: GPU not possible to have too many fpn conv
            if self.num_fpn_up > 0:
                self.conv3_ident = spconv.SparseSequential(
                    spconv.SparseConv3d(64, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_ident_3'),
                    # 128, 5, 252, 252 -> 640, 252, 252
                    norm_fn(64),
                    nn.ReLU(),
                )
                if not self.share_up_module:
                    self.conv_up_t4_to_3 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
                    self.conv_up_m4_to_3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
                    self.conv4_to_3 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

                # conv3: 1, 64, 11, 252, 252 to 128, 5, 252, 252
                self.conv3_out = spconv.SparseSequential(
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2_3'),
                    # 128, 5, 252, 252 -> 640, 252, 252
                    norm_fn(128),
                    nn.ReLU(),
                )
            if self.num_fpn_up > 1:
                self.conv2_ident = spconv.SparseSequential(
                    spconv.SparseConv3d(32, 32, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_ident_3'),
                    # 128, 5, 252, 252 -> 640, 252, 252
                    norm_fn(32),
                    nn.ReLU(),
                )
                # conv2: 1, 32, 21, 504, 504 to 64, 10, 504, 504
                self.conv2_out = spconv.SparseSequential(
                    spconv.SparseConv3d(32, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2_2'),
                    # 64, 10, 504, 504 -> 640, 504, 504
                    norm_fn(64),
                    nn.ReLU(),
                )

            # fpn deconv
            if self.num_fpn_downup > 0:
                # conv2: 64, 10, 504, 504 to 1, 32, 21, 504, 504 to 
                self.conv5 = spconv.SparseSequential(
                    # [800, 704, 21] <- [400, 352, 11]
                    block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv5', conv_type='spconv'),
                    block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
                    block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
                )

                self.conv5_out = spconv.SparseSequential(
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2_5'),
                    # 64, 10, 504, 504 -> 640, 504, 504
                    norm_fn(128),
                    nn.ReLU(),
                )

                # if not self.share_up_module:
                self.conv_up_t5_to_4 = SparseBasicBlock(64, 64, indice_key='subm5', norm_fn=norm_fn)
                self.conv_up_m5_to_4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm5')
                self.conv5_to_4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv5', conv_type='inverseconv')

                self.conv4_ident = spconv.SparseSequential(
                    spconv.SparseConv3d(64, 64, (1, 1, 1), stride=(1, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_ident_4'),
                    # 128, 11, 126, 126 
                    norm_fn(64),
                    nn.ReLU(),
                )

                self.conv4_out = spconv.SparseSequential(
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2_4'),
                    # 64, 10, 504, 504 -> 640, 504, 504
                    norm_fn(128),
                    nn.ReLU(),
                )
            # if self.num_fpn_downup > 1:
        else:
            self.FPN = False

        # decoder
        # if self.num_fpn_downup > 0:
        #     self.conv_up_t5 = SparseBasicBlock(64, 64, indice_key='subm5', norm_fn=norm_fn)
        #     self.conv_up_m5 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm5')
        #     self.inv_conv5 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv5', conv_type='inverseconv')

    #     # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2_2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2_2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv1_trans_2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv0 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x.features = x_m.features + x.features
        x = conv_inv(x)
        return x

    def FPN_block_forward(self, x_lateral, x_bottom, conv_ident, conv_inv, conv_t=None, conv_m=None): 
        #conv_t, conv_m,
        x_ident = conv_ident(x_lateral)
        # x = x_ident
        # x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
        if conv_t is not None:
            x_bottom = conv_t(x_bottom)
        if conv_m is not None:
            x_bottom = conv_m(x_bottom)
        x = conv_inv(x_bottom)
        # x = self.channel_reduction(x, x_m.features.shape[1])
        # print("channel_reduction", x.dense().shape)
        # x.features = x_m.features + x.features
        # print("x.features", x.dense().shape)
        # x = conv_inv(x)
        # print("xfi", x.dense().shape)
        x.features = x.features + x_ident.features
        # print("xfi", x.dense().shape)
        return x

    # def hrnet_block():

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x.features = features.view(n, out_channels, -1).sum(dim=2)
        return x
    
    def hr_fusion(self, hr_x, hr_modules, hr_layer_in, hr_layer_out):
        hr_out = []
        hr_next_layer_in = []

        for i in range(hr_layer_in): # 0,1: 2-1
            for j in range(hr_layer_out):# 0,1,2: 2-2
                # print("ij",i,j)
                idx_out = i*hr_layer_out + j
                if i == j:
                    x_out = hr_x[i]
                    # print("x_out", x_out.dense().shape)
                # else:
                elif i < j:
                    # print("hr_modules[idx_out][0]", hr_modules[idx_out][0])
                    x_out = hr_modules[idx_out][0](hr_x[i])
                    for k in range(1, len(hr_modules[idx_out])):
                        x_out = hr_modules[idx_out][k](x_out)
                    # print("x_out", x_out.dense().shape)
                else:
                    x_out = None

                hr_out.append(x_out)
                # hr_x
        # print("hr_out", hr_out)

        for j in range(hr_layer_out): # 0123
            for i in range(hr_layer_in): # 012
                idx_hr_next_layer_in = i*hr_layer_out + j
                # print('fuse ij', i,j)
                # print("hr_next_layer_in", hr_next_layer_in)
                # print("hr_out", hr_out)
                if i == 0:
                    if hr_layer_in == 1:
                        hr_next_layer_in.append(spconv.SparseSequential(nn.ReLU())(hr_out[idx_hr_next_layer_in]))
                    else:
                        hr_next_layer_in.append(hr_out[idx_hr_next_layer_in])
                # else:
                #     if hr_out[idx_hr_next_layer_in] is not None: 
                #         # print("hr_next_layer_in[j]", hr_next_layer_in[j].dense().shape)
                #         # print("hr_out[idx_hr_next_layer_in]", hr_out[idx_hr_next_layer_in].dense().shape)
                #         # print("hr_next_layer_in[j].features", hr_next_layer_in[j].features.shape)
                #         # print("hr_out[idx_hr_next_layer_in].features", hr_out[idx_hr_next_layer_in].features.shape)
                #         hr_next_layer_in[j] = self.cat_tensors(hr_next_layer_in[j], hr_out[idx_hr_next_layer_in])
                #         # print("new_tensor", hr_next_layer_in[j].dense().shape)
                #     # size mismatch
                #     # hr_next_layer_in[j].features = torch.cat((hr_next_layer_in[j].features, hr_out[idx_hr_next_layer_in].features), dim=1)
                #     if i == hr_layer_in - 1:
                #         hr_next_layer_in[j] = spconv.SparseSequential(nn.ReLU())(hr_next_layer_in[j])
                    # print("hr_next_layer_in[j]", hr_next_layer_in[j].dense)
        #     temp = []
        #     for i in range(hr_layer_in):
        #         # hr_next_in[j] = 
        #             x_out.features = x_out.features + .features
        #             x_out = spconv.SparseSequential(nn.ReLU())(x_out)
        #     hr_next_in.append(x_out)
            # print(f"hr_next_layer_in{j}", hr_next_layer_in[j].dense().shape)
        return hr_next_layer_in

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        # layer 1
        x_conv1 = self.conv1(x)
        x_conv1_1 = self.conv1_transition_1_1(x)
        x_conv1_2 = self.conv1_transition_1_2(x)
        # 1, 32, 21, 504, 504

        # layer 2
        x_conv2_1 = self.conv2_1(x_conv1_1)
        x_conv2_2 = self.conv2_2(x_conv1_2)

        x_trans_conv2 = [x_conv2_1, x_conv2_2] #[x_conv2_1, x_conv2_2]
        hr_trans_conv2 = [[], [self.conv2_transition_down_1], \
                          [self.conv2_transition_down_12],\
                          [self.conv2_transition_up_1],\
                          [], [self.conv2_transition_down_2]]

        x_conv2_1, x_conv2_2, x_conv2_3 = self.hr_fusion(x_trans_conv2, hr_trans_conv2, hr_layer_in=2, hr_layer_out=3)
        # print("x_conv2_outputs", x_conv2_1.dense().shape) # 8, 41, 1008, 1008
        # print("x_conv2_outputs2", x_conv2_2.dense().shape) # 16, 21, 504, 504
        # print("x_conv2_outputs3", x_conv2_3.dense().shape) # 32, 11, 252, 252

        # layer 3
        x_conv3_1 = self.conv3_1(x_conv2_1)
        x_conv3_2 = self.conv3_2(x_conv2_2)
        x_conv3_3 = self.conv3_3(x_conv2_3)

        x_trans_conv3 = [x_conv3_1, x_conv3_2, x_conv3_3]
        hr_trans_conv3 = [[], [self.conv3_transition_down_1], \
                          [self.conv3_transition_down_12],\
                          [self.conv3_transition_down_123],\
                          [self.conv3_transition_up_1], [], \
                          [self.conv3_transition_down_2], [self.conv3_transition_down_23], \
                          [self.conv3_transition_up_12], [self.conv3_transition_up_2], \
                          [], [self.conv3_transition_down_3]]

        x_conv3_1, x_conv3_2, x_conv3_3, x_conv3_4 = self.hr_fusion(x_trans_conv3, hr_trans_conv3, hr_layer_in=3, hr_layer_out=4)

        # print("x_conv3_1", x_conv3_1.dense().shape)
        # print("x_conv3_2", x_conv3_2.dense().shape)
        # print("x_conv3_3", x_conv3_3.dense().shape)
        # print("x_conv3_4", x_conv3_4.dense().shape)

        x_conv4_1 = self.conv4_1(x_conv3_1)
        x_conv4_2 = self.conv4_2(x_conv3_2)
        x_conv4_3 = self.conv4_3(x_conv3_3)
        x_conv4_4 = self.conv4_4(x_conv3_4)

        # print("x_conv4_1", x_conv4_1.dense().shape)
        # print("x_conv4_2", x_conv4_2.dense().shape)
        # print("x_conv4_3", x_conv4_3.dense().shape)
        # print("x_conv4_4", x_conv4_4.dense().shape)

        x_conv4_1_out = self.conv4_transition_down_123(x_conv4_1)
        x_conv4_2_out = self.conv4_transition_down_23(x_conv4_2)
        x_conv4_3_out = self.conv4_transition_down_3(x_conv4_3)
        x_conv4_4_out = x_conv4_4

        # print("x_conv4_1_out", x_conv4_1_out.dense().shape)
        # print("x_conv4_2_out", x_conv4_2_out.dense().shape)
        # print("x_conv4_3_out", x_conv4_3_out.dense().shape)
        # print("x_conv4_4_out", x_conv4_4_out.dense().shape)

        x_final_out = x_conv4_4_out
        # self.cat_tensors(self.cat_tensors(self.cat_tensors(x_conv4_1_out, x_conv4_2_out), x_conv4_3_out), x_conv4_4_out)
        
        # print('x_final_out', x_final_out.dense().shape)
        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_final_out)
            # print('out', out.dense().shape)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        #     if self.FPN:
        #         if self.num_fpn_up > 0:
        #             if self.share_up_module:
        #                 x_fpn3 = self.FPN_block_forward(x_conv3, x_conv4, self.conv3_ident, self.inv_conv4, conv_t=self.conv_up_t4, conv_m=self.conv_up_m4)
        #             else:
        #                 x_fpn3 = self.FPN_block_forward(x_conv3, x_conv4, self.conv3_ident, self.conv4_to_3,
        #                 conv_t=self.conv_up_t4_to_3, conv_m=self.conv_up_m4_to_3)
        #             out_3 = self.conv3_out(x_fpn3)
        #             batch_dict['encoded_spconv_tensor_fpn3'] = out_3
        #             batch_dict['encoded_spconv_tensor_stride_fpn3'] = 8#16
                
        #         # if self.num_fpn_up > 1:
        #         #     out_2 = self.conv2_out(x_conv2)
        #         #     batch_dict['encoded_spconv_tensor_fpn2'] = out_2
        #         #     batch_dict['encoded_spconv_tensor_stride_fpn2'] = 32

        #         if self.num_fpn_downup > 0:
        #             # print("x_conv4", x_conv4.dense().shape)
        #             # 64, 5, 126, 126
        #             x_conv5 = self.conv5(x_conv4)
        #             # print("conv5", x_conv5.dense().shape)
        #             # 64, 3, 63, 63
        #             out_5 = self.conv5_out(x_conv5)
        #             # print("out_5", out_5.dense().shape)
        #             # 128, 1, 63, 63
        #             batch_dict['encoded_spconv_tensor_fpn5'] = out_5
        #             batch_dict['encoded_spconv_tensor_stride_fpn5'] = 8#16

        #             x_fpn4 = self.FPN_block_forward(x_conv4, x_conv5, self.conv4_ident, self.conv5_to_4,
        #             conv_t=self.conv_up_t5_to_4, conv_m=self.conv_up_m5_to_4)
        #             # print("x_fpn4", x_fpn4.dense().shape)
        #             # 64, 5, 126, 126
        #             out_4 = self.conv4_out(x_fpn4)
        #             # print("out_4", out_4.dense().shape)
        #             # 128, 2, 126, 126

        #             batch_dict['encoded_spconv_tensor_fpn4'] = out_4
        #             batch_dict['encoded_spconv_tensor_stride_fpn4'] = 8#16
                
        #     # print("cv out", out.dense().shape)
        #     # 1, 128, 2, 126, 126 -> 256, 126, 126

        # # for segmentation head
        # # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # # print("x_up4 out", x_up4.dense().shape)
        # # 1, 64, 11, 252, 252

        # # [800, 704, 21] <- [400, 352, 11]
        # x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # # print("x_up3 out", x_up3.dense().shape)
        # # 1, 32, 21, 504, 504

        # # [1600, 1408, 41] <- [800, 704, 21]
        # x_up2 = self.UR_block_forward(x_conv2_2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # # print("x_up2 out", x_up2.dense().shape)
        # # 1, 16, 41, 1008, 1008

        # # [1600, 1408, 41] <- [1600, 1408, 41]
        # x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv0)
        # # print("x_up1 out", x_up1.dense().shape) 
        # # 16, 41, 1008, 1008

        # batch_dict['point_features'] = x_up1.features
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)



        return batch_dict

