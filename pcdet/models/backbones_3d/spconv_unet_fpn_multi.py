from functools import partial

import spconv
import torch
import torch.nn as nn

from ...utils import common_utils
from .spconv_backbone import post_act_block


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

class UNetV2FPNMulti(nn.Module):
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

        self.downup_up = self.model_cfg.get('DOWNUP_UP_MODULE', False)

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.share_up_module = self.model_cfg.get('SHARE_UP_MODULE', False)

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

        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

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

        # print("x", x.dense().shape)
        # , 16, 41, 1008, 1008

        x_conv1 = self.conv1(x)
        # print("x_conv1", x_conv1.dense().shape)
        # 1, 16, 41, 1008, 1008

        x_conv2 = self.conv2(x_conv1)
        # print("x_conv2", x_conv2.dense().shape)
        # 1, 32, 21, 504, 504

        x_conv3 = self.conv3(x_conv2)
        # print("x_conv3", x_conv3.dense().shape)
        # 1, 64, 11, 252, 252

        x_conv4 = self.conv4(x_conv3)
        # print("x_conv4", x_conv4.dense().shape)
        # 1, 64, 5, 126, 126

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8
            if self.FPN:
                if self.num_fpn_up > 0 and not self.downup_up:
                    if self.share_up_module:
                        x_fpn3 = self.FPN_block_forward(x_conv3, x_conv4, self.conv3_ident, self.inv_conv4, conv_t=self.conv_up_t4, conv_m=self.conv_up_m4)
                    else:
                        x_fpn3 = self.FPN_block_forward(x_conv3, x_conv4, self.conv3_ident, self.conv4_to_3,
                        conv_t=self.conv_up_t4_to_3, conv_m=self.conv_up_m4_to_3)
                    out_3 = self.conv3_out(x_fpn3)
                    batch_dict['encoded_spconv_tensor_fpn3'] = out_3
                    batch_dict['encoded_spconv_tensor_stride_fpn3'] = 8#16
                
                # if self.num_fpn_up > 1:
                #     out_2 = self.conv2_out(x_conv2)
                #     batch_dict['encoded_spconv_tensor_fpn2'] = out_2
                #     batch_dict['encoded_spconv_tensor_stride_fpn2'] = 32

                if self.num_fpn_downup > 0:
                    # print("x_conv4", x_conv4.dense().shape)
                    # 64, 5, 126, 126
                    x_conv5 = self.conv5(x_conv4)
                    # print("conv5", x_conv5.dense().shape)
                    # 64, 3, 63, 63
                    out_5 = self.conv5_out(x_conv5)
                    # print("out_5", out_5.dense().shape)
                    # 128, 1, 63, 63
                    batch_dict['encoded_spconv_tensor_fpn5'] = out_5
                    batch_dict['encoded_spconv_tensor_stride_fpn5'] = 8#16

                    x_fpn4 = self.FPN_block_forward(x_conv4, x_conv5, self.conv4_ident, self.conv5_to_4,
                    conv_t=self.conv_up_t5_to_4, conv_m=self.conv_up_m5_to_4)
                    # print("x_fpn4", x_fpn4.dense().shape)
                    # 64, 5, 126, 126
                    out_4 = self.conv4_out(x_fpn4)
                    # print("out_4", out_4.dense().shape)
                    # 128, 2, 126, 126

                    batch_dict['encoded_spconv_tensor_fpn4'] = out_4
                    batch_dict['encoded_spconv_tensor_stride_fpn4'] = 8#16
                
                if self.num_fpn_up > 0 and self.downup_up:
                    if self.share_up_module:
                        x_fpn3 = self.FPN_block_forward(x_conv3, x_fpn4, self.conv3_ident, self.inv_conv4, conv_t=self.conv_up_t4, conv_m=self.conv_up_m4)
                    else:
                        x_fpn3 = self.FPN_block_forward(x_conv3, x_fpn4, self.conv3_ident, self.conv4_to_3,
                        conv_t=self.conv_up_t4_to_3, conv_m=self.conv_up_m4_to_3)
                    out_3 = self.conv3_out(x_fpn3)
                    batch_dict['encoded_spconv_tensor_fpn3'] = out_3
                    batch_dict['encoded_spconv_tensor_stride_fpn3'] = 8#16
                

            # print("cv out", out.dense().shape)
            # 1, 128, 2, 126, 126 -> 256, 126, 126

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # print("x_up4 out", x_up4.dense().shape)
        # 1, 64, 11, 252, 252

        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # print("x_up3 out", x_up3.dense().shape)
        # 1, 32, 21, 504, 504

        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # print("x_up2 out", x_up2.dense().shape)
        # 1, 16, 41, 1008, 1008

        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv0)
        # print("x_up1 out", x_up1.dense().shape) 
        # 16, 41, 1008, 1008

        batch_dict['point_features'] = x_up1.features
        point_coords = common_utils.get_voxel_centers(
            x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict
