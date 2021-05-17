import torch
import torch.nn as nn

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class VoxelSetAbstraction(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, nusc=False, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.nusc = nusc

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]

            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg['raw_points'].POOL_RADIUS,
                nsamples=SA_cfg['raw_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        #POINTRCNN COMMENT
        # print("self.model_cfg.NUM_OUTPUT_FEATURES", self.model_cfg.NUM_OUTPUT_FEATURES)
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES #4
        self.num_point_features_before_fusion = c_in

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def get_sampled_points(self, batch_dict):
        # if self.nusc:
        #     point_num = 5
        # else:
        point_num = 4

        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:point_num]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:point_num],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3) # B M 4!
        # print("keypoints fuc", keypoints.shape)
        return keypoints

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        # for i in range(self.range_division):
        keypoints = self.get_sampled_points(batch_dict)

        # print("keypoints", keypoints.shape)
        # 1, 2048, 3

        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            )
            point_features_list.append(point_bev_features)

            # print("point_bev_features", point_bev_features.shape)
            # 1, 2048, 256

        # print("point_features_list", [point_features_list[i].shape for i in len(point_features_list)])

        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        # 2048 * 3
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
            
            # print("xyz_batch_cnt", xyz_batch_cnt.shape)
            # print("raw_points", raw_points.shape)
            point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None

            # print("point_features fore", point_features.shape)

            pooled_points, pooled_features = self.SA_rawpoints(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features,
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
            # print("pooled_points", pooled_points.shape)
            # [2048, 3]
            # print("pooled_features", pooled_features.shape)
            # [2048, 32]

        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            pooled_points, pooled_features = self.SA_layers[k](
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous(),
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

            # print("multi_scale_3d_features", pooled_features.shape)
            # 2048, 64,
            # 2048, 128,
            # 2048, 128

        # print("point_features_list", [point_features_list[i].shape for i in len(point_features_list)])

        point_features = torch.cat(point_features_list, dim=2)

        # print("point_features", point_features.shape) # 1, 2048, 640

        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)

        # 1, 2048, 1

        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

        # 1, 2048, 3

        # print("point_coords cat", point_coords.shape)

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])

        # move to point head
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        # print("vsa_point_feature_fusion", point_features.shape)
        # 2048 * 128
        batch_dict['point_features'] = point_features  # (BxN, C)
        # print("batch_dict['point_features']", batch_dict['point_features'].shape)
        # (1*2048), 128
        batch_dict['point_coords'] = point_coords  # (BxN, 4)
        # (1*2048), 4
        return batch_dict

class VoxelSetAbstractionRange(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, nusc=False, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.nusc = nusc
        self.range_division = self.model_cfg.get('RANGE_DIVISION', None)
        
        if self.range_division is not None:
            self.short_range_min = self.range_division['short'][0]
            self.short_range_max = self.range_division['short'][1]
            self.mid_range_min = self.range_division['mid'][0]
            self.mid_range_max = self.range_division['mid'][1]
            self.long_range_min = self.range_division['long'][0]
            self.long_range_max = self.range_division['long'][1]

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]

            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg['raw_points'].POOL_RADIUS,
                nsamples=SA_cfg['raw_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        # print("keypoints", keypoints.shape) 1024, 3
        # print("batch_size", batch_size) 1
        # print("bev_features", bev_features.shape) 256, 126, 126
        # print("bev_stride", bev_stride) 8
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))
            # print("point_bev_features_list", point_bev_features.unsqueeze(dim=0).shape)
            # 1024,256

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        # print("point_bev_features", point_bev_features.shape) # 1024,256
        return point_bev_features

    def get_sampled_points(self, batch_dict):
        # if self.nusc:
        #     point_num = 5
        # else:
        point_num = 4

        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:point_num]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:point_num],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list_short = []
        keypoints_list_mid = []
        keypoints_list_long = []

        pt_range = batch_dict['points_range']

        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            masked_points = src_points[bs_mask]
            masked_range = pt_range[bs_mask]

            short_mask = (masked_range > self.short_range_min) & (masked_range < self.short_range_max)
            long_mask = masked_range > self.long_range_min
            mid_mask = ~long_mask & ~short_mask

            sampled_points_short = masked_points[short_mask].unsqueeze(dim=0)  # (1, N, 3)
            sampled_points_mid = masked_points[mid_mask].unsqueeze(dim=0)  # (1, N, 3)
            sampled_points_long = masked_points[long_mask].unsqueeze(dim=0)  # (1, N, 3)
            # print("ori", masked_points.shape[0])
            # print("sampled_points_short", sampled_points_short.shape[1])
            # print("sampled_points_mid", sampled_points_mid.shape[1])
            # print("sampled_points_long", sampled_points_long.shape[1])
            # print("total", sampled_points_short.shape[1] + sampled_points_mid.shape[1] + sampled_points_long.shape[1])

            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs_short = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points_short[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS_RANGE['short']
                ).long()
                cur_pt_idxs_mid = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points_mid[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS_RANGE['mid']
                ).long()
                cur_pt_idxs_long = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points_long[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS_RANGE['long']
                ).long()

                if sampled_points_short.shape[1] < self.model_cfg.NUM_KEYPOINTS_RANGE['short']:
                    empty_num_short = self.model_cfg.NUM_KEYPOINTS_RANGE['short'] - sampled_points_short.shape[1]
                    cur_pt_idxs_short[0, -empty_num_short:] = cur_pt_idxs_short[0, :empty_num_short]
                if sampled_points_mid.shape[1] < self.model_cfg.NUM_KEYPOINTS_RANGE['mid']:
                    empty_num_mid = self.model_cfg.NUM_KEYPOINTS_RANGE['mid'] - sampled_points_mid.shape[1]
                    cur_pt_idxs_mid[0, -empty_num_mid:] = cur_pt_idxs_mid[0, :empty_num_mid]
                if sampled_points_long.shape[1] < self.model_cfg.NUM_KEYPOINTS_RANGE['long']:
                    empty_num_long = self.model_cfg.NUM_KEYPOINTS_RANGE['long'] - sampled_points_long.shape[1]
                    cur_pt_idxs_long[0, -empty_num_long:] = cur_pt_idxs_long[0, :empty_num_long]

                keypoints_short = sampled_points_short[0][cur_pt_idxs_short[0]].unsqueeze(dim=0)
                keypoints_mid = sampled_points_mid[0][cur_pt_idxs_mid[0]].unsqueeze(dim=0)
                keypoints_long = sampled_points_long[0][cur_pt_idxs_long[0]].unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError

            keypoints_list_short.append(keypoints_short)
            keypoints_list_mid.append(keypoints_mid)
            keypoints_list_long.append(keypoints_long)

        keypoints_short = torch.cat(keypoints_list_short, dim=0)  # (B, M, 3) # B M 4!
        keypoints_mid = torch.cat(keypoints_list_mid, dim=0)  # (B, M, 3) # B M 4!
        keypoints_long = torch.cat(keypoints_list_long, dim=0)  # (B, M, 3) # B M 4!
        # print("keypoints list", keypoints_long.shape)
        return {'short': keypoints_short, 'mid': keypoints_mid, 'long': keypoints_long}

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints_range = self.get_sampled_points(batch_dict)

        # print("keypoints", keypoints.shape)
        # 1, 2048, 3

        for i in keypoints_range.keys():
            keypoints = keypoints_range[i]
            point_features_list = []
            if 'bev' in self.model_cfg.FEATURES_SOURCE:
                point_bev_features = self.interpolate_from_bev_features(
                    keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                    bev_stride=batch_dict['spatial_features_stride']
                )
                point_features_list.append(point_bev_features)

                # print("point_bev_features", point_bev_features.shape)
                # 1, 2048, 256

            # print("point_features_list 1", [point_features_list[i].shape for i in len(point_features_list)])

            batch_size, num_keypoints, _ = keypoints.shape
            new_xyz = keypoints.view(-1, 3)
            # 2048 * 3
            new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

            if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
                raw_points = batch_dict['points']
                xyz = raw_points[:, 1:4]
                xyz_batch_cnt = xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
                
                # print("xyz_batch_cnt", xyz_batch_cnt.shape)
                # print("raw_points", raw_points.shape)
                point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None

                # print("point_features fore", point_features.shape)

                pooled_points, pooled_features = self.SA_rawpoints(
                    xyz=xyz.contiguous(),
                    xyz_batch_cnt=xyz_batch_cnt,
                    new_xyz=new_xyz,
                    new_xyz_batch_cnt=new_xyz_batch_cnt,
                    features=point_features,
                )
                point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

                # print("pooled_points", pooled_points.shape)
                # # [2048, 3]
                # print("pooled_features", pooled_features.shape)
                # # [2048, 32]

            for k, src_name in enumerate(self.SA_layer_names):
                cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
                xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=self.downsample_times_map[src_name],
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )
                xyz_batch_cnt = xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

                pooled_points, pooled_features = self.SA_layers[k](
                    xyz=xyz.contiguous(),
                    xyz_batch_cnt=xyz_batch_cnt,
                    new_xyz=new_xyz,
                    new_xyz_batch_cnt=new_xyz_batch_cnt,
                    features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous(),
                )
                point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

                # print("multi_scale_3d_features", pooled_features.shape)
                # 2048, 64,
                # 2048, 128,
                # 2048, 128

            # print("point_features_list", [point_features_list[i].shape for i in range(len(point_features_list))])
            # b,1024,256
            # b,1024,32
            # b,1024,64
            # b,1024,128
            # b,1024,128

            point_features = torch.cat(point_features_list, dim=2)

            # print("point_features", point_features.shape) # 1, 2048, 640

            batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)

            # 1, 2048, 1

            point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

            # 1, 2048, 3

            # print("point_coords cat", point_coords.shape)

            batch_dict[f'point_features_before_fusion_{i}'] = point_features.view(-1, point_features.shape[-1])
            point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

            # print("vsa_point_feature_fusion", point_features.shape)
            # 2048 * 128
            batch_dict[f'point_features_{i}'] = point_features  # (BxN, C)
            batch_dict[f'point_coords_{i}'] = point_coords  # (BxN, 4)
            # (1*2048), 4 batch_dict['points']


            # print(f"batch_dict['point_features_before_fusion_{i}']", batch_dict[f'point_features_before_fusion_{i}'].shape)
            # print(f"batch_dict['point_features_{i}']", batch_dict[f'point_features_{i}'].shape)
            # (1*2048), 128
            
        return batch_dict
