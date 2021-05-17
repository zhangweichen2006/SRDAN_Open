import torch
import numpy as np


class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config, nusc=False, fpn_layer=None):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]
        self.nusc = nusc
        # print("nusc", nusc)
        if self.nusc:
            self.anchor_velocities = [config['velocities'] for config in anchor_generator_config]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []

        # print("anchor_range", self.anchor_range)
        # -50.4 -50.4  -5.   50.4  50.4   3. 
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
            grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            # print("num_anchors_per_location", num_anchors_per_location)
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, z, num_size, num_rot, 7]

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            #anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
            # print("anchors all", anchors.shape) 
            # 126, 126, 1, 2, 7 # stride 8
            # 252, 252, 1, 2, 7? # stride 4
        return all_anchors, num_anchors_per_location

    def generate_anchors_range(self, grid_sizes):
        """
        Args:
            feature_size: list [D, H, W](zyx)
            sizes: [N, 3] list of list or array, size of anchors, xyz
            rotations: len(stride) num Reference
            velocities: ref velo along x and y axis.

        Returns:
            anchors: [*feature_size, num_sizes, num_rots, 9] tensor.
        """
        # anchor_range = self.anchor_range
        # print("anchor_range", anchor_range)
        # anchor_range = np.array(anchor_range, dtype)
        # print("feature_size", feature_size)
        # stride = (anchor_range[3] - anchor_range[0]) / feature_size[2]
        
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []

        # print("self.anchor_sizes", self.anchor_sizes)
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
            grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            # print("num_anchors_per_location", num_anchors_per_location)
                
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0


            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)

            # z_centers = np.linspace(
            #     anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype
            # )

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)

            velocities = self.anchor_velocities
            if velocities is not None:  
                velocities = x_shifts.new_tensor(velocities).reshape([-1, 2])
                combines = torch.cat([anchor_size, velocities], dim=1).reshape([-1, 5])
            else:
                combines = anchor_size

            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            # anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            combines = combines.view(1, 1, 1, -1, 5).repeat([*anchors.shape[0:3], 1, 1])

            #########

            anchors = torch.cat((anchors, combines), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, z, num_size, num_rot, 9]

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            #anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
            
        return all_anchors, num_anchors_per_location

if __name__ == '__main__':
    from easydict import EasyDict
    config = [
        EasyDict({
            'anchor_sizes': [[2.1, 4.7, 1.7], [0.86, 0.91, 1.73], [0.84, 1.78, 1.78]],
            'anchor_rotations': [0, 1.57],
            'anchor_heights': [0, 0.5]
        })
    ]

    A = AnchorGenerator(
        anchor_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        anchor_generator_config=config
    )
    import pdb
    pdb.set_trace()
    A.generate_anchors([[188, 188]])
