from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

# from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
from ..utils import box_utils, common_utils

import traceback


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None, pseudo=False, pseudo_set=None, vis=False, points_range=False, **kwargs):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.vis = vis
        self.class_names = class_names #['car']
        
        self.pseudo = pseudo
        self.points_range = points_range
        self.range_feat = self.dataset_cfg.get("RANGE_FEAT", False)
        
        # for pseudo only
        self.index_to_class_names = {}
        self.index_to_class_names[-1] = 'invalid'
        self.index_to_class_names[0] = 'bg'
        for i in range(len(self.class_names)):
            self.index_to_class_names[i+1] = self.class_names[i]

        self.pseudo_set = pseudo_set
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        print("self.dataset_cfg", self.dataset_cfg)
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        
        if dataset_cfg.DATASET == 'NuscenesDataset':
            self.point_feature_encoder = PointFeatureEncoder(
                    self.dataset_cfg.POINT_FEATURE_ENCODING,
                    point_cloud_range=self.point_cloud_range
            )
            # disable velo
            self.data_augmentor = DataAugmentor(
                self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger, nusc=False, pseudo=self.pseudo
            ) if self.training else None
            ###
            self.data_processor = DataProcessor(
                self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
                training=self.training
            )
            #####
            self.voxel_size = self.data_processor.voxel_size
            self.grid_size = self.data_processor.grid_size
            self.total_epochs = 0
            self._merge_all_iters_to_one_epoch = False

        elif dataset_cfg.DATASET == 'KittiDataset' or 'WaymoDaDataset':
            self.point_feature_encoder = PointFeatureEncoder(
                self.dataset_cfg.POINT_FEATURE_ENCODING,
                point_cloud_range=self.point_cloud_range
            )
            self.data_augmentor = DataAugmentor(
                self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger, pseudo=self.pseudo
            ) if self.training else None
            self.data_processor = DataProcessor(
                self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
            )

            self.grid_size = self.data_processor.grid_size
            self.voxel_size = self.data_processor.voxel_size
            self.total_epochs = 0
            self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...] (9+C)
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...] (9+C)
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
        # print("data_dict['gt_boxes'] ", data_dict['gt_boxes'] )
        
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            # print("gt_classes", gt_classes)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        # print("data_dict['gt_boxes'] after", data_dict['gt_boxes'] ) 7+ 1(class name idx)
        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict)
        # print("data_dict processor", data_dict)
        
        # if self.training:
        data_dict.pop('gt_names')

        return data_dict

    def _dict_select(self, dict_, inds):
        for k, v in dict_.items():
            if isinstance(v, dict):
                self._dict_select(v, inds)
            else:
                dict_[k] = v[inds]

    def prepare_data_nusc(self, input_dict,
                    voxel_generator=None,
                    max_voxels=20000,
                    remove_outside_points=False,
                    create_targets=True,
                    shuffle_points=False,
                    remove_unknown=False,
                    gt_rotation_noise=(-np.pi / 3, np.pi / 3),
                    gt_loc_noise_std=(1.0, 1.0, 1.0),
                    global_rotation_noise=(-np.pi / 4, np.pi / 4),
                    global_scaling_noise=(0.95, 1.05),
                    global_loc_noise_std=(0.2, 0.2, 0.2),
                    global_random_rot_range=(0.78, 2.35),
                    global_translate_noise_std=(0, 0, 0),
                    num_point_features=4,
                    anchor_area_threshold=1,
                    gt_points_drop=0.0,
                    gt_drop_max_keep=10,
                    remove_points_after_sample=True,
                    anchor_cache=None,
                    remove_environment=False,
                    random_crop=False,
                    reference_detections=None,
                    out_size_factor=2,
                    use_group_id=False,
                    multi_gpu=False,
                    min_points_in_gt=-1,
                    random_flip_x=True,
                    random_flip_y=True,
                    sample_importance=1.0,
                    out_dtype=np.float32,
                    with_velo=False):
        """convert point cloud to voxels, create targets if ground truths 
        exists.

        input_dict format: dataset.get_sensor_data format

        """
        '''
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
        '''

        # if dataset_cfg.DATASET in ["KittiDataset", "LyftDataset"]:
        #     points = input_dict["lidar"]["points"]
        # elif dataset_cfg.DATASET == "NuScenesDataset":
        # print('dataset training', self.training)
        points = input_dict["points_combined"]
        gt_dict = {}
        # print("self.training", self.training)

        ################## add anno ################
        if self.training:
            if self.pseudo:
                anno_dict = input_dict["annotations_lidar"]
                gt_dict = {
                    "gt_boxes": input_dict["pseudo_boxes"],
                    "gt_names": np.array([self.index_to_class_names[i] for i in input_dict["pseudo_classes_filtered"]]),
                    "gt_classes": input_dict["pseudo_classes_filtered"],
                    "gt_importance": input_dict["pseudo_importance"],
                }
                    
            else:
                anno_dict = input_dict["annotations_lidar"]
                gt_dict = {
                    "gt_boxes": anno_dict["boxes"],
                    "gt_names": anno_dict["names"],
                    "gt_importance": np.ones([anno_dict["boxes"].shape[0]], dtype=anno_dict["boxes"].dtype),
                }

                # print("out gtbox len", len(gt_dict['gt_boxes']))
                    
                if "difficulty" not in anno_dict:
                    difficulty = np.zeros([anno_dict["boxes"].shape[0]],
                                        dtype=np.int32)
                    gt_dict["difficulty"] = difficulty
                else:
                    gt_dict["difficulty"] = anno_dict["difficulty"]
                if use_group_id and "group_ids" in anno_dict:
                    group_ids = anno_dict["group_ids"]
                    gt_dict["group_ids"] = group_ids

        ################## add calib, rgb2points ################
        calib = None
        if "calib" in input_dict:
            calib = input_dict["calib"]

        # if reference_detections is not None:
        #     assert calib is not None and "image" in input_dict
        #     C, R, T = box_utils.projection_matrix_to_CRT_kitti(calib["P2"])
        #     frustums = box_utils.get_frustum_v2(reference_detections, C)
        #     frustums -= T
        #     frustums = np.einsum('ij, akj->aki', np.linalg.inv(R), frustums)
        #     frustums = box_utils.camera_to_lidar(frustums, rect, Trv2c)
        #     surfaces = box_utils.corner_to_surfaces_3d_jit(frustums)
        #     masks = points_in_convex_polygon_3d_jit(points, surfaces)
        #     points = points[masks.any(-1)]

        # if remove_outside_points:
        #     assert calib is not None
        #     image_shape = input_dict["image"]["image_shape"]
        #     points = box_utils.remove_outside_points(
        #         points, calib["rect"], calib["Trv2c"], calib["P2"], image_shape)
        # if remove_environment is True and training:
        #     selected = common_utils.keep_arrays_by_name(gt_dict['gt_names'], self.class_names)
        #     self._dict_select(gt_dict, selected)
        #     masks = self.points_in_rbbox(points, gt_dict["gt_boxes"])
        #     points = points[masks.any(-1)]

        ################## select bbox samples ################
        if self.training and not self.pseudo:
            """
            boxes_lidar = gt_dict["gt_boxes"]
            bev_map = simplevis.nuscene_vis(points, boxes_lidar)
            cv2.imshow('pre-noise', bev_map)
            """
            selected = common_utils.drop_arrays_by_name(gt_dict["gt_names"], ["DontCare", "ignore"])
            self._dict_select(gt_dict, selected)

            if remove_unknown:
                remove_mask = gt_dict["difficulty"] == -1
                """
                gt_boxes_remove = gt_boxes[remove_mask]
                gt_boxes_remove[:, 3:6] += 0.25
                points = prep.remove_points_in_boxes(points, gt_boxes_remove)
                """
                keep_mask = np.logical_not(remove_mask)

                self._dict_select(gt_dict, keep_mask)

            gt_dict.pop("difficulty")
            if min_points_in_gt > 0:
                # points_count_rbbox takes 10ms with 10 sweeps nuscenes data
                point_counts = box_utils.points_count_rbbox(points, gt_dict["gt_boxes"])
                mask = point_counts >= min_points_in_gt
                self._dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_)

            gt_dict["gt_boxes_mask"] = gt_boxes_mask
            
            # need pseudo filter ?
            # else:
            #     pseudo_class_filter_mask = input_dict["pseudo_classes_filtered"] >= 0
            #     print("pseudo_class_filter_mask", pseudo_class_filter_mask)
            #     self._dict_select(gt_dict, pseudo_class_filter_mask)
            #     print("filtered")
            #     print("gt_dict", gt_dict)

            ###### mask classes ###########

            if self.data_augmentor is not None:

                group_ids = None
                if "group_ids" in gt_dict:
                    group_ids = gt_dict["group_ids"]
                # print("gt_dict before",gt_dict)

                gt_dict["points"] = points
                # print("initial gt_dict", gt_dict['points'].shape) #voxelization

                if len(gt_dict['gt_boxes']) == 0:
                    new_index = np.random.randint(self.__len__())
                    return self.__getitem__(new_index)

                gt_dict = self.data_augmentor.forward(
                    data_dict={
                        **gt_dict
                    }
                )
                # print("after autg gt_dict", gt_dict['points'].shape) #voxelization
                # print("gt_dict after",gt_dict)

                if len(gt_dict['gt_boxes']) == 0:
                    new_index = np.random.randint(self.__len__())
                    return self.__getitem__(new_index)
                
                # sampled_dict = db_sampler.sample_all( finish
                     
            ####################### point feature encoder (Noise not included) ########################
                    # self.point_feature_encoder
            group_ids = None
            if "group_ids" in gt_dict:
                group_ids = gt_dict["group_ids"]

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]], dtype=np.int32)
            gt_dict["gt_classes"] = gt_classes

            # res["lidar"]["annotations"] = gt_dict
            #################################################################################

            # gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"],
            #                                             points, 0.5, random_flip_x, random_flip_y)
            # gt_dict["gt_boxes"], points = prep.global_rotation_v2(
            #     gt_dict["gt_boxes"], points, *global_rotation_noise)
            # gt_dict["gt_boxes"], points = prep.global_scaling_v2(
            #     gt_dict["gt_boxes"], points, *global_scaling_noise)

            # gt_dict["gt_boxes"], points = prep.global_translate_(gt_dict["gt_boxes"], points, global_translate_noise_std)
            
            # bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
            # mask = prep.filter_gt_box_outside_range_by_center(gt_dict["gt_boxes"], bv_range)
            # self._dict_select(gt_dict, mask)

            # limit rad to [-pi, pi] ???
            # gt_dict["gt_boxes"][:, 6] = self.limit_period(
            #     gt_dict["gt_boxes"][:, 6], offset=0.5, period=2 * np.pi)

            # boxes_lidar = gt_dict["gt_boxes"]
            # bev_map = simplevis.nuscene_vis(points, boxes_lidar)
            # cv2.imshow('post-noise', bev_map)
            # cv2.waitKey(0)
        # print("gt_dict['gt_boxes'] ", gt_dict['gt_boxes'] )

        ################## add class label ################

        if gt_dict.get('gt_boxes', None) is not None:
            if not self.pseudo:
                selected = common_utils.keep_arrays_by_name(gt_dict['gt_names'], self.class_names)
                gt_dict['gt_boxes'] = gt_dict['gt_boxes'][selected]
                gt_dict['gt_names'] = gt_dict['gt_names'][selected]
                gt_classes = np.array([self.class_names.index(n) + 1 for n in gt_dict['gt_names']], dtype=np.int32)
            else:
                # print("gt_dict[gt names]", gt_dict["gt_names"])
                selected = common_utils.drop_arrays_by_name(gt_dict["gt_names"], ["invalid"])
                fg_selected = common_utils.drop_arrays_by_name(gt_dict["gt_names"], ["invalid", "bg"])
                self._dict_select(gt_dict, selected)

                gt_classes = gt_dict['gt_classes']
            gt_boxes = np.concatenate((gt_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            gt_dict['gt_boxes'] = gt_boxes
        else:
            fg_selected = []
            # print("gt_dict['gt_boxes']", gt_dict['gt_boxes'].shape)
        # print("gt_dict['gt_boxes'] after", gt_dict['gt_boxes'] ) 9 + 1(class name idx) might be used for segmentation.
        #          
        if self.pseudo:
            # print("fg_selected", fg_selected)
            if len(fg_selected) == 0:
                # print("self.__len__()", self.__len__())
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            gt_dict['pseudo_weights'] = gt_dict['gt_importance']
        
        if not self.training or self.pseudo:
            gt_dict["points"] = points


        # print("gt_dict", gt_dict)
        # print("initial gt_dict points before", gt_dict['points'].shape) #voxelization

        gt_dict = self.point_feature_encoder.forward(gt_dict)

        # print("gt_dict after", gt_dict)
        # print("initial gt_dict points after", gt_dict['points'].shape) #voxelization

        gt_dict = self.data_processor.forward(data_dict=gt_dict)

        ################## visualization label ################

        if not self.training and self.vis:
            anno_dict = input_dict["annotations_lidar"]
            gt_dict["gt_boxes"] = anno_dict["boxes"]
            gt_dict["gt_names"] = anno_dict["names"]

        # print("class_names", anno_dict["class_names"])
        # print("gt_names", gt_dict["gt_names"])

        # print("PRINTE")
        if self.points_range:
            # range_points = gt_dict["points"][:,:3]
            # zero = np.zeros(3)
            # points_range = np.array([np.linalg.norm(i-zero) for i in range_points])
            points_range = np.linalg.norm(gt_dict["points"][:, 0:3], axis=1)
            gt_dict['points_range'] = points_range
            # print("points_range", points_range[..., np.newaxis].shape)
            if self.range_feat:
                # print("gt_dict[ppoint]", gt_dict["points"].shape)
                gt_dict["points"] = np.concatenate((gt_dict["points"], points_range[..., np.newaxis]), axis=1)
                # print("gt_dict['points'] after", gt_dict["points"].shape)
                # print("last point in", gt_dict["points"][-1])
                # print("last range", points_range[-1])

        # get gt name only for visualization

        # print("gt_dict", gt_dict)
        if not self.training and self.vis:
            gt_dict.pop("gt_names")

        if not self.training:
            gt_dict['metadata'] = input_dict['metadata']

            return gt_dict

    
        gt_dict.pop('gt_names')

        return gt_dict

        ###########################
        #### Did not complete #####
        ###########################

        # if self.pseudo:
        #     print("data_dict processor psue", gt_dict.keys())
        # 
        # train  gt_dict.keys())
        # 'gt_boxes', 'gt_names', 'gt_importance', 'points', 'gt_classes', 
        # 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points'

        #voxelization
        # metrics = {}

        # # [0, -40, -3, 70.4, 40, 1]
        # voxel_size = voxel_generator.voxel_size
        # pc_range = voxel_generator.point_cloud_range
        # grid_size = voxel_generator.grid_size
        # # [352, 400]
        # if not multi_gpu:
        #     res = voxel_generator.generate(
        #         points, max_voxels)
        #     voxels = res["voxels"]
        #     coordinates = res["coordinates"]
        #     num_points = res["num_points_per_voxel"]
        #     num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        # else:
        #     res = voxel_generator.generate_multi_gpu(
        #         points, max_voxels)
        #     voxels = res["voxels"]
        #     coordinates = res["coordinates"]
        #     num_points = res["num_points_per_voxel"]
        #     num_voxels = np.array([res["voxel_num"]], dtype=np.int64)
        # metrics["voxel_gene_time"] = time.time() - t1
        # example = {
        #     'voxels': voxels,
        #     'num_points': num_points,
        #     'coordinates': coordinates,
        #     "num_voxels": num_voxels,
        #     "metrics": metrics,
        # }
        # if calib is not None:
        #     example["calib"] = calib
        # feature_map_size = grid_size[:2] // out_size_factor
        # feature_map_size = [*feature_map_size, 1][::-1]
        # if anchor_cache is not None:
        #     anchors = anchor_cache["anchors"]
        #     anchors_bv = anchor_cache["anchors_bv"]
        #     anchors_dict = anchor_cache["anchors_dict"]
        #     matched_thresholds = anchor_cache["matched_thresholds"]
        #     unmatched_thresholds = anchor_cache["unmatched_thresholds"]

        # else:
        #     ret = target_assigner.generate_anchors(feature_map_size)
        #     anchors = ret["anchors"]
        #     anchors = anchors.reshape([-1, target_assigner.box_ndim])
        #     anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
        #     anchors_bv = self.rbbox2d_to_near_bbox(
        #         anchors[:, [0, 1, 3, 4, 6]])
        #     matched_thresholds = ret["matched_thresholds"]
        #     unmatched_thresholds = ret["unmatched_thresholds"]
        # example["anchors"] = anchors
        # anchors_mask = None
        # if anchor_area_threshold >= 0:
        #     # slow with high resolution. recommend disable this forever.
        #     coors = coordinates
        #     dense_voxel_map = self.sparse_sum_for_anchors_mask(
        #         coors, tuple(grid_size[::-1][1:]))
        #     dense_voxel_map = dense_voxel_map.cumsum(0)
        #     dense_voxel_map = dense_voxel_map.cumsum(1)
        #     anchors_area = self.fused_get_anchors_area(
        #         dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
        #     anchors_mask = anchors_area > anchor_area_threshold
        #     # example['anchors_mask'] = anchors_mask.astype(np.uint8)
        #     example['anchors_mask'] = anchors_mask
        # print("prep time", time.time() - t)
        # metrics["prep_time"] = time.time() - t
        # print("transform_points_to_voxels", gt_dict)

        # print("gt_dict", gt_dict)

        # gt_dict["frame_id"] = input_dict["frame_id"]


        # assign target and anchors

        # if create_targets:
        #     # t1 = time.time()
        #     targets_dict = target_assigner.assign(
        #         anchors,
        #         anchors_dict,
        #         gt_dict["gt_boxes"],
        #         anchors_mask,
        #         gt_classes=gt_dict["gt_classes"],
        #         gt_names=gt_dict["gt_names"],
        #         matched_thresholds=matched_thresholds,
        #         unmatched_thresholds=unmatched_thresholds,
        #         importance=gt_dict["gt_importance"])
            
        #     """
        #     boxes_lidar = gt_dict["gt_boxes"]
        #     bev_map = simplevis.nuscene_vis(points, boxes_lidar, gt_dict["gt_names"])
        #     assigned_anchors = anchors[targets_dict['labels'] > 0]
        #     ignored_anchors = anchors[targets_dict['labels'] == -1]
        #     bev_map = simplevis.draw_box_in_bev(bev_map, [-50, -50, 3, 50, 50, 1], ignored_anchors, [128, 128, 128], 2)
        #     bev_map = simplevis.draw_box_in_bev(bev_map, [-50, -50, 3, 50, 50, 1], assigned_anchors, [255, 0, 0])
        #     cv2.imshow('anchors', bev_map)
        #     cv2.waitKey(0)
            
        #     boxes_lidar = gt_dict["gt_boxes"]
        #     pp_map = np.zeros(grid_size[:2], dtype=np.float32)
        #     voxels_max = np.max(voxels[:, :, 2], axis=1, keepdims=False)
        #     voxels_min = np.min(voxels[:, :, 2], axis=1, keepdims=False)
        #     voxels_height = voxels_max - voxels_min
        #     voxels_height = np.minimum(voxels_height, 4)
        #     # sns.distplot(voxels_height)
        #     # plt.show()
        #     pp_map[coordinates[:, 1], coordinates[:, 2]] = voxels_height / 4
        #     pp_map = (pp_map * 255).astype(np.uint8)
        #     pp_map = cv2.cvtColor(pp_map, cv2.COLOR_GRAY2RGB)
        #     pp_map = simplevis.draw_box_in_bev(pp_map, [-50, -50, 3, 50, 50, 1], boxes_lidar, [128, 0, 128], 1)
        #     cv2.imshow('heights', pp_map)
        #     cv2.waitKey(0)
        #     """
        #     example.update({
        #         'labels': targets_dict['labels'],
        #         'reg_targets': targets_dict['bbox_targets'],
        #         # 'reg_weights': targets_dict['bbox_outside_weights'],
        #         'importance': targets_dict['importance'],
        #     })

        # # give sample idx number
        # gt_dict['idx'] = input_dict['idx']

        # return example

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)

        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            # print("key", key)
            # print("val", val)
            try:
                if key in ['voxels', 'voxel_num_points', 'points_range']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_importance','gt_classes','pseudo_weights']:
                    ret[key] = np.concatenate(val, axis=0)
                else:
                    # if key == 'gt_names':
                    #     ret[key] = val.tolist()
                    ret[key] = np.stack(val, axis=0)
                
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
            
        ret['batch_size'] = batch_size
        return ret


    # @staticmethod
    # def collate_batch_pseudo(batch_list, _unused=False):
    #     data_dict = defaultdict(list)

    #     for cur_sample in batch_list:
    #         for key, val in cur_sample.items():
    #             data_dict[key].append(val)
    #     batch_size = len(batch_list)
    #     ret = {}

    #     for key, val in data_dict.items():
    #         try:
    #             if key in ['voxels', 'voxel_num_points']:
    #                 ret[key] = np.concatenate(val, axis=0)
    #                 # sinlge val
    #                 # for i, coor in enumerate(val):
    #                 #     ret[f'{key}_single{i}'] = coor
    #             elif key in ['points', 'voxel_coords']:
    #                 coors = []
    #                 for i, coor in enumerate(val):
    #                     coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
    #                     coors.append(coor_pad)
    #                     # sinlge val
    #                     # ret[f'{key}_single{i}'] = coor
    #                 ret[key] = np.concatenate(coors, axis=0)
    #             elif key in ['gt_boxes']:
    #                 max_gt = max([len(x) for x in val])
    #                 batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
    #                 for k in range(batch_size):
    #                     batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
    #                 ret[key] = batch_gt_boxes3d
    #             elif key in ['gt_importance','gt_classes','pseudo_weights']:
    #                 ret[key] = np.concatenate(val, axis=0)
    #             else:
    #                 ret[key] = np.stack(val, axis=0)
                
    #         except:
    #             print('Error in collate_batch: key=%s' % key)
    #             raise TypeError
            
    #         # if key == 'points':
    #         #     print('after key', ret[key].shape)
    #         # print('after val', len(val))
    #         # print('after ret[key]', ret[key].shape)

    #     ret['batch_size'] = batch_size
    #     return ret