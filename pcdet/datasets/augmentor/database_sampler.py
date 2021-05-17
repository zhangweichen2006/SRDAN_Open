import pickle

import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils


class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None, nusc=False):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        self.nusc = nusc
        # print("root_path", root_path)
        for class_name in class_names:
            self.db_infos[class_name] = []

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                # print("infos_keys", infos.keys())
                for cur_class in class_names:
                    self.db_infos[cur_class].extend(infos[cur_class])
                    # if 'waymo' in str(root_path).lower() and 'VEHICLE' in cur_class: # Waymo_DA_DATASET
                    #     db_class = 'Car'
                    # else:
                    #     db_class = cur_class
                    # self.db_infos[cur_class].extend(infos[db_class])

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        self.add_rgb_to_points = sampler_cfg.get('ADD_RGB_TO_POINTS', False)
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            # sample_num = int(sample_num) ##### add
            self.sample_class_num[class_name] = sample_num

            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0
        # print("sample_num", sample_num)
        # print("indices[pointer: pointer + sample_num]", indices[pointer: pointer + sample_num])
        # print("self.db_infos[class_name][idx]", self.db_infos[class_name][0])
        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        # print("sampled_dict sample_with_fixed_number", sampled_dict)

        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict, nusc=False):
        # print("add_sampled_boxes_to_scene data", data_dict)
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            ### POINTRCNN NUSC ONLY
            # obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
            #     [-1, 5])[:, :4]

            obj_points[:, :3] += info['box3d_lidar'][:3]

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        if nusc:
            large_sampled_gt_boxes = box_utils.enlarge_box3d(
                sampled_gt_boxes[:, 0:9], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
            )
        else:
            large_sampled_gt_boxes = box_utils.enlarge_box3d(
                sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
            )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes, nusc=self.nusc)
        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points

        return data_dict

    # def add_sampled_boxes_to_scene_nusc(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):

    #     gt_boxes_mask = data_dict['gt_boxes_mask']
    #     gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
    #     # print('gt_boxes', gt_boxes.shape)
    #     gt_names = data_dict['gt_names'][gt_boxes_mask]
    #     points = data_dict['points']

    #     # print("gt_boxes_mask", gt_boxes_mask)
    #     if self.sampler_cfg.get('USE_ROAD_PLANE', False):
    #         sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
    #             sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
    #         )
    #         data_dict.pop('calib')
    #         data_dict.pop('road_plane')

    #     obj_points_list = []
    #     # print("total_valid_sampled_dict", total_valid_sampled_dict)
    #     for idx, info in enumerate(total_valid_sampled_dict):
    #         file_path = self.root_path / info['path']
    #         obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
    #             [-1, self.sampler_cfg.NUM_POINT_FEATURES])

    #         obj_points[:, :3] += info['box3d_lidar'][:3]

    #         if self.sampler_cfg.get('USE_ROAD_PLANE', False):
    #             # mv height
    #             obj_points[:, 2] -= mv_height[idx]

    #         obj_points_list.append(obj_points)

    #     obj_points = np.concatenate(obj_points_list, axis=0)
    #     sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

    #     # print("sampled_gt_names", sampled_gt_names)

    #     ###### Additional Random Crop Augmentation ###########
    #     # if random_crop:
    #     #     s_points_list_new = []
    #     #     assert calib is not None
    #     #     rect = calib["rect"]
    #     #     Trv2c = calib["Trv2c"]
    #     #     P2 = calib["P2"]
    #     #     gt_bboxes = box_np_ops.box3d_to_bbox(sampled_gt_boxes, rect, Trv2c, P2)
    #     #     crop_frustums = prep.random_crop_frustum(gt_bboxes, rect, Trv2c, P2)
    #     #     for i in range(crop_frustums.shape[0]):
    #     #         s_points = s_points_list[i]
    #     #         mask = prep.mask_points_in_corners(
    #     #             s_points, crop_frustums[i : i + 1]
    #     #         ).reshape(-1)
    #     #         num_remove = np.sum(mask)
    #     #         if num_remove > 0 and (s_points.shape[0] - num_remove) > 15:
    #     #             s_points = s_points[np.logical_not(mask)]
    #     #         s_points_list_new.append(s_points)
    #     #     s_points_list = s_points_list_new

    #     large_sampled_gt_boxes = box_utils.enlarge_box3d(
    #         sampled_gt_boxes[:, 0:9], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
    #     )
    #     points = box_utils.remove_points_in_boxes3d_nusc(points, large_sampled_gt_boxes)
    #     # print("remove_points_in_boxes3d points", points)
    #     # print("remove_points_in_boxes3d obj_points", obj_points)
    #     points = np.concatenate([obj_points, points], axis=0)
    #     gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
    #     gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
    #     data_dict['gt_boxes'] = gt_boxes
    #     data_dict['gt_names'] = gt_names
    #     data_dict['points'] = points

    #     # print("data_dict points", data_dict)
    #     # print("points after", data_dict['points'])
    #     # print("data_dict['gt_names']", data_dict['gt_names'])
    #     return data_dict

    def add_rgb_to_points_dict(self, data_dict):
        if "calib" in data_dict:
            calib = data_dict["calib"]
        else:
            calib = None

        assert calib is not None and "image" in data_dict
        image_path = data_dict["image"]["image_path"]
        image = (
            imgio.imread(str(pathlib.Path(self.root_path) / image_path)).astype(
                np.float32
            )
            / 255
        )
        points_rgb = box_utils.add_rgb_to_points(
            points, image, calib["rect"], calib["Trv2c"], calib["P2"]
        )
        points = np.concatenate([points, points_rgb], axis=1)
        data_dict['points'] = points
        # num_point_features += 3

        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                ###### kitti 7 dim ########
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                ####### nuscenes 9 dim !!!!!! #############
                gt_boxes: (N, 9 + C) ?
        Returns:

        """
        # print("sampler data_dictv  ", data_dict)
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        sampled = []
        sampled_gt_boxes = []

        # print("self.sample_groups.items()", self.sample_groups.items())
        for class_name, sample_group in self.sample_groups.items():

            # print("class_name", class_name)
            # print("sample_group", sample_group)
            # print("self.limit_whole_scene", self.limit_whole_scene)

            ## car 2
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names) # 2
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)

            # print("sample_group", sample_group) 78884 car
            # print("self.sample_class_num[class_name])", self.sample_class_num[class_name]) 2
            # print("int(sample_group['sample_num'])", int(sample_group['sample_num'])) 2

            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)
                # print("sampled_dict sample_with_fixed_number out ", sampled_dict)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                # print("existed_boxes", existed_boxes.shape) #n * 9 for nusc
                # print("sampled_boxes", sampled_boxes.shape)

                # else:
                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes, existed_boxes, nusc=self.nusc)
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes, sampled_boxes, nusc=self.nusc)
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)


        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            # if self.nusc:
            #     data_dict = self.add_sampled_boxes_to_scene_nusc(data_dict, sampled_gt_boxes, total_valid_sampled_dict)
            # else:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict, nusc=self.nusc)

            data_dict.pop('gt_boxes_mask')

        if self.add_rgb_to_points:
            data_dict = self.add_rgb_to_points_dict(data_dict)

        return data_dict
