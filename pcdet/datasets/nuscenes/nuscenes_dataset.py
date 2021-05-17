import pickle
import copy
import numpy as np
from skimage import io
import operator
from pcdet.utils import box_utils, common_utils, calibration_kitti, object3d_kitti
from pcdet.datasets import DatasetTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from tqdm import tqdm
import torch.distributed as dist
from typing import Tuple, List
import pickle
from functools import reduce
# from ...utils import box_utils, common_utils, calibration_kitti, object3d_kitti
# from ..dataset import DatasetTemplate
# from ...ops.roiaware_pool3d import roiaware_pool3d_utils
# ############
import os
import ast
import sys
import pickle
import copy
import numpy as np
from skimage import io
from pathlib import Path
import torch
import spconv
#############
import json
import time
from copy import deepcopy
import random
from functools import partial
import subprocess
from pyquaternion import Quaternion
# #############

from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval

general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bicycle": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bus": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
}

class NuscenesDataset(DatasetTemplate):

    def __init__(self, dataset_cfg,
                 class_names=None,
                 root_path=None, training=True,
                 logger=None, info_path=None, n_sweeps=0,
                 pseudo=False, pseudo_set=None, vis=False, points_range=False, **kwargs):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger, pseudo=pseudo, vis=vis, pseudo_set=pseudo_set, points_range=points_range
        )
        # print("dataset vis",  vis)
        # print('num_point_features', num_point_features)
        # x, y, z, intensity, ring_index
        self.num_point_features = 5#Pointrcnn num_point_features 4
        if self.range_feat:
            self.num_point_features += 1

        self.n_sweeps = dataset_cfg.N_SWEEPS
        assert self.n_sweeps > 0, "At least input one sweep please!"

        self.root_path = root_path
        self.training = training
        self.class_balanced = True
        # add validation samples in source
        if self.training and not self.pseudo:
            self.info_path = dataset_cfg.INFO_PATH
            # self.info_path_val = dataset_cfg.VAL_INFO_PATH
        else:
            self.info_path = dataset_cfg.VAL_INFO_PATH#INFO_PATH#
            # self.info_path_val = dataset_cfg.VAL_INFO_PATH

        print('self.info_path', self.info_path)
        ######## new ######## !!!!!
        self.class_names = class_names
        self.name_mapping = general_to_detection

        if not hasattr(self, "nusc_infos"):
            self.load_infos()

        # with open(self.info_path, 'rb') as f:
        #     infos = pickle.load(f)

        # # self.nusc_pkl = self.include_nusc_data()
        # self.nusc_infos = infos["infos"]
        # self.metadata = infos["metadata"]
        self.nusc_infos = list(sorted(self.nusc_infos, key=lambda e: e["timestamp"]))

        self.nusc_infos_token = {}
        for i in range(len(self.nusc_infos)):
            self.nusc_infos_token[self.nusc_infos[i]['token']] = self.nusc_infos[i]

        # metadata

        # if self.pseudo:
            # print("self.nusc_infos_token", self.nusc_infos_token.keys())
        # print("self.nusc_infos_token", self.nusc_infos_token.keys())
        # kitti map: nusc det name -> kitti eval name
        # self.kitti_name_mapping = {
        #     "car": "Car",
        #     "pedestrian": "Pedestrian",
        # }  # we only eval these classes in kitti
        # self.version = self.metadata["version"]
        self.version = "v1.0-trainval"#dataset_cfg.VERSION#
        self.eval_version = "cvpr_2019"
        self.with_velocity = False

        self.sample_id_list = [i for i in range(len(self.nusc_infos))]#self.sample_id()
        print("len(self.nusc_infos)", len(self.nusc_infos))
    def load_infos(self):
        if self.logger is not None:
            self.logger.info('Loading nusc dataset')

        with open(self.info_path, "rb") as f:
            nusc_infos_all = pickle.load(f)

        # with open(self.info_path_val, "rb") as f2:
        #     nusc_infos_all_val = pickle.load(f2)

        ###### class balanced #######
        if self.training and self.class_balanced:
            self.frac = int(len(nusc_infos_all) * 0.25)

            cls_infos = {name: [] for name in self.class_names}
            for info in nusc_infos_all:
                for name in set(info["gt_names"]):
                    if name in self.class_names:
                        cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
            # class ratio
            cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

            self.nusc_infos = []

            frac = 1.0 / len(self.class_names)
            ratios = [frac / v for v in cls_dist.values()]

            for cls_infos, ratio in zip(list(cls_infos.values()), ratios):
                self.nusc_infos += np.random.choice(
                    cls_infos, int(len(cls_infos) * ratio)
                ).tolist()

            cls_infos = {name: [] for name in self.class_names}
            for info in self.nusc_infos:
                for name in set(info["gt_names"]):
                    if name in self.class_names:
                        cls_infos[name].append(info)
            # class selected distributions
            cls_dist = {k: len(v) / len(self.nusc_infos) for k, v in cls_infos.items()}

        ####### normal ########
        else:# self.training and not self.pseudo:
            if isinstance(nusc_infos_all, dict):
                self.nusc_infos = []
                for v in nusc_infos_all.values():
                    self.nusc_infos.extend(v)
            else:
                self.nusc_infos = nusc_infos_all

            print("nusc_infos_all", len(nusc_infos_all))

            # if isinstance(nusc_infos_all_val, dict):
            #     for v in nusc_infos_all_val.values():
            #         self.nusc_infos.extend(v)
            # else:
            #     self.nusc_infos.extend(nusc_infos_all_val)

        # else:
        #     if isinstance(nusc_infos_all, dict):
        #         self.nusc_infos = []
        #         for v in nusc_infos_all.values():
        #             self.nusc_infos.extend(v)
        #     else:
        #         self.nusc_infos = nusc_infos_all

        self.logger.info('Total samples for nusc dataset: {}'.format(len(self.nusc_infos)))

    # def reset(self):
    #     self.logger.info(f"re-sample {self.frac} frames from full set")
    #     random.shuffle(self.nusc_infos)
    #     self.nusc_infos = self.nusc_infos[: self.frac]

    def get_sensor_data_raw(self, idx):

        info = self.nusc_infos[idx]
        # print("raw info", info)

        res = {
            "points": None,
            "n_sweeps": self.n_sweeps,
            "annotations_lidar": None,
            # "ground_plane": -gp[-1] if with_gp else None,
            "image_prefix": self.root_path,
            "calib": None,
            "image": None,
            "mode": "train" if self.training else "val",
            "metadata": {
                "image_prefix": self.dataset_cfg.DATA_PATH,
                "num_point_features": self.num_point_features,
                "token": info["token"],
            },
        }
            # "frame_id": idx

        res, info = self.load_point_cloud(res, info)
        res, info = self.load_point_cloud_anno(res, info)

        # print("get_sensor_data_raw", res)
        # 'points': n*4
        # 'n_sweeps': 10
        # 'annotations_lidar': 'boxes': n*9,
        # 'names': 'car', 'truck'
        # 'tokens': [n]
        # 'velocities': [n*3]
        # 'image_prefix': None
        # 'calib': None
        # 'image': None
        # 'mode': 'train'
        # 'metadata': {n}
        # 'times': n (0,0,0.25...)
        # 'points_combined': n*5

        return res

    def read_file(self, path, tries=2, num_point_feature=4):
        points = None
        try_cnt = 0
        while points is None and try_cnt < tries:
            try_cnt += 1
            try:
                points = np.fromfile(path, dtype=np.float32)
                s = points.shape[0]
                if s % 5 != 0:
                    points = points[: s - (s % 5)]
                points = points.reshape(-1, 5)[:, :num_point_feature]
            except Exception:
                points = None

        return points


    def read_sweep(self, sweep):
        min_distance = 1.0
        # points_sweep = np.fromfile(str(sweep["lidar_path"]),
        #                            dtype=np.float32).reshape([-1,
        #                                                       5])[:, :4].T
        points_sweep = self.read_file(str(sweep["lidar_path"])).T
        points_sweep = self.remove_close(points_sweep, min_distance) # remove in its local coordinate

        nbr_points = points_sweep.shape[1]
        if sweep["transform_matrix"] is not None:
            points_sweep[:3, :] = sweep["transform_matrix"].dot(
                np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
            )[:3, :]
        # points_sweep[3, :] /= 255

        curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

        return points_sweep.T, curr_times.T

    def remove_close(self, points, radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """
        x_filt = np.abs(points[0, :]) < radius
        y_filt = np.abs(points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        points = points[:, not_close]

        return points

    def load_point_cloud(self, res, info):

        n_sweeps = res["n_sweeps"]

        lidar_path = Path(info["lidar_path"])
        points = self.read_file(str(lidar_path))

        # points[:, 3] /= 255
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        assert (n_sweeps - 1) <= len(
            info["sweeps"]
        ), "n_sweeps {} should not greater than list length {}.".format(
            n_sweeps, len(info["sweeps"])
        )

        for i in np.random.choice(len(info["sweeps"]), n_sweeps - 1, replace=False):
            sweep = info["sweeps"][i]
            points_sweep, times_sweep = self.read_sweep(sweep)
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        res["points"] = points
        res["times"] = times
        res["points_combined"] = np.hstack([points, times])

        # POINTRCNN NUSCENS COMMENT
        # res["points_combined"] = points
        return res, info

    def load_point_cloud_anno(self, res, info):
        # for "NuScenesDataset" and "LyftDataset"
        # print("info[gt_boxes_velocity]", info["gt_boxes_velocity"])
        if "gt_boxes" in info:
            if self.with_velocity:
                res["annotations_lidar"] = {
                    "boxes": info["gt_boxes"].astype(np.float32),
                    "names": info["gt_names"],
                    "tokens": info["gt_boxes_token"],
                    "velocities": info["gt_boxes_velocity"].astype(np.float32),
                }#
            else:
                res["annotations_lidar"] = {
                    "boxes": info["gt_boxes"].astype(np.float32),
                    "names": info["gt_names"],
                    "tokens": info["gt_boxes_token"],
                }

            if self.training and not self.with_velocity:
                # res["annotations_lidar"]["boxes"] = np.delete(res["annotations_lidar"]["boxes"], np.s_[6:8], axis=1)
                res["annotations_lidar"]["velocities"] = [0, 0, 0]
                #info["gt_boxes_velocity"].astype(np.float32),
        # print("res[annotations_lidar]", res["annotations_lidar"])

        return res, info

    def get_pseudo_data_raw(self, index):

        token = self.pseudo_set[index]['metadata']['token']

        full_input_info = self.nusc_infos_token[token]

        res = {
            "points": None,
            "n_sweeps": self.n_sweeps,
            "annotations_lidar": None,
            # "ground_plane": -gp[-1] if with_gp else None,
            "image_prefix": self.root_path,
            "calib": None,
            "image": None,
            "mode": "train" if self.training else "val",
            "metadata": {
                "image_prefix": self.dataset_cfg.DATA_PATH,
                "num_point_features": self.num_point_features,
                "token": full_input_info["token"],
            },
            "pseudo_boxes": self.pseudo_set[index]["pseudo_boxes"],
            "pseudo_importance": self.pseudo_set[index]["pseudo_importance"],
            "pseudo_classes_filtered": self.pseudo_set[index]["pseudo_classes_filtered"],
            "pseudo_weights": self.pseudo_set[index]["pseudo_weights"]
        }


        res, full_input_info = self.load_point_cloud(res, full_input_info)
        res, full_input_info = self.load_point_cloud_anno(res, full_input_info)

        return res

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self.nusc_infos[0]:
            return None
        cls_range_map = config_factory(self.eval_version).serialize()['class_range']
        gt_annos = []
        for info in self.nusc_infos:
            gt_names = np.array(info["gt_names"])
            gt_boxes = info["gt_boxes"]
            mask = np.array([n != "ignore" for n in gt_names], dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            # det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = np.array([cls_range_map[n] for n in gt_names])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            N = int(np.sum(mask))
            gt_annos.append(
                {
                    "bbox": np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                    "alpha": np.full(N, -10),
                    "occluded": np.zeros(N),
                    "truncated": np.zeros(N),
                    "name": gt_names[mask],
                    "location": gt_boxes[mask][:, :3],
                    "dimensions": gt_boxes[mask][:, 3:6],
                    "rotation_y": gt_boxes[mask][:, 6],
                    "token": info["token"],
                }
            )
        return gt_annos

    def second_det_to_nusc_box(self, detection, use_velocity=True):
        box3d = detection["box3d_lidar"].detach().cpu().numpy()
        scores = detection["scores"].detach().cpu().numpy()
        labels = detection["label_preds"].detach().cpu().numpy()
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        box_list = []
        for i in range(box3d.shape[0]):
            quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
            if use_velocity:
                velocity = (*box3d[i, 6:8], 0.0)
            else:
                velocity = (0.0, 0.0, 0.0)
            box = Box(
                box3d[i, :3],
                box3d[i, 3:6],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity,
            )
            box_list.append(box)
        return box_list

    def lidar_nusc_box_to_global(self, nusc, boxes, sample_token):
        try:
            s_record = nusc.get("sample", sample_token)
            sample_data_token = s_record["data"]["LIDAR_TOP"]
        except:
            sample_data_token = sample_token

        sd_record = nusc.get("sample_data", sample_data_token)
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        sensor_record = nusc.get("sensor", cs_record["sensor_token"])
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

        data_path = nusc.get_sample_data_path(sample_data_token)
        box_list = []
        for box in boxes:
            # Move box to ego vehicle coord system
            box.rotate(Quaternion(cs_record["rotation"]))
            box.translate(np.array(cs_record["translation"]))
            # Move box to global coord system
            box.rotate(Quaternion(pose_record["rotation"]))
            box.translate(np.array(pose_record["translation"]))
            box_list.append(box)
        return box_list

    def eval_main(self, nusc, eval_version, res_path, eval_set, output_dir):
        # nusc = NuScenes(version=version, dataroot=str(root_path), verbose=True)
        cfg = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=cfg,
            result_path=res_path,
            eval_set=eval_set,
            output_dir=output_dir,
            verbose=True
        )
        metrics_summary = nusc_eval.main(plot_examples=5,)

    def evaluation(self, detections, class_names, output_dir=None, testset=False, eval_location=None, version=None):

        print("eval_location",eval_location)
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        if 'all' in eval_location:
            evaluation_set = eval_location.split('_all')[0]
        elif eval_location is not None:
            evaluation_set = F'{eval_location}_val'
        else:
            evaluation_set = eval_set_map[self.version]

        if not testset:
            dets = []
            gt_annos = self.ground_truth_annotations
            assert gt_annos is not None

            miss = 0
            for gt in gt_annos:
                try:
                    dets.append(detections[gt["token"]])
                except Exception:
                    miss += 1

            assert miss == 0
        else:
            dets = [v for _, v in detections.items()]
            # assert len(detections) == 6008

        # print("detections", dets)

        nusc_annos = {
            "results": {},
            "meta": None,
        }

        nusc = NuScenes(version=version, dataroot=str(self.root_path), verbose=True)

        mapped_class_names = []
        for n in self.class_names:
            if n in self.name_mapping:
                mapped_class_names.append(self.name_mapping[n])
            else:
                mapped_class_names.append(n)

        # print("mapped_class_names", mapped_class_names)


        for det in dets:
            annos = []
            boxes = self.second_det_to_nusc_box(det, use_velocity=self.with_velocity)
            boxes = self.lidar_nusc_box_to_global(nusc, boxes, det["metadata"]["token"])
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label-1]

                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = None
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = None

                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr
                    if attr is not None
                    else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                        0
                    ],
                }
                annos.append(nusc_anno)
            nusc_annos["results"].update({det["metadata"]["token"]: annos})

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        name = self.info_path.split("/")[-1].split(".")[0]
        res_path = str(Path(output_dir) / Path(name + ".json"))

        # print("nusc_annos", nusc_annos)

        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)


        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            self.eval_main(
                nusc,
                self.eval_version,
                res_path,
                evaluation_set,
                output_dir,
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            res_nusc = {
                "results": {"nusc": result},
                "detail": {"nusc": detail},
            }
        else:
            res_nusc = None

        if res_nusc is not None:
            res = {
                "results": {"nusc": res_nusc["results"]["nusc"],},
                "detail": {"eval.nusc": res_nusc["detail"]["nusc"],},
            }
        else:
            res = None

        return res, None

    def __len__(self):
        if self.pseudo:
            return len(self.pseudo_set)

        if not hasattr(self, "nusc_infos"):
            self.load_infos()

        return len(self.nusc_infos)

    def __getitem__(self, index):
        # index = 4
        if self.pseudo:
            input_dict = self.get_pseudo_data_raw(index)
            # print("pseudo index", index)
            # print("input_dict", input_dict)
        else:
            input_dict = self.get_sensor_data_raw(index)


        # print("input_dict points", input_dict["annotations_lidar"]["boxes"])
        # print("input_dict", input_dict)

        example = self.prepare_data_nusc(input_dict=input_dict, with_velo=self.with_velocity)
        # print('example', example)

        # print("get item", example)
        # gt_box: n*10
        # gt_importance: n 1
        # points" n*5
        # gt_classses n
        # voxels: n*5
        # voxel_coords: nn3?
        # voxel_num_points: nn 1?

        if not self.training:
            example["metadata"] = input_dict["metadata"]
        # if "anchors_mask" in example:
        #     example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        # for k,v in example.items():
        #     if type(v) is np.ndarray:
        #         print("k, v: ", k, v.shape)
        return example

    def _get_available_scenes(self, nusc, location=None):
        available_scenes = []
        print("total scene num:", len(nusc.scene))

        log_place_dict={}
        with open('/home/wzha8158/datasets/3D_Detection/Nuscenes/v1.0-trainval/log.json') as json_file:
            log_data = json.load(json_file)
            for p in log_data:
                log_place_dict[p["token"]]= p["location"]

        with open(F"/home/wzha8158/OpenLidarPerceptron/srdan_utils/nusc_scenes/nuscene_{location}_new_all.txt","r") as f: #_all
            file_scenes = ast.literal_eval(f.readlines()[0])
            for scene in nusc.scene:
                scene_token = scene["token"]
                log_token = scene['log_token']
                scene_rec = nusc.get('scene', scene_token)
                sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
                sd_rec = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
                has_more_frames = True
                scene_not_exist = False
                while has_more_frames:
                    lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
                    if not Path(lidar_path).exists():
                        scene_not_exist = True
                        break
                    else:
                        break
                    if not sd_rec['next'] == "":
                        sd_rec = nusc.get('sample_data', sd_rec['next'])
                    else:
                        has_more_frames = False
                if scene_not_exist:
                    continue

                if location is not None:
                    # print("log_token", log_token)
                    loc = log_place_dict[str(log_token)]
                    if 'half' in location:
                        if scene['name'] not in file_scenes:
                            continue
                    elif location not in loc:
                        continue
                available_scenes.append(scene)

                # f.write(F"'{scene['name']}', ")
        print("exist scene num:", len(available_scenes))
        return available_scenes

    def get_sample_data(self,
        nusc, sample_data_token: str, selected_anntokens: List[str] = None
    ) -> Tuple[str, List[Box], np.array]:
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param selected_anntokens: If provided only return the selected annotation.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """

        # Retrieve sensor & pose records
        sd_record = nusc.get("sample_data", sample_data_token)
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        sensor_record = nusc.get("sensor", cs_record["sensor_token"])
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

        data_path = nusc.get_sample_data_path(sample_data_token)

        if sensor_record["modality"] == "camera":
            cam_intrinsic = np.array(cs_record["camera_intrinsic"])
            imsize = (sd_record["width"], sd_record["height"])
        else:
            cam_intrinsic = None
            imsize = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(nusc.get_box, selected_anntokens))
        else:
            boxes = nusc.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:

            # Move box to ego vehicle coord system
            box.translate(-np.array(pose_record["translation"]))
            box.rotate(Quaternion(pose_record["rotation"]).inverse)

            #  Move box to sensor coord system
            box.translate(-np.array(cs_record["translation"]))
            box.rotate(Quaternion(cs_record["rotation"]).inverse)

            box_list.append(box)

        return data_path, box_list, cam_intrinsic


    def _fill_trainval_infos(self, nusc, train_scenes, val_scenes, test=False, nsweeps=10):
        from nuscenes.utils.geometry_utils import transform_matrix

        train_nusc_infos = []
        val_nusc_infos = []
        all_nusc_infos = []
        other_nusc_infos = []

        ref_chan = "LIDAR_TOP"  # The radar channel from which we track back n sweeps to aggregate the point cloud.
        chan = "LIDAR_TOP"  # The reference channel of the current sample_rec that the point clouds are mapped to.

        for sample in tqdm(nusc.sample):
            """ Manual save info["sweeps"] """
            # Get reference pose and timestamp
            # ref_chan == "LIDAR_TOP"
            ref_sd_token = sample["data"][ref_chan]
            ref_sd_rec = nusc.get("sample_data", ref_sd_token)
            ref_cs_rec = nusc.get(
                "calibrated_sensor", ref_sd_rec["calibrated_sensor_token"]
            )
            ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
            ref_time = 1e-6 * ref_sd_rec["timestamp"]

            ref_lidar_path, ref_boxes, _ = self.get_sample_data(nusc, ref_sd_token)

            ref_cam_front_token = sample["data"]["CAM_FRONT"]
            ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

            # Homogeneous transform from ego car frame to reference frame
            ref_from_car = transform_matrix(
                ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
            )

            # Homogeneous transformation matrix from global to _current_ ego car frame
            car_from_global = transform_matrix(
                ref_pose_rec["translation"],
                Quaternion(ref_pose_rec["rotation"]),
                inverse=True,
            )

            info = {
                "lidar_path": ref_lidar_path,
                "cam_front_path": ref_cam_path,
                "cam_intrinsic": ref_cam_intrinsic,
                "token": sample["token"],
                "sweeps": [],
                "ref_from_car": ref_from_car,
                "car_from_global": car_from_global,
                "timestamp": ref_time,
            }

            sample_data_token = sample["data"][chan]
            curr_sd_rec = nusc.get("sample_data", sample_data_token)
            sweeps = []
            while len(sweeps) < nsweeps - 1:
                if curr_sd_rec["prev"] == "":
                    if len(sweeps) == 0:
                        sweep = {
                            "lidar_path": ref_lidar_path,
                            "sample_data_token": curr_sd_rec["token"],
                            "transform_matrix": None,
                            "time_lag": curr_sd_rec["timestamp"] * 0,
                            # time_lag: 0,
                        }
                        sweeps.append(sweep)
                    else:
                        sweeps.append(sweeps[-1])
                else:
                    curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])

                    # Get past pose
                    current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
                    global_from_car = transform_matrix(
                        current_pose_rec["translation"],
                        Quaternion(current_pose_rec["rotation"]),
                        inverse=False,
                    )

                    # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                    current_cs_rec = nusc.get(
                        "calibrated_sensor", curr_sd_rec["calibrated_sensor_token"]
                    )
                    car_from_current = transform_matrix(
                        current_cs_rec["translation"],
                        Quaternion(current_cs_rec["rotation"]),
                        inverse=False,
                    )

                    tm = reduce(
                        np.dot,
                        [ref_from_car, car_from_global, global_from_car, car_from_current],
                    )

                    lidar_path = nusc.get_sample_data_path(curr_sd_rec["token"])

                    time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]

                    sweep = {
                        "lidar_path": lidar_path,
                        "sample_data_token": curr_sd_rec["token"],
                        "transform_matrix": tm,
                        "global_from_car": global_from_car,
                        "car_from_current": car_from_current,
                        "time_lag": time_lag,
                    }
                    sweeps.append(sweep)

            info["sweeps"] = sweeps

            assert (
                len(info["sweeps"]) == nsweeps - 1
            ), f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, you should duplicate to sweep num {nsweeps-1}"
            """ read from api """
            if not test:
                annotations = [
                    nusc.get("sample_annotation", token) for token in sample["anns"]
                ]

                # the filtering gives 0.5~1 map improvement
                mask = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts'])>0 for anno in annotations], dtype=bool).reshape(-1)

                locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)
                # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
                if self.with_velocity:
                    velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)


                rots = np.array([self.quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(
                    -1, 1
                )
                names = np.array([b.name for b in ref_boxes])
                tokens = np.array([b.token for b in ref_boxes])
                if self.with_velocity:
                    gt_boxes = np.concatenate(
                        [locs, dims, velocity[:, :2], -rots - np.pi / 2], axis=1
                    )
                else:
                    gt_boxes = np.concatenate(
                        [locs, dims, -rots - np.pi / 2], axis=1
                    )

                # gt_boxes = np.concatenate([locs, dims, rots], axis=1)

                assert len(annotations) == len(gt_boxes) #== len(velocity)

                info["gt_boxes"] = gt_boxes[mask, :]
                if self.with_velocity:
                    info["gt_boxes_velocity"] = velocity[mask, :]
                info["gt_names"] = np.array([general_to_detection[name] for name in names])[mask]
                info["gt_boxes_token"] = tokens[mask]

            if sample["scene_token"] in train_scenes:
                train_nusc_infos.append(info)
                all_nusc_infos.append(info)
            elif sample["scene_token"] in val_scenes:
                val_nusc_infos.append(info)
                all_nusc_infos.append(info)
            else:
                other_nusc_infos.append(info)

        print(len(train_nusc_infos), len(val_nusc_infos))

        return train_nusc_infos, val_nusc_infos, all_nusc_infos

    def create_nuscenes_infos_new(self, root_path, version="v1.0-trainval", nsweeps=10, location=None):
        nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
        available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
        # if location is not None:
        #     if 'mini' in location:
        #         version = 'v1.0-mini'

        assert version in available_vers
        if version == "v1.0-trainval":
            train_scenes = splits.train
            # random.shuffle(train_scenes)
            # train_scenes = train_scenes[:int(len(train_scenes)*0.2)]
            val_scenes = splits.val
        elif version == "v1.0-test":
            train_scenes = splits.test
            val_scenes = []
        elif version == "v1.0-mini":
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise ValueError("unknown")
        test = "test" in version
        root_path = Path(root_path)
        # filter exist scenes. you may only download part of dataset.
        if location is not None:
            print("location",location)
            available_scenes = self._get_available_scenes(nusc, location=location)
        else:
            available_scenes = self._get_available_scenes(nusc)
        available_scene_names = [s["name"] for s in available_scenes]
        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set(
            [
                available_scenes[available_scene_names.index(s)]["token"]
                for s in train_scenes
            ]
        )
        val_scenes = set(
            [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
        )


        if test:
            # print(f"test scene: {train_scenes}")
            print(f"test scene: {len(train_scenes)}")
        else:
            # print(f"train scene: {train_scenes}, val scene: {val_scenes}")
            print(f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

        train_nusc_infos, val_nusc_infos, all_nusc_infos = self._fill_trainval_infos(
            nusc, train_scenes, val_scenes, test, nsweeps=nsweeps
        )

        if location is None:
            prefix = ''
        else:
            prefix = f'{location}_'

        if test:
            print(f"test sample: {len(train_nusc_infos)}")
            with open(
                root_path / f"{prefix}infos_test_{nsweeps:02d}sweeps_{self.with_velocity}velo_PCDet.pkl", "wb"
            ) as f:
                pickle.dump(train_nusc_infos, f)
        else:
            print(
                f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}, all sample: {len(all_nusc_infos)}"
            )
            with open(
                root_path / f"{prefix}infos_train_{nsweeps:02d}sweeps_{self.with_velocity}velo_PCDet.pkl", "wb"
            ) as f:
                pickle.dump(train_nusc_infos, f)
            with open(
                root_path / f"{prefix}infos_val_{nsweeps:02d}sweeps_{self.with_velocity}velo_PCDet.pkl", "wb"
            ) as f:
                pickle.dump(val_nusc_infos, f)
            with open(
                root_path / f"{prefix}infos_all_{nsweeps:02d}sweeps_{self.with_velocity}velo_PCDet.pkl", "wb"
            ) as f:
                pickle.dump(all_nusc_infos, f)

    def quaternion_yaw(self, q: Quaternion) -> float:
        """
        Calculate the yaw angle from a quaternion.
        Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
        It does not work for a box in the camera frame.
        :param q: Quaternion of interest.
        :return: Yaw angle in radians.
        """

        # Project into xy plane.
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

        # Measure yaw using arctan.
        yaw = np.arctan2(v[1], v[0])

        return yaw

    def create_groundtruth_database(self,
                                data_path,
                                info_path=None,
                                used_classes=None,
                                db_path=None,
                                dbinfo_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None,
                                location=None,
                                **kwargs,
                                ):

        root_path = Path(data_path)
        nsweeps = self.n_sweeps

        if db_path is None:
            db_path = root_path / f"gt_database_{nsweeps}sweeps_{self.with_velocity}velo_PCDet"
        if dbinfo_path is None:
            if location is None:
                dbinfo_path = root_path / f"dbinfos_train_{nsweeps}sweeps_{self.with_velocity}velo_PCDet.pkl"
            else:
                dbinfo_path = root_path / f"{location}_dbinfos_train_{nsweeps}sweeps_{self.with_velocity}velo_PCDet.pkl"

        point_features = 5

        db_path.mkdir(parents=True, exist_ok=True)

        all_db_infos = {}
        group_counter = 0

        # def prepare_single_data(index):
        for index in tqdm(range(int(self.__len__()))):
            image_idx = index
            # modified to nuscenes
            sensor_data = self.get_sensor_data_raw(index)
            # for nsweep, sensor_data in enumerate(sensor_datas):
            if "image_idx" in sensor_data["metadata"]:
                image_idx = sensor_data["metadata"]["image_idx"]

            points = sensor_data["points_combined"]

            annos = sensor_data["annotations_lidar"]
            gt_boxes = annos["boxes"]
            names = annos["names"]
            group_dict = {}
            group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
            if "group_ids" in annos:
                group_ids = annos["group_ids"]
            else:
                group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
            difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
            if "difficulty" in annos:
                difficulty = annos["difficulty"]

            num_obj = gt_boxes.shape[0]
            point_indices = box_utils.points_in_rbbox(points, gt_boxes)
            for i in range(num_obj):
                filename = f"{image_idx}_{names[i]}_{i}.bin"
                filepath = db_path / filename
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    gt_points[:, :point_features].tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    if relative_path:
                        db_dump_path = str(db_path.stem + "/" + filename)
                    else:
                        db_dump_path = str(filepath)

                    db_info = {
                        "name": names[i],
                        "path": db_dump_path,
                        "image_idx": image_idx,
                        "gt_idx": i,
                        "box3d_lidar": gt_boxes[i],
                        "num_points_in_gt": gt_points.shape[0],
                        "difficulty": difficulty[i],
                        # "group_id": -1,
                        # "bbox": bboxes[i],
                    }
                    local_group_id = group_ids[i]
                    # if local_group_id >= 0:
                    if local_group_id not in group_dict:
                        group_dict[local_group_id] = group_counter
                        group_counter += 1
                    db_info["group_id"] = group_dict[local_group_id]
                    if "score" in annos:
                        db_info["score"] = annos["score"][i]
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
            # print(f"Finish {index}th sample")

        print("self length: ", self.__len__())
        for k, v in all_db_infos.items():
            print(f"load {len(v)} {k} database infos")

        with open(dbinfo_path, "wb") as f:
            pickle.dump(all_db_infos, f)

if __name__ == '__main__':
    if sys.argv.__len__() > 1:
        root_path = Path('/home/wzha8158/datasets/3D_Detection/Nuscenes')
        version = "v1.0-trainval"
        max_sweeps = 10
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        if sys.argv.__len__() > 2:
            loc = sys.argv[2]
            add_loc = '_'+loc
            # loc_full = '_'+loc.split('_')[-1]
            print(loc)#, loc_full)
        else:
            loc = None
            add_loc = ''
        dataset_cfg = EasyDict(yaml.load(open(f'/home/wzha8158/Dropbox/CVPR2021_LidarPerceptron/Home/OpenLidarPerceptron/tools/cfgs/dataset_configs/nuscenes_dataset'+add_loc+'_novelo.yaml')))#boston
        class_names=['car', 'pedestrian', 'cyclist']
        log_file = 'debug_loger.txt'
        logger = common_utils.create_logger(log_file, rank=0)
        if sys.argv[1] == 'create_nuscenes_infos':
            A = NuscenesDataset(dataset_cfg=dataset_cfg, root_path=Path(dataset_cfg.DATA_PATH), class_names=dataset_cfg.CLASS_NAMES, n_sweeps=dataset_cfg.N_SWEEPS, info_path=Path(dataset_cfg.INFO_PATH), training=True, logger=logger)

            # A.create_nuscenes_infos_new(root_path='/home/wzha8158/datasets/3D_Detection/Nuscenes/', location=loc) #, version='v1.0-mini'

        if sys.argv[1] == 'create_nuscenes_dbinfos':
            A = NuscenesDataset(dataset_cfg=dataset_cfg, root_path=Path(dataset_cfg.DATA_PATH), class_names=dataset_cfg.CLASS_NAMES, n_sweeps=dataset_cfg.N_SWEEPS, info_path=Path(dataset_cfg.INFO_PATH), training=True, logger=logger)

            # A.create_groundtruth_database(dataset_cfg.DATA_PATH, info_path=dataset_cfg.INFO_PATH, used_classes=None, db_path=None, dbinfo_path=None,  location=loc)

    else:
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.load(open('/home/wzha8158/Dropbox/CVPR2021_LidarPerceptron/Home/OpenLidarPerceptron/tools/cfgs/dataset_configs/nuscenes_dataset.yaml')))
        # print("dataset_cfg", dataset_cfg)
        log_file = 'debug_loger.txt'
        logger = common_utils.create_logger(log_file, rank=0)

        A = NuscenesDataset(dataset_cfg=dataset_cfg, root_path=Path(dataset_cfg.DATA_PATH), class_names=dataset_cfg.CLASS_NAMES, n_sweeps=dataset_cfg.N_SWEEPS, info_path=Path(dataset_cfg.INFO_PATH), training=True, logger=logger)

        # for i in range(int(A.__len__())):
        #     A[i]
        # print(A[0])
        # A[1]
        import pdb
        pdb.set_trace()

        # class_names=['barrier','bicycle', 'bus', 'car', 'construction_vehicle','motorcycle','pedestrian', 'traffic_cone', 'trailer', 'truck']
        #class_names=NuscenesDataset.NameMapping.keys()#['Car', 'Pedestrian', 'Cyclist']
        # print("class_names",class_names)

        # if sys.argv[1] == 'create_nuscenes_infos':
        #     # create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps)
        #     info_path = "singapore_infos_train.pkl"
        #     nusc_ds = NuscenesDataset(dataset_cfg=dataset_cfg, root_path=Path('/home/wzha8158/datasets/3D_Detection/Nuscenes'),
        #                                class_names=class_names, info_path=Path(root_path) / info_path)
        #     # nusc_ds.create_groundtruth_database()

        # elif sys.argv[1] == 'create_nuscenes_boston_infos':
        #     #create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps, location="boston")
        #     location="boston"
        #     info_path = "pcdet_boston_infos_train.pkl"

        #     nusc_ds = NuscenesDataset(dataset_cfg=dataset_cfg, root_path=Path('/home/wzha8158/datasets/3D_Detection/Nuscenes'),
        #                                class_names=class_names, info_path=Path(root_path) / info_path)
        #     # nusc_ds.create_groundtruth_database(location="boston")

        # elif sys.argv[1] == 'create_nuscenes_singapore_infos':
        #     #create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps, location="singapore")
        #     location='singapore'
        #     info_path = "pcdet_singapore_infos_train.pkl"

        #     nusc_ds = NuscenesDataset(dataset_cfg=dataset_cfg, root_path=Path('/home/wzha8158/datasets/3D_Detection/Nuscenes'),
        #                                class_names=class_names, info_path=Path(root_path) / info_path)
        #     # nusc_ds.create_groundtruth_database(location="singapore")





        # return self.get_sensor_data(idx)
        # idx = query
        # read_test_image = False
        # if isinstance(query, dict):
        #     assert "lidar" in query
        #     idx = query["lidar"]["idx"]
        #     read_test_image = "cam" in query

        # points = self.get_lidar(sample_idx)
        # calib = self.get_calib(sample_idx)

        # # if cfg.DATA_CONFIG.FOV_POINTS_ONLY:
        # #     pts_rect = calib.lidar_to_rect(points[:, 0:3])
        # #     fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
        # #     points = points[fov_flag]

        # info = self.nusc_infos[idx]
        # res = {
        #     "points": points,
        #     "metadata": {
        #         "token": info["token"]
        #     },
        #     "sample_idx": idx,
        #     "calib": calib,
        # }

        # if read_test_image:
        #     res["image"] = self.get_image(idx)
        #     res["img_shape"] = self.get_image_shape(idx)

        # if 'gt_boxes' in info:
        #     # gt_names = info["gt_names"]
        #     gt_boxes = info["gt_boxes"]
        #     num_lidar_pts = info["num_lidar_pts"]
        #     mask = num_lidar_pts > 0

        #     gt_names = gt_names[mask]
        #     gt_boxes = gt_boxes[mask]

        #     if self.with_velocity:
        #         gt_velocity = info["gt_velocity"][mask]
        #         nan_mask = np.isnan(gt_velocity[:, 0])
        #         gt_velocity[nan_mask] = [0.0, 0.0]
        #         gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=-1)
        #     res["lidar"]["annotations"] = {
        #         'boxes': gt_boxes,
        #         'names': info["gt_names"][mask],
        #     }
        # return res


    # def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
    #     import concurrent.futures as futures

    #     # def process_single_scene(sample_idx):
    #     #     print('%s sample_idx: %s' % (self.split, sample_idx))
    #     #     temp_info = self.nusc_infos[sample_idx]
    #     #     info = {}
    #     #     pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
    #     #     info['point_cloud'] = pc_info
    #     #     # CAM_FRONT
    #     #     # image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
    #     #     # info['image'] = image_info
    #     #     # calib = self.get_calib(sample_idx)

    #     #     # P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
    #     #     # R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
    #     #     # R0_4x4[3, 3] = 1.
    #     #     # R0_4x4[:3, :3] = calib.R0
    #     #     # V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
    #     #     # calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

    #     #     # info['calib'] = calib_info
    #     #     gt_annos = []

    #     #     if has_label:
    #     #         # obj_list = self.get_label(sample_idx)
    #     #         gt_names = temp_info["gt_names"]
    #     #         gt_boxes = temp_info["gt_boxes"]
    #     #         num_lidar_pts = temp_info["num_lidar_pts"]
    #     #         mask = num_lidar_pts > 0
    #     #         gt_names = gt_names[mask]
    #     #         gt_boxes = gt_boxes[mask]
    #     #         num_lidar_pts = num_lidar_pts[mask]

    #     #         mask = np.array([n in self.kitti_name_mapping for n in gt_names],
    #     #                         dtype=np.bool_)
    #     #         gt_names = gt_names[mask]
    #     #         gt_boxes = gt_boxes[mask]
    #     #         num_lidar_pts = num_lidar_pts[mask]
    #     #         gt_names_mapped = [self.kitti_name_mapping[n] for n in gt_names]
    #     #         det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
    #     #         det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
    #     #         mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
    #     #         mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
    #     #         gt_names = gt_names[mask]
    #     #         gt_boxes = gt_boxes[mask]
    #     #         num_lidar_pts = num_lidar_pts[mask]
    #     #         # use occluded to control easy/moderate/hard in kitti
    #     #         easy_mask = num_lidar_pts > 15
    #     #         moderate_mask = num_lidar_pts > 7
    #     #         occluded = np.zeros([num_lidar_pts.shape[0]])
    #     #         occluded[:] = 2
    #     #         occluded[moderate_mask] = 1
    #     #         occluded[easy_mask] = 0
    #     #         N = len(gt_boxes)

    #     #         annotations = {}
    #     #         annotations['name'] = gt_names
    #     #         annotations['truncated'] = np.zeros(N)
    #     #         annotations['occluded'] = occluded
    #     #         annotations['alpha'] = np.full(N, -10)
    #     #         annotations['bbox'] = np.tile(np.array([[0, 0, 50, 50]]), [N, 1])
    #     #         annotations['dimensions'] = gt_boxes[:, 3:6]  # lhw(camera) format
    #     #         annotations['location'] = gt_boxes[:, :3]
    #     #         annotations['rotation_y'] = gt_boxes[:, 6]
    #     #         annotations['difficulty'] = temp_info['difficulty']#???

    #     #         # annotations['score'] = np.array([obj.score for obj in obj_list])
    #     #         # annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

    #     #         # num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
    #     #         # num_gt = len(annotations['name'])
    #     #         # index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    #     #         # annotations['index'] = np.array(index, dtype=np.int32)

    #     #         # gt box lidar !!!!

    #     #         # loc = annotations['location'][:num_objects]
    #     #         # dims = annotations['dimensions'][:num_objects]
    #     #         # rots = annotations['rotation_y'][:num_objects]
    #     #         # loc_lidar = calib.rect_to_lidar(loc)
    #     #         # l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    #     #         # gt_boxes_lidar = np.concatenate([loc_lidar, w, l, h, rots[..., np.newaxis]], axis=1)
    #     #         # annotations['gt_boxes_lidar'] = gt_boxes_lidar

    #     #         info['annos'] = annotations

    #     #         # count points!!!!

    #     #         # if count_inside_pts:
    #     #         #     points = self.get_lidar(sample_idx)
    #     #         #     calib = self.get_calib(sample_idx)
    #     #         #     pts_rect = calib.lidar_to_rect(points[:, 0:3])

    #     #         #     fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
    #     #         #     pts_fov = points[fov_flag]
    #     #         #     corners_lidar = box_utils.boxes3d_to_corners3d_lidar(gt_boxes_lidar)
    #     #         #     num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

    #     #         #     for k in range(num_objects):
    #     #         #         flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
    #     #         #         num_points_in_gt[k] = flag.sum()
    #     #         #     annotations['num_points_in_gt'] = num_points_in_gt

    #     #     return info


    #     # print("_nusc_infos", self.nusc_infos)
    #     # temp = process_single_scene(self.sample_id_list[0])
    #     sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

    #     print("sample_idx_list", sample_id_list)
    #     # print("sample_idxlist self", self.sample_id_list)

    #     with futures.ThreadPoolExecutor(num_workers) as executor:
    #         infos = executor.map(get_sensor_data_raw, sample_id_list)
    #     return list(infos)

    # def create_groundtruth_database(self,
    #                             used_classes=None,
    #                             database_save_path=None,
    #                             db_info_save_path=None,
    #                             relative_path=True,
    #                             add_rgb=False,
    #                             lidar_only=False,
    #                             bev_only=False,
    #                             coors_range=None,location=None,day=False,night=False,dense=False):

    #     root_path = self.root_path
    #     if database_save_path is None:
    #         database_save_path = root_path / 'pcdet_gt_database'
    #     else:
    #         database_save_path = Path(database_save_path)
    #     if db_info_save_path is None:
    #         if location is not None:
    #             db_info_save_path = root_path / F"pcdet_{location}_dbinfos_train.pkl"
    #         elif day:
    #             db_info_save_path = root_path / "pcdet_day_dbinfos_train.pkl"
    #         elif night:
    #             db_info_save_path = root_path / "pcdet_night_dbinfos_train.pkl"
    #         elif dense:
    #             db_info_save_path = root_path / "pcdet_dense_dbinfos_train.pkl"
    #         else:
    #             db_info_save_path = root_path / "pcdet_dbinfos_train.pkl"

    #     database_save_path.mkdir(parents=True, exist_ok=True)
    #     all_db_infos = {}

    #     # with open(self.info_path, 'rb') as f:
    #     #     infos = pickle.load(f)

    #     # print("infos", infos)
    #     ################
    #     print('######### generate all db info ########')
    #     group_counter = 0
    #     for j in list(range(len(self.nusc_infos))):
    #         image_idx = j
    #         sensor_data = self.get_sensor_data(j)
    #         if "image_idx" in sensor_data["metadata"]:
    #             image_idx = sensor_data["metadata"]["image_idx"]
    #         points = sensor_data["lidar"]["points"]
    #         annos = sensor_data["lidar"]["annotations"]
    #         gt_boxes = annos["boxes"]
    #         names = annos["names"]
    #         group_dict = {}
    #         group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
    #         if "group_ids" in annos:
    #             group_ids = annos["group_ids"]
    #         else:
    #             group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
    #         difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    #         if "difficulty" in annos:
    #             difficulty = annos["difficulty"]

    #         num_obj = gt_boxes.shape[0]
    #         # print("num_obj", num_obj)
    #         point_indices = box_utils.points_in_rbbox(points, gt_boxes)
    #         for i in range(num_obj):
    #             filename = f"{image_idx}_{names[i]}_{i}.bin"
    #             filepath = database_save_path / filename
    #             gt_points = points[point_indices[:, i]]

    #             gt_points[:, :3] -= gt_boxes[i, :3]
    #             with open(filepath, 'w') as f:
    #                 gt_points.tofile(f)

    #             if used_classes is None:
    #                 cond = True
    #             else:
    #                 cond = names[i] in used_classes

    #             if cond:
    #                 if relative_path:
    #                     db_path = str(database_save_path.stem + "/" + filename)
    #                 else:
    #                     db_path = str(filepath)
    #                 db_info = {
    #                     "name": names[i],
    #                     "path": db_path,
    #                     "image_idx": image_idx,
    #                     "gt_idx": i,
    #                     "box3d_lidar": gt_boxes[i],
    #                     "num_points_in_gt": gt_points.shape[0],
    #                     "difficulty": difficulty[i],
    #                     # "group_id": -1,
    #                     # "bbox": bboxes[i],
    #                 }
    #                 local_group_id = group_ids[i]
    #                 # if local_group_id >= 0:
    #                 if local_group_id not in group_dict:
    #                     group_dict[local_group_id] = group_counter
    #                     group_counter += 1
    #                 db_info["group_id"] = group_dict[local_group_id]
    #                 if "score" in annos:
    #                     db_info["score"] = annos["score"][i]
    #                 if names[i] in all_db_infos:
    #                     all_db_infos[names[i]].append(db_info)
    #                 else:
    #                     all_db_infos[names[i]] = [db_info]
    #     for k, v in all_db_infos.items():
    #         print(f"load {len(v)} {k} database infos")

    #     with open(db_info_save_path, 'wb') as f:
    #         pickle.dump(all_db_infos, f)

    #     print('---------------Start create groundtruth database for data augmentation---------------')


    # @staticmethod
    # def generate_prediction_dict(batch_dict, pred_dicts, class_names, output_path=None):
        # # finally generate predictions.
        # sample_idx = input_dict['sample_idx'][index] if 'sample_idx' in input_dict else -1
        # boxes3d_lidar_preds = record_dict['boxes'].cpu().numpy()

        # if boxes3d_lidar_preds.shape[0] == 0:
        #     return {'sample_idx': sample_idx}

        # calib = input_dict['calib'][index]
        # image_shape = input_dict['image_shape'][index]

        # boxes3d_camera_preds = box_utils.boxes3d_lidar_to_camera(boxes3d_lidar_preds, calib)
        # boxes2d_image_preds = box_utils.boxes3d_camera_to_imageboxes(boxes3d_camera_preds, calib, image_shape=image_shape)
        # # predictions
        # predictions_dict = {
        #     'bbox': boxes2d_image_preds,
        #     'box3d_camera': boxes3d_camera_preds,
        #     'box3d_lidar': boxes3d_lidar_preds,
        #     'scores': record_dict['scores'].cpu().numpy(),
        #     'label_preds': record_dict['labels'].cpu().numpy(),
        #     'sample_idx': sample_idx,
        # }
        # return predictions_dict

    # @staticmethod
    # def generate_prediction_dicts(input_dict, pred_dicts, class_names, save_to_file=False, output_dir=None):
    #     def get_empty_prediction():
    #         ret_dict = {
    #             'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]),
    #             'alpha': np.array([]), 'bbox': np.zeros([0, 5]), 'dimensions': np.zeros([0, 3]),
    #             'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([]),
    #             'boxes_lidar': np.zeros([0, 7])
    #         }
    #         return ret_dict

    #     def generate_single_anno(idx, box_dict):
    #         num_example = 0
    #         if 'bbox' not in box_dict:
    #             return get_empty_prediction(), num_example

    #         area_limit = image_shape = None
    #         if cfg.MODEL.TEST.BOX_FILTER['USE_IMAGE_AREA_FILTER']:
    #             image_shape = input_dict['image_shape'][idx]
    #             area_limit = image_shape[0] * image_shape[1] * 0.8

    #         sample_idx = box_dict['sample_idx']
    #         box_preds_image = box_dict['bbox']
    #         box_preds_camera = box_dict['box3d_camera']
    #         box_preds_lidar = box_dict['box3d_lidar']
    #         scores = box_dict['scores']
    #         label_preds = box_dict['label_preds']

    #         anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [],
    #                 'location': [], 'rotation_y': [], 'score': [], 'boxes_lidar': []}

    #         for box_camera, box_lidar, bbox, score, label in zip(box_preds_camera, box_preds_lidar, box_preds_image,
    #                                                              scores, label_preds):
    #             if area_limit is not None:
    #                 if bbox[0] > image_shape[1] or bbox[1] > image_shape[0] or bbox[2] < 0 or bbox[3] < 0:
    #                     continue
    #                 bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
    #                 bbox[:2] = np.maximum(bbox[:2], [0, 0])
    #                 area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    #                 if area > area_limit:
    #                     continue

    #             if 'LIMIT_RANGE' in cfg.MODEL.TEST.BOX_FILTER:
    #                 limit_range = np.array(cfg.MODEL.TEST.BOX_FILTER['LIMIT_RANGE'])
    #                 if np.any(box_lidar[:3] < limit_range[:3]) or np.any(box_lidar[:3] > limit_range[3:]):
    #                     continue

    #             if not (np.all(box_lidar[3:6] > -0.1)):
    #                 print('Invalid size(sample %s): ' % str(sample_idx), box_lidar)
    #                 continue

    #             anno['name'].append(class_names[int(label - 1)])
    #             anno['truncated'].append(0.0)
    #             anno['occluded'].append(0)
    #             anno['alpha'].append(-np.arctan2(-box_lidar[1], box_lidar[0]) + box_camera[6])
    #             anno['bbox'].append(bbox)
    #             anno['dimensions'].append(box_camera[3:6])
    #             anno['location'].append(box_camera[:3])
    #             anno['rotation_y'].append(box_camera[6])
    #             anno['score'].append(score)
    #             anno['boxes_lidar'].append(box_lidar)

    #             num_example += 1

    #         if num_example != 0:
    #             anno = {k: np.stack(v) for k, v in anno.items()}
    #         else:
    #             anno = get_empty_prediction()

    #         return anno, num_example

    #     annos = []
    #     for i, box_dict in enumerate(pred_dicts):
    #         sample_idx = box_dict['sample_idx']
    #         single_anno, num_example = generate_single_anno(i, box_dict)
    #         single_anno['num_example'] = num_example
    #         single_anno['sample_idx'] = np.array([sample_idx] * num_example, dtype=np.int64)
    #         annos.append(single_anno)
    #         if save_to_file:
    #             cur_det_file = os.path.join(output_dir, '%s.txt' % sample_idx)
    #             with open(cur_det_file, 'w') as f:
    #                 bbox = single_anno['bbox']
    #                 loc = single_anno['location']
    #                 dims = single_anno['dimensions']  # lhw -> hwl

    #                 for idx in range(len(bbox)):
    #                     print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
    #                           % (single_anno['name'][idx], single_anno['alpha'][idx], bbox[idx][0], bbox[idx][1],
    #                              bbox[idx][2], bbox[idx][3], dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
    #                              loc[idx][1], loc[idx][2], single_anno['rotation_y'][idx], single_anno['score'][idx]),
    #                           file=f)

    #     return annos


    # def evaluation_nusc(self, detections, output_dir):
    #     version = self.version
    #     eval_set_map = {
    #         "v1.0-mini": "mini_train",
    #         "v1.0-trainval": "val",
    #     }
    #     gt_annos = self.ground_truth_annotations
    #     if gt_annos is None:
    #         return None
    #     nusc_annos = {}
    #     mapped_class_names = self.class_names
    #     token2info = {}
    #     for info in self.nusc_infos:
    #         token2info[info["token"]] = info
    #     for det in detections:
    #         annos = []
    #         boxes = _second_det_to_nusc_box(det)
    #         for i, box in enumerate(boxes):
    #             name = mapped_class_names[box.label]
    #             velocity = box.velocity[:2].tolist()
    #             if len(token2info[det["metadata"]["token"]]["sweeps"]) == 0:
    #                 velocity = (np.nan, np.nan)
    #             box.velocity = np.array([*velocity, 0.0])
    #         boxes = _lidar_nusc_box_to_global(
    #             token2info[det["metadata"]["token"]], boxes,
    #             mapped_class_names, "cvpr_2019")
    #         for i, box in enumerate(boxes):
    #             name = mapped_class_names[box.label]
    #             velocity = box.velocity[:2].tolist()
    #             nusc_anno = {
    #                 "sample_token": det["metadata"]["token"],
    #                 "translation": box.center.tolist(),
    #                 "size": box.wlh.tolist(),
    #                 "rotation": box.orientation.elements.tolist(),
    #                 "velocity": velocity,
    #                 "detection_name": name,
    #                 "detection_score": box.score,
    #                 "attribute_name": NuScenesDataset.DefaultAttribute[name],
    #             }
    #             annos.append(nusc_anno)
    #         nusc_annos[det["metadata"]["token"]] = annos
    #     nusc_submissions = {
    #         "meta": {
    #             "use_camera": False,
    #             "use_lidar": False,
    #             "use_radar": False,
    #             "use_map": False,
    #             "use_external": False,
    #         },
    #         "results": nusc_annos,
    #     }
    #     res_path = Path(output_dir) / "results_nusc.json"
    #     with open(res_path, "w") as f:
    #         json.dump(nusc_submissions, f)
    #     eval_main_file = Path(__file__).resolve().parent / "nusc_eval.py"
    #     # why add \"{}\"? to support path with spaces.
    #     cmd = f"python {str(eval_main_file)} --root_path=\"{str(self.root_path)}\""
    #     cmd += f" --version={self.version} --eval_version={self.eval_version}"
    #     cmd += f" --res_path=\"{str(res_path)}\" --eval_set={eval_set_map[self.version]}"
    #     cmd += f" --output_dir=\"{output_dir}\""
    #     # use subprocess can release all nusc memory after evaluation
    #     subprocess.check_output(cmd, shell=True)
    #     with open(Path(output_dir) / "metrics_summary.json", "r") as f:
    #         metrics = json.load(f)
    #     detail = {}
    #     res_path.unlink()  # delete results_nusc.json since it's very large
    #     result = f"Nusc {version} Evaluation\n"
    #     for name in mapped_class_names:
    #         detail[name] = {}
    #         for k, v in metrics["label_aps"][name].items():
    #             detail[name][f"dist@{k}"] = v
    #         tp_errs = []
    #         tp_names = []
    #         for k, v in metrics["label_tp_errors"][name].items():
    #             detail[name][k] = v
    #             tp_errs.append(f"{v:.4f}")
    #             tp_names.append(k)
    #         threshs = ', '.join(list(metrics["label_aps"][name].keys()))
    #         scores = list(metrics["label_aps"][name].values())
    #         scores = ', '.join([f"{s * 100:.2f}" for s in scores])
    #         result += f"{name} Nusc dist AP@{threshs} and TP errors\n"
    #         result += scores
    #         result += "\n"
    #         result += ', '.join(tp_names) + ": " + ', '.join(tp_errs)
    #         result += "\n"
    #     return {
    #         "results": {
    #             "nusc": result
    #         },
    #         "detail": {
    #             "nusc": detail
    #         },
    #     }

    # def evaluation(self, det_annos, output_dir, location, **kwargs):
    #     # assert 'annos' in self.kitti_infos[0].keys()
    #     # import pcdet.datasets.kitti.kitti_object_eval_python.eval as kitti_eval

    #     # if 'annos' not in self.kitti_infos[0]:
    #     #     return 'None', {}

    #     # eval_det_annos = copy.deepcopy(det_annos)
    #     # eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
    #     # ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

    #     # return ap_result_str, ap_dict

    #     """kitti evaluation is very slow, remove it.
    #     """
    #     res_nusc = self.evaluation_nusc(det_annos, output_dir, location)
    #     res = {
    #         "results": {
    #             "nusc": res_nusc["results"]["nusc"],
    #         },
    #         "detail": {
    #             "eval.nusc": res_nusc["detail"]["nusc"],
    #         },
    #     }
    #     return res



##### add nuscenes info #####

# def create_nuscenes_infos(root_path, version="v1.0-trainval", max_sweeps=10, location=None, day_night=None):
#     print("version", version)
#     from nuscenes.nuscenes import NuScenes
#     nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
#     from nuscenes.utils import splits
#     available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
#     assert version in available_vers
#     if version == "v1.0-trainval":
#         train_scenes = splits.train
#         val_scenes = splits.val
#     elif version == "v1.0-test":
#         train_scenes = splits.test
#         val_scenes = []
#     elif version == "v1.0-mini":
#         train_scenes = splits.mini_train
#         val_scenes = splits.mini_val
#     else:
#         raise ValueError("unknown")
#     test = "test" in version
#     root_path = Path(root_path)
#     # filter exist scenes. you may only download part of dataset.
#     if location is not None:
#         print("location",location)
#         available_scenes = _get_available_scenes(nusc, location=location)
#     else:
#         available_scenes = _get_available_scenes(nusc)
#     available_scene_names = [s["name"] for s in available_scenes]
#     train_scenes = list(
#         filter(lambda x: x in available_scene_names, train_scenes))
#     val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
#     train_scenes = set([
#         available_scenes[available_scene_names.index(s)]["token"]
#         for s in train_scenes
#     ])
#     val_scenes = set([
#         available_scenes[available_scene_names.index(s)]["token"]
#         for s in val_scenes
#     ])
#     if test:
#         print(f"test scene: {len(train_scenes)}")
#     else:
#         print(
#             f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

#     train_nusc_infos, val_nusc_infos, others_nusc_infos = _fill_trainval_infos(
#         nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

#     print("finish fill infos")
#     metadata = {
#         "version": version,
#     }

#     if location is None:
#         location = ''

#     if test:
#         print(f"test sample: {len(train_nusc_infos)}")
#         data = {
#             "infos": train_nusc_infos,
#             "metadata": metadata,
#         }
#         with open(root_path / F"pcdet_{location}_infos_test.pkl", 'wb') as f:
#             pickle.dump(data, f)
#     else:
#         print(
#             f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
#         )
#         data = {
#             "infos": train_nusc_infos,
#             "metadata": metadata,
#         }
#         with open(root_path / F"pcdet_{location}_infos_train.pkl", 'wb') as f:
#             pickle.dump(data, f)

#         data["infos"] = val_nusc_infos
#         with open(root_path / F"pcdet_{location}_infos_val.pkl", 'wb') as f:
#             pickle.dump(data, f)

#         data["infos"] = others_nusc_infos
#         with open(root_path / F"pcdet_{location}_infos_other.pkl", 'wb') as f:
#             pickle.dump(data, f)

#     print('---------------Start create groundtruth database for data augmentation---------------')

#     # dataset = BaseNuscenesDataset(root_path=data_path)

#     # dataset.set_split(train_split)
#     # dataset.create_groundtruth_database(train_filename, split=train_split)

#     print('---------------Data preparation Done---------------')


# def _get_available_scenes(nusc, location=None):
#     available_scenes = []
#     print("total scene num:", len(nusc.scene))

#     log_place_dict={}
#     with open('/home/wzha8158/datasets/3D_Detection/Nuscenes/v1.0-trainval/log.json') as json_file:
#         log_data = json.load(json_file)
#         for p in log_data:
#             log_place_dict[p["token"]]= p["location"]

#     with open(F"/home/wzha8158/datasets/3D_Detection/Nuscenes/ImageSets/nuscenes_{location}.txt","w") as f:
#         for scene in nusc.scene:
#             scene_token = scene["token"]
#             log_token = scene['log_token']
#             scene_rec = nusc.get('scene', scene_token)
#             sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
#             sd_rec = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
#             has_more_frames = True
#             scene_not_exist = False
#             while has_more_frames:
#                 lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
#                 if not Path(lidar_path).exists():
#                     scene_not_exist = True
#                     break
#                 else:
#                     break
#                 if not sd_rec['next'] == "":
#                     sd_rec = nusc.get('sample_data', sd_rec['next'])
#                 else:
#                     has_more_frames = False
#             if scene_not_exist:
#                 continue

#             if location is not None:
#                 # print("log_token", log_token)
#                 loc = log_place_dict[str(log_token)]
#                 if location not in loc:
#                     continue
#                 # print("scene spec", scene)
#             available_scenes.append(scene)

#             f.write(F"'{scene['name']}', ")
#     print("exist scene num:", len(available_scenes))
#     return available_scenes

# def _fill_trainval_infos(nusc,
#                          train_scenes,
#                          val_scenes,
#                          test=False,
#                          max_sweeps=10):
#     train_nusc_infos = []
#     val_nusc_infos = []
#     others_nusc_infos = []
#     from pyquaternion import Quaternion
#     for sample in iter(nusc.sample):
#         lidar_token = sample["data"]["LIDAR_TOP"]
#         cam_front_token = sample["data"]["CAM_FRONT"]
#         sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
#         cs_record = nusc.get('calibrated_sensor',
#                              sd_rec['calibrated_sensor_token'])
#         pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
#         lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

#         cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_front_token)
#         assert Path(lidar_path).exists(), (
#             "you must download all trainval data, key-frame only dataset performs far worse than sweeps."
#         )
#         info = {
#             "lidar_path": lidar_path,
#             "cam_front_path": cam_path,
#             "token": sample["token"],
#             "sweeps": [],
#             "lidar2ego_translation": cs_record['translation'],
#             "lidar2ego_rotation": cs_record['rotation'],
#             "ego2global_translation": pose_record['translation'],
#             "ego2global_rotation": pose_record['rotation'],
#             "timestamp": sample["timestamp"],
#         }

#         l2e_r = info["lidar2ego_rotation"]
#         l2e_t = info["lidar2ego_translation"]
#         e2g_r = info["ego2global_rotation"]
#         e2g_t = info["ego2global_translation"]
#         l2e_r_mat = Quaternion(l2e_r).rotation_matrix
#         e2g_r_mat = Quaternion(e2g_r).rotation_matrix

#         sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
#         sweeps = []
#         while len(sweeps) < max_sweeps:
#             if not sd_rec['prev'] == "":
#                 sd_rec = nusc.get('sample_data', sd_rec['prev'])
#                 cs_record = nusc.get('calibrated_sensor',
#                                      sd_rec['calibrated_sensor_token'])
#                 pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
#                 lidar_path = nusc.get_sample_data_path(sd_rec['token'])
#                 sweep = {
#                     "lidar_path": lidar_path,
#                     "sample_data_token": sd_rec['token'],
#                     "lidar2ego_translation": cs_record['translation'],
#                     "lidar2ego_rotation": cs_record['rotation'],
#                     "ego2global_translation": pose_record['translation'],
#                     "ego2global_rotation": pose_record['rotation'],
#                     "timestamp": sd_rec["timestamp"]
#                 }
#                 l2e_r_s = sweep["lidar2ego_rotation"]
#                 l2e_t_s = sweep["lidar2ego_translation"]
#                 e2g_r_s = sweep["ego2global_rotation"]
#                 e2g_t_s = sweep["ego2global_translation"]
#                 # sweep->ego->global->ego'->lidar
#                 l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
#                 e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

#                 R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
#                     np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
#                 T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
#                     np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
#                 T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
#                     l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
#                 sweep["sweep2lidar_rotation"] = R.T  # points @ R.T + T
#                 sweep["sweep2lidar_translation"] = T
#                 sweeps.append(sweep)
#             else:
#                 break
#         info["sweeps"] = sweeps
#         if not test:
#             annotations = [
#                 nusc.get('sample_annotation', token)
#                 for token in sample['anns']
#             ]
#             locs = np.array([b.center for b in boxes]).reshape(-1, 3)
#             dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
#             rots = np.array([b.orientation.yaw_pitch_roll[0]
#                              for b in boxes]).reshape(-1, 1)
#             velocity = np.array(
#                 [nusc.box_velocity(token)[:2] for token in sample['anns']])
#             # convert velo from global to lidar
#             for i in range(len(boxes)):
#                 velo = np.array([*velocity[i], 0.0])
#                 velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
#                     l2e_r_mat).T
#                 velocity[i] = velo[:2]

#             names = [b.name for b in boxes]
#             for i in range(len(names)):
#                 if names[i] in NuscenesDataset.NameMapping:
#                     names[i] = NuscenesDataset.NameMapping[names[i]]
#             names = np.array(names)
#             # we need to convert rot to SECOND format.
#             # change the rot format will break all checkpoint, so...
#             gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
#             assert len(gt_boxes) == len(
#                 annotations), f"{len(gt_boxes)}, {len(annotations)}"
#             info["gt_boxes"] = gt_boxes
#             info["gt_names"] = names
#             info["gt_velocity"] = velocity.reshape(-1, 2)
#             info["num_lidar_pts"] = np.array(
#                 [a["num_lidar_pts"] for a in annotations])
#             info["num_radar_pts"] = np.array(
#                 [a["num_radar_pts"] for a in annotations])
#         if sample["scene_token"] in train_scenes:
#             train_nusc_infos.append(info)
#         elif sample["scene_token"] in val_scenes:
#             val_nusc_infos.append(info)
#         else:
#             others_nusc_infos.append(info)

#     return train_nusc_infos, val_nusc_infos, others_nusc_infos



    # def set_split(self, split):
    #     self.init__(self.root_path, split)

    # def get_lidar(self, idx):

    #     lidar_path = Path(self.nusc_infos[idx]['lidar_path'])
    #     points = np.fromfile(
    #         str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
    #     points[:, 3] /= 255
    #     points[:, 4] = 0
    #     sweep_points_list = [points]
    #     ts = self.nusc_infos[idx]["timestamp"] / 1e6

    #     for sweep in self.nusc_infos[idx]["sweeps"]:
    #         points_sweep = np.fromfile(
    #             str(sweep["lidar_path"]), dtype=np.float32,
    #             count=-1).reshape([-1, 5])
    #         sweep_ts = sweep["timestamp"] / 1e6
    #         points_sweep[:, 3] /= 255
    #         points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
    #             "sweep2lidar_rotation"].T
    #         points_sweep[:, :3] += sweep["sweep2lidar_translation"]
    #         points_sweep[:, 4] = ts - sweep_ts
    #         sweep_points_list.append(points_sweep)

    #     points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]

    #     return points
    #     # lidar_file = os.path.join(self.root_split_path, 'velodyne', '%s.bin' % idx)
    #     # assert os.path.exists(lidar_file)
    #     # return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    # def get_image_shape(self, idx):
    #     img_file = Path(self.nusc_infos[idx]['cam_front_path'])
    #     assert os.path.exists(img_file)
    #     return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    # def get_image(self, idx):
    #     img_file = Path(self.nusc_infos[idx]['cam_front_path'])
    #     assert os.path.exists(img_file)
    #     return np.array(io.imread(img_file))

    # def get_label(self, idx):
    #     label_file = os.path.join(self.root_split_path, 'label_2', '%s.txt' % idx)
    #     assert os.path.exists(label_file)
    #     return object3d_utils.get_objects_from_label(label_file)

    # def get_calib(self, idx):
    #     return self.nusc_infos[idx]["cam_intrinsic"]

    # def get_road_plane(self, idx):
    #     plane_file = os.path.join(self.root_split_path, 'planes', '%s.txt' % idx)
    #     with open(plane_file, 'r') as f:
    #         lines = f.readlines()
    #     lines = [float(i) for i in lines[3].split()]
    #     plane = np.asarray(lines)

    #     # Ensure normal is always facing up, this is in the rectified camera coordinate
    #     if plane[1] > 0:
    #         plane = -plane

    #     norm = np.linalg.norm(plane[0:3])
    #     plane = plane / norm
    #     return plane

    # @staticmethod
    # def get_fov_flag(pts_rect, img_shape, calib):
    #     '''
    #     Valid point should be in the image (and in the PC_AREA_SCOPE)
    #     :param pts_rect:
    #     :param img_shape:
    #     :return:
    #     '''
    #     pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    #     val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    #     val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    #     val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    #     pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    #     return pts_valid_flag
