import torch
import numpy as np
from ....utils import box_utils
from ....ops.iou3d_nms import iou3d_nms_utils

class AxisAlignedTargetAssigner(object):
    def __init__(self, anchor_target_cfg, anchor_generator_cfg, class_names, box_coder, match_height=False, nusc=False, fpn_layers=[]):
        super().__init__()
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.pseudo_pos_fraction = float(anchor_target_cfg.get("PSEUDO_POS_FRACTION", -1)) if float(anchor_target_cfg.get("PSEUDO_POS_FRACTION", -1)) >= 0 else None
        self.pseu_bg_default_w = float(anchor_target_cfg.get("PSEUDO_BG_DEFAULT_W", 0.5)) if float(anchor_target_cfg.get("PSEUDO_BG_DEFAULT_W", 0.5)) >= 0 else 0.5

        self.pseudo_balance_pos_neg_sample_ratio = float(anchor_target_cfg.get("PSEUDO_BALANCE_POS_NEG_SAMPLE_RATIO", -1)) if float(anchor_target_cfg.get("PSEUDO_BALANCE_POS_NEG_SAMPLE_RATIO", -1)) >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.pseudo_sample_size = anchor_target_cfg.get("PSEUDO_SAMPLE_SIZE", None)
        if anchor_target_cfg.get("RANGE", None):
            self.range_conditioned = True
        else:
            self.range_conditioned = False

        # self.FPN = False
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        self.nusc = nusc
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']
        self.fpn_layers = fpn_layers


    def assign_targets(self, all_anchors, gt_boxes_with_classes, use_multihead=False, dom_src=None, pseudo=False, pseudo_weights=[], fpn_layer=None, fpn_only=False, patch_tensor=None, dom_squeeze=True):#, localdom=False):
        """
        Args:
            all_anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8) #10
        Returns:

        """
        if fpn_layer is not None:
            self.FPN = True
        else:
            self.FPN = False

        bbox_targets = []
        bbox_src_targets = []
        cls_labels = []
        dom_labels = []
        box_dom_labels = []
        reg_weights = []
        # pseudo_weights = []
        # print("FPN", self.FPN)
        # print("all_anchors", all_anchors)

        if self.nusc:
            cls_idx = 9
        else:
            cls_idx = 7
        # print("pseudo", pseudo)
        # if pseudo:
        #     print("gt_boxes_with_classes", gt_boxes_with_classes.shape)
        # if dom_batch_size != 0:
        #     batch_size = dom_batch_size
        # else:
        batch_size = gt_boxes_with_classes.shape[0]

        gt_classes = gt_boxes_with_classes[:, :, cls_idx]
        gt_boxes = gt_boxes_with_classes[:, :, :cls_idx]

        # print("fpn", self.FPN)
        # print("fpn_layer", fpn_layer)
        # print("gt_classes", gt_classes.shape)
        # print("gt_boxes", gt_boxes.shape)
        # print("dom_src", dom_src)
        # print('dom_squeeze asssi', dom_squeeze)

        if dom_src is not None:
            if dom_src:
                dom_labels = torch.zeros((1), dtype=torch.float32, device=gt_boxes_with_classes.device)
            else:
                dom_labels = torch.ones((1), dtype=torch.float32, device=gt_boxes_with_classes.device)

            if patch_tensor is not None:
                # print("dom_labels", dom_labels.shape)
                dom_labels = dom_labels.expand_as(patch_tensor).view(-1)
                # print("dom_labels ", dom_labels.shape)

            if self.FPN:
                all_targets_dict = {
                    'dom_img_labels': dom_labels,
                }
                for layers in self.fpn_layers:
                    all_targets_dict.update({f'dom_img_labels_fpn{layers}': dom_labels})
            else:
                all_targets_dict = {
                    'dom_img_labels': dom_labels
                }

            return all_targets_dict


        for k in range(batch_size):
            cur_gt = gt_boxes[k]

            # print("cur_gt", cur_gt) #n*7,9[x, y, z, dx, dy, dz, sx, sy?, heading]
            # reversely get all bounding boxes, sort ascending order
            cnt = cur_gt.__len__() - 1 # last bbx id, no. bbx -1?
            while cnt > 0 and cur_gt[cnt].sum() == 0: # remove empty bbx?
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1] #select all valid
            cur_gt_classes = gt_classes[k][:cnt + 1].int() # bbx classes

            target_list = []
            # print("self.anchor_class_names", self.anchor_class_names)
            # print("self.class_names", self.class_names)
            # print("all_anchors",all_anchors)
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                # print("anchors", anchors.shape)
                #[1, 126, 126, 1, 2, 9] 252, 252
                if use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                else:
                    feature_map_size = anchors.shape[:3]
                    anchors = anchors.view(-1, anchors.shape[-1])
                # print("feature_map_size", feature_map_size)
                # print("anchors", anchors.shape) # 31752, 7 (stride 8) # 127008, 7

                single_target = self.assign_targets_single(
                    anchors,
                    cur_gt[mask],
                    gt_classes=cur_gt_classes[mask],
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name],
                    dom_src=dom_src,
                    pseudo=pseudo,
                    pseudo_weights=pseudo_weights,
                    range_conditioned=self.range_conditioned,
                    pseu_bg_default_w=self.pseu_bg_default_w,
                    fpn_layer=fpn_layer
                )
                # print("single_target", single_target)
                # box_cls_labels 31752
                # box_reg_targets 31752, 9
                # reg_weights 31752
                # box_cls_labels e.g. 0 0 0 0 0 1 1 1 -1 -1 -1 (invalid?) 0 0 0 0 0
                # print('inx', (single_target['box_cls_labels'] == 1).nonzero().flatten())
                target_list.append(single_target)

            # print("target_list", target_list)
            # box_cls_labels
            # box_reg_targets
            # reg_weights

            if use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
                if dom_src is not None:
                    target_dict['box_dom_labels'] = [t['box_dom_labels'].view(-1) for t in target_list]
                    target_dict['box_dom_labels'] = torch.cat(target_dict['box_dom_labels'], dim=0).view(-1)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]
                }
                # print("target_dict['box_reg_targets']", target_dict['box_reg_targets'][0].shape) # 252，252， 2，7
                #
                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'],
                                                           dim=-2).view(-1, self.box_coder.code_size)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)
                # print("box_cls_labels", target_dict["box_cls_labels"].shape) #31752
                # print("box_reg_targets", target_dict["box_reg_targets"].shape)# 31752,9
                # print("reg_weights", target_dict["reg_weights"].shape) #31752

                if dom_src is not None:
                    target_dict['box_dom_labels'] = [t['box_dom_labels'].view(*feature_map_size, -1) for t in target_list]
                    target_dict['box_dom_labels'] = torch.cat(target_dict['box_dom_labels'], dim=-1).view(-1)

            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])
            if dom_src is not None:
                box_dom_labels.append(target_dict['box_dom_labels'])

            # print("target_dict", target_dict)

        bbox_targets = torch.stack(bbox_targets, dim=0)

        cls_labels = torch.stack(cls_labels, dim=0)

        reg_weights = torch.stack(reg_weights, dim=0)

        if self.FPN:
            all_targets_dict = {
                f'box_cls_labels_fpn{fpn_layer}': cls_labels,
                f'box_reg_targets_fpn{fpn_layer}': bbox_targets,
                f'reg_weights_fpn{fpn_layer}': reg_weights
            }
        else:
            all_targets_dict = {
                'box_cls_labels': cls_labels,
                'box_reg_targets': bbox_targets,
                'reg_weights': reg_weights
            }
        # print("cls_labels", all_targets_dict["box_cls_labels"].shape) # 31752
        # print('inx', (all_targets_dict['box_cls_labels'] == 1).nonzero().flatten()) # ids 2,512,21342 ...

        # if pseudo:
        #     print("all_targets_dict['box_cls_labels']", all_targets_dict['box_cls_labels'])
        #     print("all_targets_dict['box_reg_targets']", all_targets_dict['box_reg_targets'])
        #     print("all_targets_dict['reg_weights']", all_targets_dict['reg_weights'])
        #     print('inx cls', (all_targets_dict['box_cls_labels'] == 1).nonzero().flatten())
        #     print('inx w', (all_targets_dict['reg_weights'] != 0).nonzero().flatten())

        if dom_src is not None:
            box_dom_labels = torch.stack(box_dom_labels, dim=0)
            all_targets_dict['box_dom_labels'] = box_dom_labels
            if self.FPN:
                all_targets_dict[f'box_dom_labels_fpn{fpn_layer}'] = box_dom_labels

        return all_targets_dict

    def assign_targets_single(self, anchors,
                         gt_boxes,
                         gt_classes,
                         matched_threshold=0.6,
                         unmatched_threshold=0.45,
                         dom_src=None,
                         pseudo=False,
                         pseudo_weights=[],
                         range_conditioned=False,
                         pseu_bg_default_w=0.5,
                         fpn_layer=None
                        ):

        # print("fpn_layer", fpn_layer) # 3
        # print("anchors", anchors.shape) # 127008
        # print("gt_boxes", gt_boxes.shape) # 14

        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        if pseudo:
            # print("gt_classes", gt_classes)
            if len(gt_classes) > 0:
                fg_pseu_classes_idx = torch.nonzero(gt_classes > 0)[:, 0]
                bg_pseu_classes_idx = torch.nonzero(gt_classes == 0)[:, 0]
                num_gt = len(fg_pseu_classes_idx)
                num_bg = len(bg_pseu_classes_idx)
            else:
                return {}

        # if fpn_layer is not None:


        # box_ndim = anchors.shape[1]

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        invalid_lbs = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        if dom_src is not None:
            if dom_src:
                dom_labels = torch.zeros((1), dtype=torch.float32, device=gt_boxes_with_classes.device)
            else:
                dom_labels = torch.ones((1), dtype=torch.float32, device=gt_boxes_with_classes.device)

        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        if pseudo:
            pseu_w = torch.zeros((num_anchors,), device=anchors.device)

        if self.nusc:
            bbx_len=9
        else:
            bbx_len=7

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            if pseudo:
                if num_gt > 0:
                    anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:bbx_len], gt_boxes[fg_pseu_classes_idx, 0:bbx_len], nusc=self.nusc) \
                        if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:bbx_len], gt_boxes[fg_pseu_classes_idx, 0:bbx_len], nusc=self.nusc)

                    # print("anchor_by_gt_overlap", anchor_by_gt_overlap)

                    anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
                    anchor_to_gt_max = anchor_by_gt_overlap[
                        torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax
                    ]

                    gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
                    gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
                    empty_gt_mask = gt_to_anchor_max == 0
                    gt_to_anchor_max[empty_gt_mask] = -1
                    anchors_with_max_overlap = torch.nonzero(anchor_by_gt_overlap == gt_to_anchor_max)[:, 0]

                    gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]

                ############ bg ###########
                if num_bg > 0:
                    anchor_by_bg_overlap_bg = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:bbx_len], gt_boxes[bg_pseu_classes_idx, 0:bbx_len], nusc=self.nusc) \
                        if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:bbx_len], gt_boxes[bg_pseu_classes_idx, 0:bbx_len], nusc=self.nusc)

                    anchor_to_bg_argmax = torch.from_numpy(anchor_by_bg_overlap_bg.cpu().numpy().argmax(axis=1)).cuda()
                    anchor_to_bg_max = anchor_by_bg_overlap_bg[
                        torch.arange(num_anchors, device=anchors.device), anchor_to_bg_argmax
                    ]

                    bg_to_anchor_argmax = torch.from_numpy(anchor_by_bg_overlap_bg.cpu().numpy().argmax(axis=0)).cuda()
                    bg_to_anchor_max = anchor_by_bg_overlap_bg[bg_to_anchor_argmax, torch.arange(num_bg, device=anchors.device)]
                    empty_bg_mask = bg_to_anchor_max == 0
                    bg_to_anchor_max[empty_bg_mask] = -1
                    anchors_with_max_overlap_bg = torch.nonzero(anchor_by_bg_overlap_bg == bg_to_anchor_max)[:, 0]

                    bg_inds_force = anchor_to_bg_argmax[anchors_with_max_overlap_bg]

            else:
                anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:bbx_len], gt_boxes[:, 0:bbx_len], nusc=self.nusc) \
                    if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:bbx_len], gt_boxes[:, 0:bbx_len], nusc=self.nusc)

                anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
                anchor_to_gt_max = anchor_by_gt_overlap[
                    torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax
                ]

                gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
                gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
                empty_gt_mask = gt_to_anchor_max == 0
                gt_to_anchor_max[empty_gt_mask] = -1
                anchors_with_max_overlap = torch.nonzero(anchor_by_gt_overlap == gt_to_anchor_max)[:, 0]

                gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]

            ############ assign force labels ##########
            if pseudo:
                # force fg and bg
                # print("num_gt", num_gt)
                if num_gt > 0:
                    # print("labels", labels)
                    # print("pseu_w", pseu_w)
                    # print("gt_classes", gt_classes)
                    # print("pseudo_weights", pseudo_weights)
                    labels[anchors_with_max_overlap] = gt_classes[fg_pseu_classes_idx][gt_inds_force]
                    pseu_w[anchors_with_max_overlap] = pseudo_weights[fg_pseu_classes_idx][gt_inds_force]

                if num_bg > 0:
                    labels[anchors_with_max_overlap_bg] = gt_classes[bg_pseu_classes_idx][bg_inds_force]
                    pseu_w[anchors_with_max_overlap_bg] = pseudo_weights[bg_pseu_classes_idx][bg_inds_force]
            else:
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

            ############ assign related anchor labels ##########
            #torch.where(labels >= 0, labels, invalid_lbs))
            if pseudo:
                if num_gt > 0:
                    pos_inds = anchor_to_gt_max >= matched_threshold
                    #31752
                    gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
                    #[001] 31752
                    gt_ids[pos_inds] = gt_inds_over_thresh.int()
                    #all anchor id with gt idx. 7 7 1 2 3 0 0 5...
                    # pseu_fg_inds = gt_ids[pos_inds][torch.nonzero(gt_classes[gt_ids[pos_inds].long()] > 0)[:, 0]]#
                    # print("pseu_fg_inds", pseu_fg_inds)
                    labels[pos_inds] = gt_classes[fg_pseu_classes_idx][gt_inds_over_thresh]
                ########## backgrounds #########
                if num_bg > 0:
                    bg_inds = anchor_to_bg_max >= matched_threshold
                    bg_inds_over_thresh = anchor_to_bg_argmax[bg_inds]
                    gt_ids[bg_inds] = bg_inds_over_thresh.int()

                    # given background
                    pseu_bg_inds = torch.nonzero(anchor_to_bg_max > matched_threshold)[:, 0]
                    # remaining background
                    bg_inds = torch.nonzero(anchor_to_gt_max < unmatched_threshold)[:, 0]

                    labels[pseu_bg_inds] = 0
                    pseu_w[pseu_bg_inds] = pseudo_weights[bg_pseu_classes_idx][bg_inds_over_thresh]
                else:
                    pseu_bg_inds = []
                    bg_inds = torch.nonzero(anchor_to_gt_max < unmatched_threshold)[:, 0]

            else:
                pos_inds = anchor_to_gt_max >= matched_threshold
                gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
                labels[pos_inds] = gt_classes[gt_inds_over_thresh]
                gt_ids[pos_inds] = gt_inds_over_thresh.int()
                bg_inds = torch.nonzero(anchor_to_gt_max < unmatched_threshold)[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = torch.nonzero(labels > 0)[:, 0]
        reg_weights = anchors.new_zeros((num_anchors,))
        # foreground anchor ratio
        ##############  assign background and invalid #####################
        if pseudo:
            if self.pseudo_pos_fraction is not None:
                # foreground boxes
                num_fg = int(self.pseudo_pos_fraction * self.pseudo_sample_size) # fg anchor max nums
                # max bbx , disable some foreground, label as -1
                if len(fg_inds) > num_fg:
                    num_disabled = len(fg_inds) - num_fg
                    disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                    labels[disable_inds] = -1
                    pseu_w[disable_inds] = 0
                    fg_inds = torch.nonzero(labels > 0)[:, 0]

                # background boxes
                if self.pseudo_balance_pos_neg_sample_ratio is not None:
                    num_bg = int(self.pseudo_balance_pos_neg_sample_ratio * len(fg_inds))
                    # print("num_bg", num_bg)
                    # disable some bg if exceed
                    if len(pseu_bg_inds) > num_bg:
                        # print("bg exceed")
                        num_disabled = len(pseu_bg_inds) - num_bg
                        disable_inds = pseu_bg_inds[torch.randperm(len(pseu_bg_inds))[:num_disabled]]
                        labels[disable_inds] = -1
                        pseu_w[disable_inds] = 0
                    else:
                        # print("bg 0.5")
                        remain_bg = num_bg - len(pseu_bg_inds)
                        enable_inds = bg_inds[torch.randperm(len(bg_inds))[:remain_bg]]

                        labels[enable_inds] = 0
                        pseu_w[enable_inds] = self.pseu_bg_default_w

                else:
                    # if no pseu bbxs are selected as background,
                    # randomly select neg bbx
                    remain_bg = self.pseudo_sample_size - (labels > 0).sum() - len(pseu_bg_inds)
                    if len(bg_inds) > remain_bg:
                        enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(remain_bg,))]
                        labels[enable_inds] = 0
                        pseu_w[enable_inds] = self.pseu_bg_default_w

            else:
                if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                    labels[:] = 0
                else:
                    labels[pseu_bg_inds] = 0#gt_classes[bg_pseu_classes_idx][bg_inds_over_thresh]
                    pseu_w[pseu_bg_inds] = pseudo_weights[bg_pseu_classes_idx][bg_inds_over_thresh]

                    labels[anchors_with_max_overlap_bg] = gt_classes[bg_pseu_classes_idx][bg_inds_force]
                    pseu_w[anchors_with_max_overlap_bg] = pseudo_weights[bg_pseu_classes_idx][bg_inds_force]

        else:
            if self.pos_fraction is not None:
                # foreground boxes
                num_fg = int(self.pos_fraction * self.sample_size) # fg anchor max nums
                # print("")
                # max bbx , disable some foreground, label as -1
                if len(fg_inds) > num_fg:
                    num_disabled = len(fg_inds) - num_fg
                    disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                    labels[disable_inds] = -1
                    fg_inds = torch.nonzero(labels > 0)[:, 0]

                # no labels are selected as background
                num_bg = self.sample_size - (labels > 0).sum()
                if len(bg_inds) > num_bg:
                    enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                    labels[enable_inds] = 0
                # bg_inds = torch.nonzero(labels == 0)[:, 0]
            else:
                if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                    labels[:] = 0
                else:
                    labels[bg_inds] = 0
                    labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        # print("self.box_coder.code_size", self.box_coder.code_size)
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            if self.nusc:
                bbox_targets[fg_inds, :] = self.box_coder.encode_torch_velo(fg_gt_boxes, fg_anchors)
            else:
                bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        if pseudo:
            reg_weights = pseu_w
        else:
            if self.norm_by_num_examples:
                num_examples = (labels >= 0).sum()
                num_examples = num_examples if num_examples > 1.0 else 1.0
                reg_weights[labels > 0] = 1.0 / num_examples
            else:
                reg_weights[labels > 0] = 1.0

        ret_dict = {'box_cls_labels': labels,
                    'box_reg_targets': bbox_targets,
                    'reg_weights': reg_weights,
                    }

        if dom_src is not None:
            ret_dict['box_dom_labels'] = box_dom_labels

        return ret_dict


