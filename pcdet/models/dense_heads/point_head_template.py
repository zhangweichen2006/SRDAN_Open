import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils

# from emd import EMDLoss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1+1, batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss1 += kernels[s1, s2] + kernels[t1, t2]
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    for s1 in range(batch_size):
        for s2 in range(batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss2 -= kernels[s1, t2] + kernels[s2, t1]
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2

class PointHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, nusc=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None
        self.nusc = nusc
        self.range_keys = self.model_cfg.get('NUM_KEYPOINTS_RANGE', {}).keys()

        self.range_dom_inv = self.model_cfg.get('RANGE_DOM_INVATIANT', False)
        self.range_inv = self.model_cfg.get('RANGE_INVATIANT', False)
        self.emd = self.model_cfg.get('EMD', False)
        self.mcd = self.model_cfg.get('MCD', False)
        self.point_interpolation = self.model_cfg.get('POINT_INTERPOLATION', False)
        # self.point_da = self.model_cfg.get('POINT_DA', False)
        self.dom_attention = self.model_cfg.get('DOM_ATTENTION', False)
        self.pos_da = self.model_cfg.get('POS_DA', False)
        # print("emd", self.emd)
        # if self.emd:
            # self.emd_loss = EMDLoss()
            # self.mmd_loss = MMD()

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
            )
        else:
            self.reg_loss_func = F.smooth_l1_loss
        
        self.add_module(
            'mmd_loss_func',
            loss_utils.RBFMMDLoss(sigma_list=[0.01, 0.1, 1, 10, 100])
        )
        
    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0, dom_src=None, range_key=None, range_inv=False, joint=False, localdom=False):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8) #(B, M, 10)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        point_size = 4
        if not self.nusc:
            gt_box_size = 8
        else:
            gt_box_size = 10

        if range_key is not None:
            suffix = f'_{range_key}' 
        elif joint:
            suffix = '_joint'
        else:
            suffix = ''

        # print("points", points.shape)
        assert len(points.shape) == 2 and points.shape[1] == point_size, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == gt_box_size, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == gt_box_size, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        
        if range_inv:
            range_labels1 = torch.ones((batch_size), dtype=torch.float32, device=points.device)

            range_labels2 = torch.zeros((batch_size), dtype=torch.float32, device=points.device)
            
            all_targets_dict = {
                f'range_point_labels1{suffix}': range_labels1,
                f'range_point_labels2{suffix}': range_labels2
            }
                    # 'dom_ins_labels': dom_labels
            return all_targets_dict


        if dom_src is not None:
            if dom_src:
                dom_labels = torch.zeros((1), dtype=torch.float32, device=points.device)
            else:
                dom_labels = torch.ones((1), dtype=torch.float32, device=points.device)
            # if dom_src:
            #     if localdom:
            #         dom_labels = torch.ones(points.shape[0], dtype=torch.float32, device=points.device)
            #     else:
            #         dom_labels = torch.ones(batch_size, dtype=torch.float32, device=points.device)
            # else:
            #     if localdom:
            #         dom_labels = torch.zeros(points.shape[0], dtype=torch.float32, device=points.device)
            #     else:
            #         dom_labels = torch.zeros(batch_size, dtype=torch.float32, device=points.device)

            all_targets_dict = {
                f'dom_point_labels{suffix}': dom_labels
                # 'dom_ins_labels': dom_labels
            }
            return all_targets_dict
            
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], gt_box_size)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:gt_box_size-1].contiguous(), nusc=self.nusc
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:gt_box_size-1].contiguous(), nusc=self.nusc
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, gt_box_size-1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            if ret_box_labels:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), gt_box_size))
                # fg_point_box_labels = self.box_coder.encode_torch(points_single[fg_flag], gt_box_of_fg_points)
                # point_box_labels_single[fg_flag] = fg_point_box_labels
                # point_box_labels[bs_mask] = point_box_labels_single
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        # print("point_cls_labels", point_cls_labels.shape)
        # print("point_box_labels", point_box_labels.shape)
        # print("suffix", suffix)
        # if not self.nusc:
        targets_dict = {
            f'point_cls_labels{suffix}': point_cls_labels,
            f'point_box_labels{suffix}': point_box_labels,
            f'point_part_labels{suffix}': point_part_labels
        }
        # else:
        #     # extension
        #     targets_dict = {
        #         'point_cls_labels': point_cls_labels,
        #         'point_box_labels': point_box_labels,
        #         'point_part_labels': point_part_labels
        #     }
        return targets_dict

    def get_box_layer_loss(self, tb_dict={}, joint=False, mcd_id=None):
        if joint:
            suffix = '_joint'
        elif mcd_id is not None:
            suffix = f'_{mcd_id}'
        else:
            suffix = ''
        pos_mask = self.forward_ret_dict[f'point_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict[f'point_box_labels']
        point_box_preds = self.forward_ret_dict[f'point_box_preds{suffix}']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_box = point_loss_box_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        
        tb_dict.update({'point_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict
    
    def get_range_box_layer_loss(self, tb_dict={}):
        range_point_loss_box = []
        for i in self.range_keys:

            pos_mask = self.forward_ret_dict[f'point_cls_labels_{i}'] > 0
            point_box_labels = self.forward_ret_dict[f'point_box_labels_{i}']
            point_box_preds = self.forward_ret_dict[f'point_box_preds_{i}']

            reg_weights = pos_mask.float()
            pos_normalizer = pos_mask.sum().float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)

            point_loss_box_src = self.reg_loss_func(
                point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
            )
            point_loss_box = point_loss_box_src.sum()

            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']

            range_point_loss_box.append(point_loss_box)
        
        point_loss_box_all = sum(range_point_loss_box) 

        
        tb_dict.update({f'point_loss_box': point_loss_box_all.item()})
        return point_loss_box, tb_dict
        

    def get_cls_layer_loss(self, tb_dict={}, joint=False, mcd_id=None):
        if joint:
            suffix = '_joint'
        elif mcd_id is not None:
            suffix = f'_{mcd_id}'
        else:
            suffix = ''
        point_cls_labels = self.forward_ret_dict[f'point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict[f'point_cls_preds{suffix}'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_range_cls_layer_loss(self, tb_dict={}):
        
        range_point_loss_cls = []
        for i in self.range_keys:
                
            point_cls_labels = self.forward_ret_dict[f'point_cls_labels_{i}'].view(-1)
            point_cls_preds = self.forward_ret_dict[f'point_cls_preds_{i}'].view(-1, self.num_class)

            positives = (point_cls_labels > 0)
            negative_cls_weights = (point_cls_labels == 0) * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            pos_normalizer = positives.sum(dim=0).float()
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]
            cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
            point_loss_cls = cls_loss_src.sum()
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS

            range_point_loss_cls.append(point_loss_cls * loss_weights_dict['point_cls_weight'])

        point_loss_cls_all = sum(range_point_loss_cls) 
        
        tb_dict.update({
            'point_loss_cls': point_loss_cls_all.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_dom_layer_loss(self, tb_dict={}):
        domain_criterion = torch.nn.BCEWithLogitsLoss().cuda()

        dom_labels = self.forward_ret_dict['dom_point_labels']
        dom_preds = self.forward_ret_dict['dom_point_preds']
        # print("dom_labels", dom_labels)
        # print("dom_preds", dom_preds)
        dom_loss = domain_criterion(dom_preds, dom_labels)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        dom_loss = dom_loss * loss_weights_dict['point_dom_weight']
        
        tb_dict.update({'point_dom_loss': dom_loss.item()})
        return dom_loss, tb_dict
    
    def get_range_dom_layer_loss(self, tb_dict={}):
        domain_criterion = torch.nn.BCEWithLogitsLoss().cuda()

        range_point_loss_dom = []
        for i in self.range_keys: 
            dom_labels = self.forward_ret_dict[f'dom_point_labels_{i}']
            dom_preds = self.forward_ret_dict[f'dom_point_preds_{i}']
            
            dom_loss = domain_criterion(dom_preds, dom_labels)

            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            dom_loss = dom_loss * loss_weights_dict['point_dom_weight']
        
            range_point_loss_dom.append(dom_loss)

        point_loss_dom = sum(range_point_loss_dom) 
        
        tb_dict.update({'point_dom_loss': point_loss_dom.item()})
        return point_loss_dom, tb_dict

    def get_range_layer_loss(self, tb_dict={}, mcd_id=None):
        if mcd_id is not None:
            suffix = f'_{mcd_id}'

        domain_criterion = torch.nn.BCEWithLogitsLoss().cuda()
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS

        range_point_loss = []
        for i in self.range_keys: 
            range_labels = self.forward_ret_dict[f'range_labels_{i}']
            range_preds = self.forward_ret_dict[f'range_preds_{i}{suffix}']
            # print("dom_labels", dom_labels)
            # print("dom_preds", dom_preds)
            range_loss = domain_criterion(range_preds, range_labels) * loss_weights_dict['point_range_inv_weight']

            range_point_loss.append(range_loss)

        point_loss_range = sum(range_point_loss) 
        
        tb_dict.update({'point_range_loss': point_loss_range.item()})
        return point_loss_range, tb_dict


    def get_range_inv_layer_loss(self, tb_dict={}):
        range_criterion = torch.nn.BCEWithLogitsLoss().cuda()
        range_point_loss = []

        for i in range(len(self.range_keys)-1): #[1,2]#,3
            
            key1 = list(self.range_keys)[i]
            key2 = list(self.range_keys)[i+1]

            if self.emd:
                feat1 = self.forward_ret_dict[f'emd_feat1_{key1}']
                feat2 = self.forward_ret_dict[f'emd_feat2_{key1}']
                
                # print("feat1", feat1.shape)
                # print("feat2", feat2.shape)
                range_loss = MMD(feat1, feat2)
                range_point_loss.append(range_loss)
            else:

                range_preds1 = self.forward_ret_dict[f'range_point_preds1_{key1}']
                range_preds2 = self.forward_ret_dict[f'range_point_preds2_{key1}']

                range_labels1 = self.forward_ret_dict[f'range_point_labels1_{key1}']
                range_labels2 = self.forward_ret_dict[f'range_point_labels2_{key1}']

                # print("dom_labels", dom_labels)
                # print("dom_preds", dom_preds)
                range_loss1 = range_criterion(range_preds1, range_labels1)
                range_loss2 = range_criterion(range_preds2, range_labels2)

                range_point_loss.append(range_loss1)
                range_point_loss.append(range_loss2)

        point_loss_range_all = sum(range_point_loss) 

        tb_dict.update({'point_loss_range': point_loss_range_all.item()})

        return point_loss_range_all, tb_dict

    def mcd_discrepancy(self, out1, out2):
        """discrepancy loss"""
        # print("F.softmax(out1, dim=-1)", F.softmax(out1, dim=-1).shape)
        out = torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))
        return out

    def get_mcd_cls_layer_loss(self, tb_dict={}):
        # print("self.forward_ret_dict", self.forward_ret_dict)
        pred_t1 = self.forward_ret_dict[f'mcd_point_cls_preds_1']
        pred_t2 = self.forward_ret_dict[f'mcd_point_cls_preds_2']
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS

        # print("pred_t1", pred_t1.shape)
        
        mcd_loss_adv = -1 * self.mcd_discrepancy(pred_t1, pred_t2) * loss_weights_dict['point_disc_cls_weight']
        
        tb_dict.update({'point_mcd_cls_loss': mcd_loss_adv.item()})

        return mcd_loss_adv, tb_dict
    
    def get_mcd_box_layer_loss(self, tb_dict={}):
        
        pred_t1 = self.forward_ret_dict[f'mcd_point_box_preds_1']
        pred_t2 = self.forward_ret_dict[f'mcd_point_box_preds_2']
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        
        mcd_loss_adv = -1 * self.mcd_discrepancy(pred_t1, pred_t2) * loss_weights_dict['point_disc_box_weight']
        
        tb_dict.update({'point_mcd_box_loss': mcd_loss_adv.item()})

        return mcd_loss_adv, tb_dict

    def get_part_layer_loss(self, tb_dict={}):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['point_part_labels']
        point_part_preds = self.forward_ret_dict['point_part_preds']
        point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        
        tb_dict.update({'point_loss_part': point_loss_part.item()})

        return point_loss_part, tb_dict

    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Args:
            points: (N, 3)
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)
        Returns:
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)

        """
        _, pred_classes = point_cls_preds.max(dim=-1)
        point_box_preds = self.box_coder.decode_torch(point_box_preds, points, pred_classes + 1)

        return point_cls_preds, point_box_preds


    def forward(self, **kwargs):
        raise NotImplementedError
