from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(model, optim_cfg, dom_opt=False, mcd_opt=False, atten_opt=False, feat_lr_ratio=0.1):
    
    if dom_opt or mcd_opt or atten_opt:
        if optim_cfg.OPTIMIZER == 'adam_onecycle':
            def children(m: nn.Module):
                return list(m.children())

            def num_children(m: nn.Module) -> int:
                return len(children(m))

            def get_sep_layer_groups(m):
                base_layers = []
                head_layers = []
                dom_layers = []
                att_layers = []
                for name, module in m.named_children():
                    exist = False
                    if name in ["dense_head", "point_head"]:
                        for child_name, child_module in module.named_children():
                            # if "att" in child_name:
                            #     # print("att child_name", child_name)
                            #     att_layers.append(child_module)
                            # else:
                                # print("dom child_name", child_name)
                                # if "dom" in child_name:
                            dom_layers.append(child_module)
                            att_layers.append(child_module)
                            # else:
                            #     head_layers.append(child_module)

                    else:
                        base_layers.append(module)

                return [nn.Sequential((*base_layers)), nn.Sequential((*head_layers)), nn.Sequential((*dom_layers)), nn.Sequential((*att_layers))]

            flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
            get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

            optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
            feat_lr = feat_lr_ratio * 3e-3

            base_layers, head_layers, dom_layers, att_layers = get_sep_layer_groups(model)

            if dom_opt or mcd_opt:
                optimizer_dom = OptimWrapper.create(
                    optimizer_func, [feat_lr, 3e-3], [base_layers, dom_layers], wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
                )
            elif atten_opt:
                atten_lr = optim_cfg.LR
                # print('atten_lr', atten_lr)
                optimizer_dom = OptimWrapper.create(
                    optimizer_func, atten_lr, att_layers, wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
                )

        elif optim_cfg.OPTIMIZER == 'adam' or optim_cfg.OPTIMIZER == 'sgd':
            def get_sep_layer_groups(m):
                base_layers = []
                head_layers = []
                dom_layers = []
                att_layers = []
                for name, module in m.named_children():
                    exist = False
                    if name in ["dense_head", "point_head"]:
                        for child_name, child_module in module.named_children():
                            if "att" in child_name:
                                print("att child_name", child_name)
                                att_layers.append(child_module)
                            else:
                                # print("dom child_name", child_name)
                                # if "dom" in child_name:
                                dom_layers.append(child_module)
                            # att_layers.append(child_module)
                            # else:
                            #     head_layers.append(child_module)

                    else:
                        base_layers.append(module)

                return [nn.Sequential((*base_layers)), nn.Sequential((*head_layers)), nn.Sequential((*dom_layers)), nn.Sequential((*att_layers))]

            base_layers, head_layers, dom_layers, att_layers = get_sep_layer_groups(model)

            if atten_opt:
                if optim_cfg.OPTIMIZER == 'adam':
                    optimizer = optim.Adam(
                        att_layers.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY
                    )
                elif optim_cfg.OPTIMIZER == 'sgd':
                    optimizer = optim.SGD(
                        att_layers.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY, momentum=optim_cfg.MOMENTUM
                    )

            else:
                raise NotImplementedError

            return optimizer

        else:
            raise NotImplementedError
        return optimizer_dom

    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg, dom=False):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)
    
    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs

    # if feat_lr_ratio != 1:
    #     if optim_cfg.OPTIMIZER == 'adam_onecycle':
    #         lr_scheduler = OneCycle(
    #             optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
    #         )
    #     else:
    #         lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    #         if optim_cfg.LR_WARMUP:
    #             lr_warmup_scheduler = CosineWarmupLR(
    #                 optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
    #                 eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
    #             )
    #     return lr_scheduler, lr_warmup_scheduler

    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler