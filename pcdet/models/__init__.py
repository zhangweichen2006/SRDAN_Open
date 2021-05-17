from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector


def build_network(model_cfg, num_class, dataset, nusc=False):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, nusc=nusc
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict, t_mode='det', l=1):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict, t_mode=t_mode, l=l)

        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()
            
        if isinstance(ret_dict['loss'], tuple):
            loss = ret_dict['loss'][0].mean() 
            loss2 = ret_dict['loss'][1].mean()
            return ModelReturn((loss, loss2), tb_dict, disp_dict)
        else:
            loss = ret_dict['loss'].mean()
            return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
