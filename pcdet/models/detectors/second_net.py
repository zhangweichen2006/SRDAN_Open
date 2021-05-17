from .detector3d_template import Detector3DTemplate


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, nusc=False):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, nusc=nusc)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, t_mode='det', l=1):
        # print("batch_dict", batch_dict)
        batch_dict['t_mode'] = t_mode
        batch_dict['l'] = l
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # print("batch_dict", batch_dict)
        # print("spatial_features_2d", batch_dict['spatial_features_2d'].nonzero())

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(t_mode=t_mode)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, t_mode='det'):
        disp_dict = {}

        if 'det' in t_mode or 'pseudo' in t_mode:
            loss_rpn, tb_dict = self.dense_head.get_loss()

            disp_dict.update({
                f'loss_{t_mode}': loss_rpn.item(),
            })
            loss = loss_rpn
            return loss, tb_dict, disp_dict
        elif 'dom' in t_mode:


            if self.model_cfg.DENSE_HEAD.get('DIFF_DOM_OPT', False):
                if 'diffdom' in t_mode:
                    loss_dom, tb_dict = self.dense_head.get_dom_loss(diffdom=True)
                elif 'reg' in t_mode:
                    loss_dom, tb_dict = self.dense_head.get_dom_loss(reg=True)
                else:
                    loss_dom, tb_dict = self.dense_head.get_dom_loss()
                    # print("loss_dom ", loss_dom)
            else:
                loss_dom, tb_dict = self.dense_head.get_dom_loss()
            disp_dict.update({
                f'loss_{t_mode}': loss_dom.item(),
            })
            loss = loss_dom
            return loss, tb_dict, disp_dict


