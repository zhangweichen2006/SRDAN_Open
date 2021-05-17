from .detector3d_template import Detector3DTemplate


class SECONDNetRangeInv(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, nusc=False):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, nusc=nusc)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, t_mode='det', l=1):   
        # print("batch_dict", batch_dict)
        batch_dict['t_mode'] = t_mode
        batch_dict['l'] = l
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

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
            loss_range, tb_dict = self.dense_head.get_range_inv_loss()

            tb_dict = {
                'loss_rpn': loss_rpn.item(),
                'loss_range': loss_range.item(),
                **tb_dict
            }
            loss = loss_rpn #+ loss_range
            return loss, tb_dict, disp_dict
        
        elif 'dom_range' in t_mode:
            loss_range, tb_dict = self.dense_head.get_range_inv_loss()

            tb_dict = {
                'loss_range': loss_range.item(),
                **tb_dict
            }
            loss = loss_range
            return loss, tb_dict, disp_dict

        elif 'dom' in t_mode:
            loss_dom, tb_dict = self.dense_head.get_dom_loss()

            tb_dict = {
                'loss_dann': loss_dom.item(),
                **tb_dict
            }
            loss = loss_dom 
            return loss, tb_dict, disp_dict
        
        
