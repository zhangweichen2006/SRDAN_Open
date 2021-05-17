from .detector3d_template import Detector3DTemplate


class SECONDNetMCDContext(Detector3DTemplate):
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
        if 'dom_img_det' in t_mode:
            loss_rpn, tb_dict = self.dense_head.get_mcd_src_loss()

            tb_dict = {
                'loss_rpn': loss_rpn.item(),
                **tb_dict
            }
            loss = loss_rpn
            return loss, tb_dict, disp_dict
            
        elif 'dom_img_det_tgt' in t_mode:
            loss_rpn, tb_dict = self.dense_head.get_mcd_tgt_loss()
            
            tb_dict = {
                'loss_dann': loss_rpn.item(),
                **tb_dict
            }
            loss = loss_rpn
            return loss, tb_dict, disp_dict
            
        elif 'dom' in t_mode:
            loss_dann, tb_dict = self.dense_head.get_dom_loss()
            
            tb_dict = {
                'loss_dann': loss_dann.item(),
                **tb_dict
            }
            loss = loss_dann
            return loss, tb_dict, disp_dict

        el
        
        
