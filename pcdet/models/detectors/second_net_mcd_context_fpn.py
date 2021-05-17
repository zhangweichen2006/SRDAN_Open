from .detector3d_template import Detector3DTemplate


class SECONDNetMCDContextFPN(Detector3DTemplate):
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

            num_fpn_up = self.model_cfg.get('FPN_UP_LAYERS', 0)
            num_fpn_down = self.model_cfg.get('FPN_DOWN_LAYERS',0)
            num_fpn_downup = self.model_cfg.get('FPN_DOWNUP_LAYERS', 0)
            fpn_layers = [str(3 - l) for l in range(num_fpn_up)] + [str(4 + 1 + l) for l in range(num_fpn_down) if num_fpn_down > 0] + [str(4 + l) for l in range(num_fpn_downup+1) if num_fpn_downup > 0]

            if num_fpn_up + num_fpn_downup > 0:
                pred_dicts_fpn, recall_dicts_fpn = self.post_processing_FPN(batch_dict, fpn_layers)
                return pred_dicts, recall_dicts, pred_dicts_fpn, recall_dicts_fpn

            return pred_dicts, recall_dicts

    def get_training_loss(self, t_mode='det'):
        disp_dict = {}

        if 'dom_img_det' in t_mode:
            # loss_rpn, tb_dict = self.dense_head.get_mcd_src_loss()
            if not self.fpn_only:
                loss_rpn, tb_dict = self.dense_head.get_mcd_src_loss(tb_dict)
            loss_rpn_fpn, tb_dict = self.dense_head.get_mcd_src_fpn_loss(tb_dict)

            disp_dict.update({
                'loss_rpn': loss_rpn.item(),
                'loss_rpn_fpn': loss_rpn_fpn.item()
            })

            if not self.fpn_only:
                loss = loss_rpn + loss_rpn_fpn 
            else:
                loss = loss_rpn_fpn 

            return loss, tb_dict, disp_dict

        elif 'dom_img_det_tgt' in t_mode:
            if not self.fpn_only:
                loss_rpn, tb_dict = self.dense_head.get_mcd_tgt_loss()

            loss_rpn_fpn, tb_dict = self.dense_head.get_mcd_tgt_fpn_loss(tb_dict)
            
            tb_dict = {
                'loss_mcd_tgt': loss_rpn.item(),
                'loss_mcd_tgt_fpn': loss_rpn_fpn.item(),
                **tb_dict
            }
            if not self.fpn_only:
                loss = loss_rpn + loss_rpn_fpn 
            else:
                loss = loss_rpn_fpn 

            return loss, tb_dict, disp_dict
            
        elif 'dom' in t_mode:
            # loss_dann, tb_dict = self.dense_head.get_dom_loss()
            
            
            if not self.fpn_only:
                loss_dom, tb_dict = self.dense_head.get_dom_loss(tb_dict)
            loss_dom_fpn, tb_dict = self.dense_head.get_fpn_dom_loss(tb_dict)

            tb_dict = {
                'loss_dann': loss_dann.item(),
                **tb_dict
            }

            if not self.fpn_only:
                loss = loss_dom + loss_dom_fpn
            else:
                loss = loss_dom_fpn

            return loss, tb_dict, disp_dict

        el
        
        
