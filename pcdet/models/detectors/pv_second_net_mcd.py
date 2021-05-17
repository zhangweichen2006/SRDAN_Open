from .detector3d_template import Detector3DTemplate


class PVSECONDNetMCD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, nusc=False):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, nusc=nusc)
        self.module_list = self.build_networks()
        self.model_cfg = model_cfg

    def forward(self, batch_dict, t_mode='det', l=1):   
        # print("batch_dict", batch_dict)
        batch_dict['t_mode'] = t_mode
        batch_dict['l'] = l

        # if not self.training and self.model_cfg.get('CONTEXT', None):
        #     batch_dict['t_mode'] = 'dom_img_det'

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
            loss_point, tb_dict = self.point_head.get_mcd_src_loss()
            loss_rpn, tb_dict = self.dense_head.get_mcd_src_loss(tb_dict)
            # print("loss_point", loss_point)
            # print("loss_rpn", loss_rpn)
            # loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

            # tb_dict = {
            #     'loss_rpn': loss_rpn.item(),
            #     **tb_dict
            # }
            loss = loss_point + loss_rpn

            disp_dict = {
                'src_loss_point': loss_point.item(),
                'src_loss_rpn': loss_rpn.item(),
                **disp_dict
            }

            return loss, tb_dict, disp_dict

        elif 'dom' in t_mode:
            loss_point, tb_dict = self.point_head.get_mcd_tgt_loss()
            loss_rpn, tb_dict = self.dense_head.get_mcd_tgt_loss(tb_dict)
            # loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

            loss = loss_point + loss_rpn

            disp_dict = {
                'tgt_mcd_loss': loss.item(),
                **disp_dict
            }
            # disp_dict = {
            #     'loss_dom': loss_dom.item(),
            #     'loss_dom_point': loss_dom_point.item(),
            #     **disp_dict
            # }

            return loss, tb_dict, disp_dict
        
        
