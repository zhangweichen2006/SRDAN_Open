from .detector3d_template import Detector3DTemplate


class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, nusc=False):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, nusc=nusc)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, t_mode='det', l=1):
        batch_dict['t_mode'] = t_mode
        batch_dict['l'] = l
        
        # if 'dom_img' in t_mode:
        #     for i in self.module_list:
        #         print(i.named_mo)
        #     module_list = self.module_list
        # else:
        #     module_list = self.module_list
            
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

        if t_mode=='det' or t_mode=='pseudo':
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

            loss = loss_rpn + loss_point + loss_rcnn

            tb_dict = {
                'loss_det': loss.item(),
                **tb_dict
            }

            return loss, tb_dict, disp_dict
            
        elif 'dom_img' in t_mode:
            loss_dom, tb_dict_dom = self.dense_head.get_dom_loss()
            
            loss = loss_dom #+ loss_dom_rcnn

            tb_dict_dom = {
                'loss_dom_img': loss.item(),
                **tb_dict_dom
            }

            return loss, tb_dict_dom, disp_dict

        elif 'dom_ins' in t_mode:
            loss_dom_rcnn, tb_dict_dom = self.roi_head.get_dom_loss()

            loss = loss_dom_rcnn

            tb_dict_dom = {
                'loss_dom_ins': loss.item(),
                **tb_dict_dom
            }

            return loss, tb_dict_dom, disp_dict