import torch.nn as nn


class HeightCompressionFPNStrongWeak(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_fpn_up = self.model_cfg.get('NUM_BEV_FEATURES_FPN_UP',[])
        self.num_bev_features_fpn_down = self.model_cfg.get('NUM_BEV_FEATURES_FPN_DOWN',[])
        self.num_bev_features_fpn_downup = self.model_cfg.get('NUM_BEV_FEATURES_FPN_DOWNUP',[])

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        layers = [str(3-l) for l in range(len(self.num_bev_features_fpn_up))] + [str(4 + 1 + l) for l in range(len(self.num_bev_features_fpn_down))] \
                  + [str(4 + l) for l in range(len(self.num_bev_features_fpn_downup))]
        # print("layers", layers)
        # for layer in layers:
        #     encoded_spconv_tensor = batch_dict[f'encoded_spconv_tensor_fpn{layer}']
        #     spatial_features = encoded_spconv_tensor.dense()
        #     N, C, D, H, W = spatial_features.shape
        #     spatial_features = spatial_features.view(N, C * D, H, W)
        #     batch_dict[f'spatial_features_fpn{layer}'] = spatial_features
        #     batch_dict[f'spatial_features_stride_fpn{layer}'] = batch_dict[f'encoded_spconv_tensor_stride_fpn{layer}']

        batch_dict['spatial_features_multi'] = {}
        batch_dict['spatial_features_stride_multi'] = {}

        # print('multi_scale_3d_features', batch_dict['multi_scale_3d_features'])
        # for k, v in batch_dict['multi_scale_3d_features'].items():

            # k ['x_conv2']
        encoded_spconv_tensor = batch_dict['multi_scale_3d_features']['x_conv2']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        # if k in batch_dict['spatial_features_multi']:
        #     batch_dict['spatial_features_multi'][k] = spatial_features
        # else:

        encoded_spconv_tensor2 = batch_dict['multi_scale_3d_features']['x_conv4']
        spatial_features2 = encoded_spconv_tensor2.dense()
        N2, C2, D2, H2, W2 = spatial_features2.shape
        spatial_features2 = spatial_features2.view(N2, C2 * D2, H2, W2)

        batch_dict['spatial_features_multi']['x_conv2'] = spatial_features
        batch_dict['spatial_features_stride_multi']['x_conv2'] = batch_dict['encoded_spconv_tensor_stride']

        batch_dict['spatial_features_multi']['x_conv4'] = spatial_features2
        batch_dict['spatial_features_stride_multi']['x_conv4'] = batch_dict['encoded_spconv_tensor_stride']

        return batch_dict
