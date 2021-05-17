import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackboneFPN(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        #####################################################

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = []
            layer_strides = []
            num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = []
            num_upsample_filters = []


        num_levels = len(layer_nums)
        # print("input_channels normal", input_channels)
        # print("*num_filters[:-1]", *num_filters[:-1])
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        #####################################################
        self.FPN = False
        input_channels_fpn = None
        layer_nums_fpn = {}
        layer_strides_fpn = {}
        num_filters_fpn = {}
        upsample_strides_fpn = {}
        num_upsample_filters_fpn = {}

        if self.model_cfg.get('NUM_BEV_FEATURES_FPN_UP', None) is not None:
            self.FPN = True
            input_channels_fpn_up = self.model_cfg.get('NUM_BEV_FEATURES_FPN_UP', [])
            assert len(self.model_cfg.LAYER_NUMS_FPN_UP) == len(self.model_cfg.LAYER_STRIDES_FPN_UP) == len(self.model_cfg.NUM_FILTERS_FPN_UP)
            
            for l in range(len(self.model_cfg.LAYER_NUMS_FPN_UP)):
                layer = str(3 - l)
                layer_nums_fpn[layer] = self.model_cfg.LAYER_NUMS_FPN_UP[l]
                layer_strides_fpn[layer] = self.model_cfg.LAYER_STRIDES_FPN_UP[l]
                num_filters_fpn[layer] = self.model_cfg.NUM_FILTERS_FPN_UP[l]

        if self.model_cfg.get('NUM_BEV_FEATURES_FPN_DOWN', None) is not None:
            self.FPN = True
            input_channels_fpn_down = self.model_cfg.get('NUM_BEV_FEATURES_FPN_DOWN', [])
            assert len(self.model_cfg.LAYER_NUMS_FPN_DOWN) == len(self.model_cfg.LAYER_STRIDES_FPN_DOWN) == len(self.model_cfg.NUM_FILTERS_FPN_DOWN)
            
            for l in range(len(self.model_cfg.LAYER_NUMS_FPN_DOWN)):
                layer = str(4 + 1+ l)
                layer_nums_fpn[layer] = self.model_cfg.LAYER_NUMS_FPN_DOWN[l]
                layer_strides_fpn[layer] = self.model_cfg.LAYER_STRIDES_FPN_DOWN[l]
                num_filters_fpn[layer] = self.model_cfg.NUM_FILTERS_FPN_DOWN[l]

        if self.model_cfg.get('NUM_BEV_FEATURES_FPN_DOWNUP', None) is not None:
            self.FPN = True
            input_channels_fpn_downup = self.model_cfg.get('NUM_BEV_FEATURES_FPN_DOWNUP', [])
            assert len(self.model_cfg.LAYER_NUMS_FPN_DOWNUP) == len(self.model_cfg.LAYER_STRIDES_FPN_DOWNUP) == len(self.model_cfg.NUM_FILTERS_FPN_DOWNUP)
            
            for l in range(len(self.model_cfg.LAYER_NUMS_FPN_DOWNUP)):
                layer = str(4 + l)
                layer_nums_fpn[layer] = self.model_cfg.LAYER_NUMS_FPN_DOWNUP[l]
                layer_strides_fpn[layer] = self.model_cfg.LAYER_STRIDES_FPN_DOWNUP[l]
                num_filters_fpn[layer] = self.model_cfg.NUM_FILTERS_FPN_DOWNUP[l]

        if self.model_cfg.get('UPSAMPLE_STRIDES_FPN_UP', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES_FPN_UP) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS_FPN_UP)

            for l in range(len(self.model_cfg.UPSAMPLE_STRIDES_FPN_UP)):
                layer = str(3 - l)
                upsample_strides_fpn[layer] = self.model_cfg.UPSAMPLE_STRIDES_FPN_UP[l]
                num_upsample_filters_fpn[layer] = self.model_cfg.NUM_UPSAMPLE_FILTERS_FPN_UP[l]

        if self.model_cfg.get('UPSAMPLE_STRIDES_FPN_DOWN', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES_FPN_DOWN) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS_FPN_DOWN)
            
            for l in range(len(self.model_cfg.UPSAMPLE_STRIDES_FPN_DOWN)):
                layer = str(4 + 1+ l)
                upsample_strides_fpn[layer] = self.model_cfg.UPSAMPLE_STRIDES_FPN_DOWN[l]
                num_upsample_filters_fpn[layer] = self.model_cfg.NUM_UPSAMPLE_FILTERS_FPN_DOWN[l]

        if self.model_cfg.get('UPSAMPLE_STRIDES_FPN_DOWNUP', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES_FPN_DOWNUP) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS_FPN_DOWNUP)
            
            for l in range(len(self.model_cfg.UPSAMPLE_STRIDES_FPN_DOWNUP)):
                layer = str(4 + l)
                upsample_strides_fpn[layer] = self.model_cfg.UPSAMPLE_STRIDES_FPN_DOWNUP[l]
                num_upsample_filters_fpn[layer] = self.model_cfg.NUM_UPSAMPLE_FILTERS_FPN_DOWNUP[l]

        ####################################

        if self.FPN:
            self.blocks_FPN = nn.ModuleDict()
            self.deblocks_FPN = nn.ModuleDict()

            self.num_bev_features_fpn = {}

            for l in range(len(self.model_cfg.NUM_BEV_FEATURES_FPN_UP)):
                layer = str(3 - l)
                num_levels = len(layer_nums_fpn[layer])
                c_in_list = [input_channels_fpn_up[l], *num_filters_fpn[layer][:-1]]

                for idx in range(num_levels):
                    cur_layers = [
                        nn.ZeroPad2d(1),
                        nn.Conv2d(
                            c_in_list[idx], num_filters_fpn[layer][idx], kernel_size=3,
                            stride=layer_strides_fpn[layer][idx], padding=0, bias=False
                        ),
                        nn.BatchNorm2d(num_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ]
                    for k in range(layer_nums_fpn[layer][idx]):
                        cur_layers.extend([
                            nn.Conv2d(num_filters_fpn[layer][idx], num_filters_fpn[layer][idx], kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                            nn.ReLU()
                        ])
                    if layer in self.blocks_FPN.keys():
                        self.blocks_FPN[layer].append(nn.Sequential(*cur_layers))
                    else:
                        self.blocks_FPN[layer] = nn.ModuleList().append(nn.Sequential(*cur_layers))

                    if len(upsample_strides_fpn[layer]) > 0:
                        stride = upsample_strides_fpn[layer][idx]
                        if stride >= 1:
                            cur_layers_deblock = nn.Sequential(
                                nn.ConvTranspose2d(
                                    num_filters_fpn[layer][idx], num_upsample_filters_fpn[layer][idx],
                                    upsample_strides_fpn[layer][idx],
                                    stride=upsample_strides_fpn[layer][idx], bias=False
                                ),
                                nn.BatchNorm2d(num_upsample_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                                nn.ReLU()
                            )
                        else:
                            stride = np.round(1 / stride).astype(np.int)
                            cur_layers_deblock = nn.Sequential(
                                nn.Conv2d(
                                    num_filters_fpn[layer][idx], num_upsample_filters_fpn[layer][idx],
                                    stride,
                                    stride=stride, bias=False
                                ),
                                nn.BatchNorm2d(num_upsample_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                                nn.ReLU()
                            )
                        
                        if layer in self.deblocks_FPN.keys():
                            self.deblocks_FPN[layer].append(nn.Sequential(*cur_layers_deblock))
                        else:
                            self.deblocks_FPN[layer] = nn.ModuleList().append(nn.Sequential(*cur_layers_deblock))

                c_in = sum(num_upsample_filters_fpn[layer])
                if len(upsample_strides_fpn[layer]) > num_levels:
                    cur_layers_up = nn.Sequential(
                        nn.ConvTranspose2d(c_in, c_in, upsample_strides_fpn[layer][-1], stride=upsample_strides_fpn[layer][-1], bias=False),
                        nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    )

                    if layer in self.deblocks_FPN.keys():
                        self.deblocks_FPN[layer].append(cur_layers_up)
                    else:
                        self.deblocks_FPN[layer] = nn.ModuleList().append(cur_layers_up)

                self.num_bev_features_fpn[layer] = c_in
            
            # print('input_channels_fpn_downup', input_channels_fpn_downup)
            for l in range(len(self.model_cfg.get('NUM_BEV_FEATURES_FPN_DOWN', []))):
                layer = str(4 + 1 + l)
                num_levels = len(layer_nums_fpn[layer])
                c_in_list = [input_channels_fpn_down[l], *num_filters_fpn[layer][:-1]]

                for idx in range(num_levels):
                    cur_layers = [
                        nn.ZeroPad2d(1),
                        nn.Conv2d(
                            c_in_list[idx], num_filters_fpn[layer][idx], kernel_size=3,
                            stride=layer_strides_fpn[layer][idx], padding=0, bias=False
                        ),
                        nn.BatchNorm2d(num_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ]
                    for k in range(layer_nums_fpn[layer][idx]):
                        cur_layers.extend([
                            nn.Conv2d(num_filters_fpn[layer][idx], num_filters_fpn[layer][idx], kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                            nn.ReLU()
                        ])
                    if layer in self.blocks_FPN.keys():
                        self.blocks_FPN[layer].append(nn.Sequential(*cur_layers))
                    else:
                        self.blocks_FPN[layer] = nn.ModuleList().append(nn.Sequential(*cur_layers))
                        
                    if len(upsample_strides_fpn[layer]) > 0:
                        stride = upsample_strides_fpn[layer][idx]
                        if stride >= 1:
                            cur_layers_deblock = nn.Sequential(
                                nn.ConvTranspose2d(
                                    num_filters_fpn[layer][idx], num_upsample_filters_fpn[layer][idx],
                                    upsample_strides_fpn[layer][idx],
                                    stride=upsample_strides_fpn[layer][idx], bias=False
                                ),
                                nn.BatchNorm2d(num_upsample_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                                nn.ReLU()
                            )
                        else:
                            stride = np.round(1 / stride).astype(np.int)
                            cur_layers_deblock = nn.Sequential(
                                nn.Conv2d(
                                    num_filters_fpn[layer][idx], num_upsample_filters_fpn[layer][idx],
                                    stride,
                                    stride=stride, bias=False
                                ),
                                nn.BatchNorm2d(num_upsample_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                                nn.ReLU()
                            )
                        
                        if layer in self.deblocks_FPN.keys():
                            self.deblocks_FPN[layer].append(nn.Sequential(*cur_layers_deblock))
                        else:
                            self.deblocks_FPN[layer] = nn.ModuleList().append(nn.Sequential(*cur_layers_deblock))


                c_in = sum(num_upsample_filters_fpn[layer])
                if len(upsample_strides_fpn[layer]) > num_levels:
                    cur_layers_up = nn.Sequential(
                        nn.ConvTranspose2d(c_in, c_in, upsample_strides_fpn[layer][-1], stride=upsample_strides_fpn[layer][-1], bias=False),
                        nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    )

                    if layer in self.deblocks_FPN.keys():
                        self.deblocks_FPN[layer].append(cur_layers_up)
                    else:
                        self.deblocks_FPN[layer] = nn.ModuleList().append(cur_layers_up)

                self.num_bev_features_fpn[layer] = c_in
                
            for l in range(len(self.model_cfg.NUM_BEV_FEATURES_FPN_DOWNUP)):
                layer = str(4 + l)
                num_levels = len(layer_nums_fpn[layer])
                c_in_list = [input_channels_fpn_downup[l], *num_filters_fpn[layer][:-1]]

                for idx in range(num_levels):
                    cur_layers = [
                        nn.ZeroPad2d(1),
                        nn.Conv2d(
                            c_in_list[idx], num_filters_fpn[layer][idx], kernel_size=3,
                            stride=layer_strides_fpn[layer][idx], padding=0, bias=False
                        ),
                        nn.BatchNorm2d(num_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ]
                    for k in range(layer_nums_fpn[layer][idx]):
                        cur_layers.extend([
                            nn.Conv2d(num_filters_fpn[layer][idx], num_filters_fpn[layer][idx], kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                            nn.ReLU()
                        ])
                    if layer in self.blocks_FPN.keys():
                        self.blocks_FPN[layer].append(nn.Sequential(*cur_layers))
                    else:
                        self.blocks_FPN[layer] = nn.ModuleList().append(nn.Sequential(*cur_layers))
                        
                    if len(upsample_strides_fpn[layer]) > 0:
                        stride = upsample_strides_fpn[layer][idx]
                        if stride >= 1:
                            cur_layers_deblock = nn.Sequential(
                                nn.ConvTranspose2d(
                                    num_filters_fpn[layer][idx], num_upsample_filters_fpn[layer][idx],
                                    upsample_strides_fpn[layer][idx],
                                    stride=upsample_strides_fpn[layer][idx], bias=False
                                ),
                                nn.BatchNorm2d(num_upsample_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                                nn.ReLU()
                            )
                        else:
                            stride = np.round(1 / stride).astype(np.int)
                            cur_layers_deblock = nn.Sequential(
                                nn.Conv2d(
                                    num_filters_fpn[layer][idx], num_upsample_filters_fpn[layer][idx],
                                    stride,
                                    stride=stride, bias=False
                                ),
                                nn.BatchNorm2d(num_upsample_filters_fpn[layer][idx], eps=1e-3, momentum=0.01),
                                nn.ReLU()
                            )
                        
                        if layer in self.deblocks_FPN.keys():
                            self.deblocks_FPN[layer].append(nn.Sequential(*cur_layers_deblock))
                        else:
                            self.deblocks_FPN[layer] = nn.ModuleList().append(nn.Sequential(*cur_layers_deblock))

                c_in = sum(num_upsample_filters_fpn[layer])
                if len(upsample_strides_fpn[layer]) > num_levels:
                    cur_layers_up = nn.ModuleList(nn.Sequential(
                        nn.ConvTranspose2d(c_in, c_in, upsample_strides_fpn[layer][-1], stride=upsample_strides_fpn[layer][-1], bias=False),
                        nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ))

                    if layer in self.deblocks_FPN.keys():
                        self.deblocks_FPN[layer].append(cur_layers_up)
                    else:
                        self.deblocks_FPN[layer] = nn.ModuleList().append(cur_layers_up)

                self.num_bev_features_fpn[layer] = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        # 512, 126, 126

        # print('data_dict', data_dict)
        if self.FPN:
            for layer in self.blocks_FPN.keys():
                spatial_features = data_dict[f'spatial_features_fpn{layer}']
                ups = []
                ret_dict = {}
                x = spatial_features
                # layer3, 640, 252, 252
                # layer4, 256, 126, 126
                # layer5, 128, 63, 63(4)
                for i in range(len(self.blocks_FPN[layer])):
                    x = self.blocks_FPN[layer][i](x)
                    
                    stride = int(spatial_features.shape[2] / x.shape[2])
                    ret_dict[f'spatial_features_fpn{layer}_{stride}x'] = x
                    if len(self.deblocks_FPN[layer]) > 0:
                        ups.append(self.deblocks_FPN[layer][i](x))
                    else:
                        ups.append(x)
                        
                if len(ups) > 1:
                    x = torch.cat(ups, dim=1)
                elif len(ups) == 1:
                    x = ups[0]

                if len(self.deblocks_FPN[layer]) > len(self.blocks_FPN[layer]):
                    x = self.deblocks_FPN[layer][-1](x)

                data_dict[f'spatial_features_2d_fpn{layer}'] = x
                # print("spatial_features_2d_fpn fpn", data_dict[f'spatial_features_2d_fpn{layer}'].shape)
                # 512, 252, 252

            # for l in range(len(self.blocks_FPN_DOWNUP)):
            #     layer = str(3 - l)

        return data_dict
