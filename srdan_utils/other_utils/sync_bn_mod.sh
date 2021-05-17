nano /home/wzha8158/PCDet/lib/python3.6/site-packages/torch/nn/modules/batchnorm.py

ctrl w: ddp

def _check_input_dim(self, input):
        # if input.dim() <= 2:
        #     raise ValueError('expected at least 3D input (got {}D input)'
        #                      .format(input.dim()))
        if input.dim() < 2:  # modified to support input (B, C)
            raise ValueError('expected at least 2D input (got {}D input)'
                               .format(input.dim()))