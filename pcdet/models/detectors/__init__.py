from .detector3d_template import Detector3DTemplate
from .second_net import SECONDNet
from .pv_second_net import PVSECONDNet
from .pv_second_net_mcd import PVSECONDNetMCD
from .pv_second_fpn_net import PVSECONDFPNNet
from .pv_second_net_range import PVSECONDNetRange
from .pv_second_net_range_joint_sep import PVSECONDNetRangeJointSep
from .pv_second_net_range_joint_sep_fpn import PVSECONDNetRangeJointSepFPN
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_fpn_net import SECONDFPNNet
from .second_fpn_nounet import SECONDFPNNOUNet
from .second_fpn_netonly import SECONDFPNNetOnly
from .second_fpn_nounetonly import SECONDFPNNOUNetOnly
from .second_hrnet_fpn import SECONDHRNetFPN
from .second_hrnet import SECONDHRNet
from .second_net_range_inv import SECONDNetRangeInv
from .second_net_mcd import SECONDNetMCD
from .second_net_mcd_context import SECONDNetMCDContext
from .pv_second_net_mcd_context import PVSECONDNetMCDContext

from .second_net_mcd_context_fpn import SECONDNetMCDContextFPN

from .second_net_tsne import SECONDNetTSNE

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PVSECONDNet': PVSECONDNet,
    'PVSECONDNetMCD': PVSECONDNetMCD,
    'PVSECONDFPNNet': PVSECONDFPNNet,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDFPNNet': SECONDFPNNet,
    'SECONDFPNNetOnly': SECONDFPNNetOnly,
    'SECONDFPNNOUNet': SECONDFPNNOUNet, 
    'SECONDFPNNOUNetOnly': SECONDFPNNOUNetOnly,
    'SECONDHRNetFPN': SECONDHRNetFPN,
    'SECONDHRNet': SECONDHRNet,
    'PVSECONDNetRange': PVSECONDNetRange,
    'PVSECONDNetRangeJointSep': PVSECONDNetRangeJointSep,
    'SECONDNetRangeInv': SECONDNetRangeInv,
    'SECONDNetMCD': SECONDNetMCD,
    'PVSECONDNetRangeJointSepFPN': PVSECONDNetRangeJointSepFPN,
    'SECONDNetMCDContext': SECONDNetMCDContext,
    'PVSECONDNetMCDContext': PVSECONDNetMCDContext,
    'SECONDNetMCDContextFPN':SECONDNetMCDContextFPN,
    'SECONDNetTSNE':SECONDNetTSNE,
}


def build_detector(model_cfg, num_class, dataset, nusc=False):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, nusc=nusc
    )

    return model
