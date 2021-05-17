from .anchor_head_template import AnchorHeadTemplate
from .point_head_dom import PointHeadDom
from .point_head_box import PointHeadBox
from .point_head_box_dom import PointHeadBoxDom
from .point_head_simple import PointHeadSimple
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_single_range import AnchorHeadSingleRange
from .anchor_head_single_fpn import AnchorHeadSingleFPN
from .anchor_head_fuse import AnchorHeadFuse
from .anchor_head_fuse_range import AnchorHeadFuseRange
from .anchor_head_fuse_range_fpn import AnchorHeadFuseRangeFPN
from .anchor_head_multi import AnchorHeadMulti
from .point_intra_part_head import PointIntraPartOffsetHead
from .point_intra_part_head_fpn import PointIntraPartOffsetHeadFPN
from .point_head_simple import PointHeadSimple
from .anchor_head_fuse_context import AnchorHeadFuseContext
from .anchor_head_fuse_mcd import AnchorHeadFuseMCD
from .anchor_head_fuse_context_fpn import AnchorHeadFuseContextFPN
from .point_head_box_dom_context import PointHeadBoxDomContext
from .anchor_head_single_context_fpn import AnchorHeadSingleContextFPN
from .point_head_box_dom_range import PointHeadBoxDomRange
from .point_head_box_dom_mcd import PointHeadBoxDomMCD
from .point_head_box_dom_range_joint_sep import PointHeadBoxDomRangeJointSep
from .anchor_head_single_mcd import AnchorHeadSingleMCD
from .anchor_head_single_mcd_context import AnchorHeadSingleMCDContext
from .anchor_head_fuse_mcd_context import AnchorHeadFuseMCDContext
from .point_head_box_localdom import PointHeadBoxLocalDom
from .point_head_box_localdom_context import PointHeadBoxLocalDomContext
from .anchor_head_single_mcd_context_fpn import AnchorHeadSingleMCDContextFPN
from .anchor_head_single_range_guidance import AnchorHeadSingleRangeGuidance
from .anchor_head_single_fpn_range import AnchorHeadSingleFPNRange
from .anchor_head_fuse_fpn_combine import AnchorHeadFuseFPNCombine
from .anchor_head_single_context_fpn_strong_weak import AnchorHeadSingleContextFPNStrongWeak
from .anchor_head_single_context_fpn_weak import AnchorHeadSingleContextFPNWeak
from .anchor_head_fuse_fpn_combine_self_atten import AnchorHeadFuseFPNCombineSelfAttention
from .anchor_head_fuse_fpn_combine_cross_scale import AnchorHeadFuseFPNCombineCrossScale
from .anchor_head_fuse_fpn_combine_transformer import AnchorHeadFuseFPNCombineTransformer
from .anchor_head_single_CDN import AnchorHeadSingleCDN
__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'AnchorHeadSingleRange': AnchorHeadSingleRange,
    'AnchorHeadSingleFPN': AnchorHeadSingleFPN,
    'AnchorHeadFuse': AnchorHeadFuse,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointIntraPartOffsetHeadFPN': PointIntraPartOffsetHeadFPN,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadDom': PointHeadDom,
    'PointHeadBox': PointHeadBox,
    'PointHeadBoxDom': PointHeadBoxDom,
    'AnchorHeadMulti': AnchorHeadMulti,
    'AnchorHeadFuseContext': AnchorHeadFuseContext,
    'AnchorHeadSingleContextFPN': AnchorHeadSingleContextFPN,
    'AnchorHeadFuseContextFPN': AnchorHeadFuseContextFPN,
    'PointHeadBoxDomContext': PointHeadBoxDomContext,
    'PointHeadBoxDomRange': PointHeadBoxDomRange,
    'PointHeadBoxDomMCD': PointHeadBoxDomMCD,
    'AnchorHeadFuseRange': AnchorHeadFuseRange,
    'AnchorHeadFuseMCD': AnchorHeadFuseMCD,
    'PointHeadBoxDomRangeJointSep': PointHeadBoxDomRangeJointSep,
    'AnchorHeadSingleMCD':AnchorHeadSingleMCD,
    'AnchorHeadFuseRangeFPN': AnchorHeadFuseRangeFPN,
    'AnchorHeadSingleMCDContext': AnchorHeadSingleMCDContext,
    'PointHeadBoxLocalDom': PointHeadBoxLocalDom,
    'PointHeadBoxLocalDomContext': PointHeadBoxLocalDomContext,
    'AnchorHeadFuseMCDContext':
    AnchorHeadFuseMCDContext,
    'AnchorHeadSingleMCDContextFPN':AnchorHeadSingleMCDContextFPN,
    'AnchorHeadSingleRangeGuidance':AnchorHeadSingleRangeGuidance,
    'AnchorHeadSingleFPNRange': AnchorHeadSingleFPNRange,
    'AnchorHeadFuseFPNCombine':
    AnchorHeadFuseFPNCombine,
    'AnchorHeadSingleContextFPNStrongWeak': AnchorHeadSingleContextFPNStrongWeak,
    'AnchorHeadSingleContextFPNWeak':
    AnchorHeadSingleContextFPNWeak,
    'AnchorHeadFuseFPNCombineSelfAttention': AnchorHeadFuseFPNCombineSelfAttention,
    'AnchorHeadFuseFPNCombineCrossScale': AnchorHeadFuseFPNCombineCrossScale,
    'AnchorHeadSingleCDN': AnchorHeadSingleCDN,
    'AnchorHeadFuseFPNCombineTransformer': AnchorHeadFuseFPNCombineTransformer,
}
    # 'PointHeadBox': PointHeadBox,
