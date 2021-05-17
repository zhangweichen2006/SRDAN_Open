#!/bin/bash
set -e
# git stash
# git pull
#set -x
function optional {
    flag=$(test $1 = 'True' && echo true)
    echo ${flag:+$2}
}

GPU=0,1
#,1
#,2
GPUNUM=2
#2
#4
#2
#2
PORT=$RANDOM

MODEL=waymo_models
VER=second_waymo_center_val_car
#second_da_nounet_halfbost2halfsing_novelo_PMA-0.1lr0.5_RCD-0.1lr0.5_FPN_Up1Down1_NoShare_Large_2PMA_SepPMA
#second_waymo_center_val_car
# second_da_halfbost2halfsing_novelo_voxel-PMA_RCD_FPN
#pv_second_fuse_halfbost2halfsing_novelo_voxel-FPN_Up1Down1_interpolate-fast_NoShare_Large
#second_da_nounet_bost2sing_novelo_PMA-late-RCD-0.1lr0.5_FPN_Up1Down1_NoShare_Large

#second_da_nounet_bost2sing_novelo_PatchMatrixAttention-late-lr0.5_FPN_Up1Down1_NoShare_Large_TWOJOINTDOM
#second_da_nounet_bost2sing_novelo_PatchMatrixAttention-late-lr0.5_FPN_Up1Down1_NoShare_Large_TWOJOINTDOM
#_bost2sing_novelo_all
# second_da_halfbost2halfsing_novelo_PatchMatrixAttention-0.1lr1_RangeGuidanceDom-0.1lr1_JOINTTWODOM_PATCHATTLATE_NOGCTX_NOSIGMOID
# second_da_halfbost2halfsing_novelo_PMA-0.2lr0.5_RCD-0.2lr0.5_JOINTTWODOM_PATCHATTLATE
# second_da_halfbost2halfsing_novelo_PMA-0.2lr0.2_RCD-0.2lr0.2_JOINTTWODOM_PATCHATTLATE
# second_da_halfbost2halfsing_novelo_voxel-dom-patch-attention_range-context-dom-only_newconvdom0.1-lr1_nolocalattention_JOINTTWODOM_PATCHATTLATE
# second_da_halfbost2halfsing_novelo_range-context-dom-only_newconvdom0.2-lr1
#-pos
#second_da_halfbost2halfsing_novelo_RealMCD_tgt0.3c0l_context_nodafeat
#pv_second_fuse_halfbost2halfsing_novelo_localda
#_SE3_New
#pv_second_fuse_halfbost2halfsing_novelo_da_MCD_all0.1
#pv_second_fuse_halfbost2halfsing_novelo_da_context_0.3_512_balanced_range
#_context
# SYNC_BN=True
MCD=False
TSNE=False
MCD_CURVE=False
FPN_ONLY=False
EVAL_LAST=False
DRAW_MATRIX=False
CONTEXT=True
L_POW=10
DOM_ATTEN=False
FIX_RANDOM_SEED=False
RANGE=False
SEP_DA_NUM=1
SELECT_POSPROP=0.4
SELECT_NEGPROP=0.2
#True
#pv_second_fuse_halfbost2halfsing_novelo_da
#second_da_halfbost2halfsing_novelo_0.1dom_0.3_256_balanced_pseu
#_0.1dom
#
#minibost2minising_novelo
#bost2sing
#_pseudo
CONFIG=./tools/cfgs/${MODEL}/${VER}.yaml
# nusc
EP=50
MAX_EP=50
DA_W=0.3
L_Ratio=1.0
BATCH_SIZE=2
#$((${GPUNUM}))
TEST_BATCH_SIZE=2
# if (($BATCH_SIZE < 4)) || (($GPUNUM > 2)); then
#     SYNC_BN=True
# else
#     SYNC_BN=False
# fi
SYNC_BN=False
#$((${GPUNUM}))
PSEUDO_LABEL=False
PSEUDO_TRAIN_RATIO=0.5
DOUBLE_TEST=True
WORKERS=2
PIN_MEMORY=False
# SELECT_PROP=0.25
#.2
DEBUG=False
INST_DA=False
VIS=False
#${DA_W}daW_
MOD=nusc_${GPUNUM}GPU_${BATCH_SIZE}Batch_${MAX_EP}Ep_${PSEUDO_LABEL}pseudo_${PSEUDO_TRAIN_RATIO}ratio_${DOUBLE_TEST}2Test_Pos${SELECT_POSPROP}-Neg${SELECT_NEGPROP}SELECT_PROP_${L_POW}lpow

TASK_DESC=${VER}_${MOD}
#$(date +'%d-%m-%y')_${VER}_${MOD}

OUT_DIR=/home/wzha8158/CVPR_Outputs/${TASK_DESC}
#\(78.9\)25-10-2020_second_da_nounet_bost2sing_novelo_PMA-0.1lr0.5_RCD-0.1lr0.5_FPN_Up1Down1_NoShare_Large_nusc_4GPU_8Batch_50Ep_Falsepseudo_0.5ratio_False2Test_0.25SELECT_PROP_10L_POW_Falsemcd_curve_1.0L_Ratio
#${TASK_DESC}_0003/

CHECKPOINT=epoch_${TOTAL_EP}.pth
EVAL_ALL=True
TRAINVAL=True

OUTPUT_LOG=$OUT_DIR/log_${TASK_DESC}.txt
TEST_LOG=txt/eval_log_${TASK_DESC}.txt

mkdir -p $OUT_DIR

if [ ! $TASK_DESC ]
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

echo $0

cp $0 $OUT_DIR
cp -r pcdet $OUT_DIR
cp -r tools $OUT_DIR

mkdir -p txt

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

# rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_NewDet3D/Home/Det3D ~/Dropbox/CVPR2021_NewDet3D/Ti5/
# rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_NewDet3D/Home/Det3D ~/Dropbox/CVPR2021_NewDet3D/Ti5/

# python setup.py build develop
# rm -r build
rm -rf pcdet.egg-info
rm -rf ~/PCDet/lib/python3.6/site-packages/pcdet.egg-link
pip uninstall pcdet -y
python setup.py build develop

if (($GPUNUM > 1)); then
    set -x

    CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${GPUNUM} tools/traintest.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs $EP --out_dir=$OUT_DIR --test_batch_size $TEST_BATCH_SIZE --trainval ${TRAINVAL} --tcp_port=${PORT} --max_epochs ${MAX_EP} --pseudo_label ${PSEUDO_LABEL} --debug ${DEBUG} --pseu_train_ratio ${PSEUDO_TRAIN_RATIO} --double_test ${DOUBLE_TEST} --select_posprop ${SELECT_POSPROP}  --select_negprop ${SELECT_NEGPROP} --vis ${VIS} --pin_memory ${PIN_MEMORY} --workers ${WORKERS} --ins_da ${INST_DA}  --context ${CONTEXT} --fpn_only ${FPN_ONLY} --eval_last ${EVAL_LAST} --points_range ${RANGE}  --mcd ${MCD} --l_pow ${L_POW} --mcd_curve ${MCD_CURVE} $(optional $FIX_RANDOM_SEED --fix_random_seed) --draw_matrix ${DRAW_MATRIX} --draw_matrix ${DRAW_MATRIX}  --tsne $TSNE |& tee $OUTPUT_LOG
# --tsne $TSNE
else
    CUDA_VISIBLE_DEVICES=$GPU python -u tools/traintest.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs $EP --out_dir=$OUT_DIR --test_batch_size $TEST_BATCH_SIZE --trainval ${TRAINVAL} --tcp_port=${PORT} --max_epochs ${MAX_EP} --pseudo_label ${PSEUDO_LABEL} --debug ${DEBUG} --pseu_train_ratio ${PSEUDO_TRAIN_RATIO} --vis ${VIS} --double_test ${DOUBLE_TEST} --select_posprop ${SELECT_POSPROP}  --select_negprop ${SELECT_NEGPROP} --pin_memory ${PIN_MEMORY} --workers ${WORKERS} --ins_da ${INST_DA} --context ${CONTEXT} --fpn_only ${FPN_ONLY} --eval_last ${EVAL_LAST} --points_range ${RANGE}  --mcd ${MCD} --l_pow ${L_POW} --mcd_curve ${MCD_CURVE} --dom_atten ${DOM_ATTEN} $(optional $FIX_RANDOM_SEED --fix_random_seed) --draw_matrix ${DRAW_MATRIX} --tsne $TSNE|& tee $OUTPUT_LOG

    # CUDA_VISIBLE_DEVICES=$GPU python -u tools/test.py --cfg_file ${CONFIG} --batch_size 1 --eval_all |& tee -a $TEST_LOG
fi

cp $OUTPUT_LOG $OUT_DIR
cp $CONFIG $OUT_DIR
cp run_nusc_pvrcnn_da_midbost2sing.sh $OUT_DIR