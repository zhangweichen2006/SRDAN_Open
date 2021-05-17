#!/bin/bash
set -e
# git stash
# git pull
#set -x

GPU=0,1,2,3
#,2,3
# ,2,3
#,2
#,2
GPUNUM=4
#2
PORT=$RANDOM

VER=second_da_bost2sing_novelo_RCD-0.1lr1-RangeDist
#second_da_bost2sing_novelo_FPN5_PMA_large_adam-0003
#second_da_nounet_sing2bost_novelo_PMA-late-0.1lr0.5_RCD-0.1lr0.5_FPN_Up1Down1_NoShare_Large_TWOJOINTDOM
#second_da_nounet_halfbost2halfsing_novelo_PatchMatrixAttention-late-lr0.1_RangeGuidanceDom_FPN_Up1Down1-lr1_NoShare_Large_TWOJOINTDOM

L_POW=10
MCD=False
MCD_CURVE=False
FPN_ONLY=True
CONTEXT=False
DRAW_MATRIX=True
DOM_ATTEN=False
EVAL_LAST=False
RANGE=False
SEP_DA_NUM=1
#_pseudo
CONFIG=./tools/cfgs/nusc_models/${VER}.yaml

EP=50
MAX_EP=50

DA_W=0.3
L_Ratio=1.0
BATCH_SIZE=8
#$((${GPUNUM}))
TEST_BATCH_SIZE=8
#$((${GPUNUM}))
PSEUDO_LABEL=False
PSEUDO_TRAIN_RATIO=0.5
DOUBLE_TEST=True
WORKERS=4
PIN_MEMORY=True
SELECT_PROP=0.25
#.2
DEBUG=False
INST_DA=False
VIS=False
MOD=nusc_2GPU_4Batch_${MAX_EP}Ep_${PSEUDO_LABEL}pseudo_${PSEUDO_TRAIN_RATIO}ratio_${DOUBLE_TEST}2Test_${SELECT_PROP}SELECT_PROP_${L_POW}L_POW_${MCD_CURVE}mcd_curve_${L_Ratio}L_Ratio
#${BATCH_SIZE}
TASK_DESC=${VER}_${MOD}
#$(date +'%d-%m-%y')_${VER}_${MOD}

OUT_DIR=/root/Lidar_Outputs/$TASK_DESC

CHECKPOINT=epoch_${TOTAL_EP}.pth
EVAL_ALL=True
TRAINVAL=True

OUTPUT_LOG=txt/log_${TASK_DESC}.txt
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
# cp -r pcdet $OUT_DIR

mkdir -p txt

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

# rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_NewDet3D/Home/Det3D ~/Dropbox/CVPR2021_NewDet3D/Ti5/
# rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_NewDet3D/Home/Det3D ~/Dropbox/CVPR2021_NewDet3D/Ti5/

# python setup.py build develop
# rm -r build
pip uninstall pcdet -y
rm -rf pcdet.egg-info
python setup.py build develop

if (($GPUNUM > 1)); then
    set -x

    CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${GPUNUM} tools/traintest.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs $EP --out_dir=$OUT_DIR --test_batch_size $TEST_BATCH_SIZE --trainval ${TRAINVAL} --tcp_port=${PORT} --max_epochs ${MAX_EP} --pseudo_label ${PSEUDO_LABEL} --debug ${DEBUG} --pseu_train_ratio ${PSEUDO_TRAIN_RATIO} --vis ${VIS} --double_test ${DOUBLE_TEST} --select_prop ${SELECT_PROP} --pin_memory ${PIN_MEMORY} --workers ${WORKERS} --ins_da ${INST_DA} --context ${CONTEXT} --fpn_only ${FPN_ONLY} --eval_last ${EVAL_LAST}  --points_range ${RANGE} --mcd ${MCD} --l_pow ${L_POW} --mcd_curve ${MCD_CURVE} --dom_atten ${DOM_ATTEN}  --draw_matrix ${DRAW_MATRIX}|& tee $OUTPUT_LOG
#
    # CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${NGPUS} tools/test.py --launcher pytorch --cfg_file ${CONFIG} --batch_size 1 --eval_all |& tee -a $TEST_LOG

else
    CUDA_VISIBLE_DEVICES=$GPU python -u tools/traintest.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs $EP --out_dir=$OUT_DIR --test_batch_size $TEST_BATCH_SIZE --trainval ${TRAINVAL} --tcp_port=${PORT} --max_epochs ${MAX_EP} --pseudo_label ${PSEUDO_LABEL} --debug ${DEBUG} --pseu_train_ratio ${PSEUDO_TRAIN_RATIO} --vis ${VIS} --double_test ${DOUBLE_TEST} --select_prop ${SELECT_PROP} --pin_memory ${PIN_MEMORY} --workers ${WORKERS} --ins_da ${INST_DA} --context ${CONTEXT} --fpn_only ${FPN_ONLY} --eval_last ${EVAL_LAST}  --points_range ${RANGE} --mcd ${MCD} --l_pow ${L_POW} --mcd_curve ${MCD_CURVE} --dom_atten ${DOM_ATTEN}  --draw_matrix ${DRAW_MATRIX}|& tee $OUTPUT_LOG

    # CUDA_VISIBLE_DEVICES=$GPU python -u tools/test.py --cfg_file ${CONFIG} --batch_size 1 --eval_all |& tee -a $TEST_LOG
fi

cp $OUTPUT_LOG $OUT_DIR
cp $CONFIG $OUT_DIR
cp run_nusc_pvrcnn_da_midbost2sing.sh $OUT_DIR