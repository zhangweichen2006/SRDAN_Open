#!/bin/bash 
set -e 
# git stash
# git pull
#set -x

GPU=0
#,2
GPUNUM=1
#2
PORT=$RANDOM

VER=second_waymo
#_sep2da
#pv_second_fuse_halfbost2halfsing_novelo_da_SE3_New
# second_da_halfbost2halfsing_novelo_RealMCD_tgt0.2c0l
# pv_second_fuse_halfbost2halfsing_novelo_noda_interpolate
#pv_second_fuse_halfbost2halfsing_novelo_da_MCD
#_FPN4Only
#pv_second_fuse_halfbost2halfsing_novelo_da_context_0.3_512_balanced_range
#_context
L_POW=10
MCD=False
MCD_CURVE=False
FPN_ONLY=False
CONTEXT=False
DOM_ATTEN=False
EVAL_LAST=False
RANGE=False
SEP_DA_NUM=1
#2
#True
#pv_second_fuse_halfbost2halfsing_novelo_da
#second_da_halfbost2halfsing_novelo_0.1dom_0.3_256_balanced_pseu
#_0.1dom
#
#minibost2minising_novelo
#bost2sing
#_pseudo
CONFIG=./tools/cfgs/waymo_models/${VER}.yaml

EP=50
MAX_EP=50
DA_W=0.3
L_Ratio=1.0
BATCH_SIZE=4
#$((${GPUNUM}))
TEST_BATCH_SIZE=4
#$((${GPUNUM}))
PSEUDO_LABEL=False
PSEUDO_TRAIN_RATIO=0.5
DOUBLE_TEST=True
WORKERS=0
PIN_MEMORY=True
SELECT_PROP=0.25
#.2
DEBUG=False
INST_DA=False
VIS=False
#${DA_W}daW_
MOD=nusc_${GPUNUM}GPU_${BATCH_SIZE}Batch_${MAX_EP}Ep_${PSEUDO_LABEL}pseudo_${PSEUDO_TRAIN_RATIO}ratio_${DOUBLE_TEST}2Test_${SELECT_PROP}SELECT_PROP_${L_POW}L_POW_${MCD_CURVE}mcd_curve

TASK_DESC=$(date +'%d-%m-%y')_${VER}_${MOD}
#$(date +'%d-%m-%y')_${VER}_${MOD}

OUT_DIR=/home/wzha8158/Lidar_Outputs/$TASK_DESC

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

    CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${GPUNUM} tools/traintest.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs $EP --out_dir=$OUT_DIR --test_batch_size $TEST_BATCH_SIZE --trainval ${TRAINVAL} --tcp_port=${PORT} --max_epochs ${MAX_EP} --pseudo_label ${PSEUDO_LABEL} --debug ${DEBUG} --pseu_train_ratio ${PSEUDO_TRAIN_RATIO} --vis ${VIS} --double_test ${DOUBLE_TEST} --select_prop ${SELECT_PROP} --pin_memory ${PIN_MEMORY} --workers ${WORKERS} --ins_da ${INST_DA} --context ${CONTEXT} --fpn_only ${FPN_ONLY} --eval_last ${EVAL_LAST}  --points_range ${RANGE} --mcd ${MCD} --l_pow ${L_POW} --mcd_curve ${MCD_CURVE} --dom_atten ${DOM_ATTEN} |& tee $OUTPUT_LOG
# 
    # CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${NGPUS} tools/test.py --launcher pytorch --cfg_file ${CONFIG} --batch_size 1 --eval_all |& tee -a $TEST_LOG

else
    CUDA_VISIBLE_DEVICES=$GPU python -u tools/traintest.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs $EP --out_dir=$OUT_DIR --test_batch_size $TEST_BATCH_SIZE --trainval ${TRAINVAL} --tcp_port=${PORT} --max_epochs ${MAX_EP} --pseudo_label ${PSEUDO_LABEL} --debug ${DEBUG} --pseu_train_ratio ${PSEUDO_TRAIN_RATIO} --vis ${VIS} --double_test ${DOUBLE_TEST} --select_prop ${SELECT_PROP} --pin_memory ${PIN_MEMORY} --workers ${WORKERS} --ins_da ${INST_DA} --context ${CONTEXT} --fpn_only ${FPN_ONLY} --eval_last ${EVAL_LAST}  --points_range ${RANGE} --mcd ${MCD} --l_pow ${L_POW} --mcd_curve ${MCD_CURVE} --dom_atten ${DOM_ATTEN} |& tee $OUTPUT_LOG

    # CUDA_VISIBLE_DEVICES=$GPU python -u tools/test.py --cfg_file ${CONFIG} --batch_size 1 --eval_all |& tee -a $TEST_LOG
fi

cp $OUTPUT_LOG $OUT_DIR 
cp $CONFIG $OUT_DIR
cp run_nusc_pvrcnn_da_midbost2sing.sh $OUT_DIR