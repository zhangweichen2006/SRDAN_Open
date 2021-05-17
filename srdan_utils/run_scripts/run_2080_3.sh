#!/bin/bash
set -e
# git stash
# git pull
#set -x
function optional {
    flag=$(test $1 = 'True' && echo true)
    echo ${flag:+$2}
}

GPU=0,1,2,3
#,1
#,3
#,1
#,3
#,1,2,3
GPUNUM=4
#2
#2
#4
#2
PORT=$RANDOM
VER=second_da_nounet_halfbost2halfsing_novelo_CDN
# _PMA-0.1lr0.5_RCD-0.1lr0.5_FPN
# second_da_nounet_halfbost2halfsing_novelo_NoShare_Large_weak
# second_da_nounet_sing2bost_novelo_PMA-0.1lr0.5_RCD-0.1lr0.5_FPN_Up1Down1_NoShare_Large_SRDAN_0.25loc
#_allclass
#_all
# second_da_nounet_bost2sing_novelo_RCDlr0.5_FPN_Up1Down1_NoShare_Large_TWOJOINTDOM
#_Large
#pv_second_fuse_halfbost2halfsing_novelo_PatchMatrixAttention_RangeGuidanceDom-lr1_FPN_U1D1_interp
# second_da_halfbost2halfsing_novelo_voxel-dom-patch-attention_range-context-dom-only_newconvdom0.2-lr1
# SYNC_BN=True
L_POW=10
ADD_INFO=''
CONTEXT=False
FPN_ONLY=False
EVAL_LAST=False
DRAW_MATRIX=False
MCD=False
MCD_CURVE=False
RANGE=False
VOXEL_ATTENTION=False
FIX_RANDOM_SEED=False
CONFIG=./tools/cfgs/cvpr_models/${VER}.yaml

EP=50
MAX_EP=50
L_Ratio=1.0
BATCH_SIZE=4
#4
#$((${GPUNUM}))
TEST_BATCH_SIZE=4
# if (($BATCH_SIZE < 4)) || (($GPUNUM > 2)); then
#     SYNC_BN=True
# else
SYNC_BN=False
# fi
DOUBLE_TEST=False
#$((${GPUNUM}))
PSEUDO_LABEL=False
PSEUDO_TRAIN_RATIO=0.5
#.2
DEBUG=False
VIS=False
WORKERS=2
PIN_MEMORY=True
INST_DA=False
SELECT_POSPROP=0.3
SELECT_NEGPROP=0.1
#${DA_W}daW_
MOD=nusc_dann_${PSEUDO_LABEL}pseudo_reinit_${PSEUDO_TRAIN_RATIO}ratio_${GPU}GPU_${BATCH_SIZE}Batch_${SELECT_POSPROP}:${SELECT_NEGPROP}PseuProp

TASK_DESC=${VER}_${MOD}_CVPR_NEWFCFULLY

#$(date +'%d-%m-%Y')_${VER}_${MOD}
#$(date +'%d-%m-%Y')
OUT_DIR=/home/wzha8158/CVPR_Outputs/$TASK_DESC

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
cp -r pcdet $OUT_DIR

mkdir -p txt
cp run_Ti5.sh $OUT_DIR

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

	if [ $SYNC_BN = 'True' ]; then
        echo 'USE SYNC BN'
        CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${GPUNUM} tools/traintest.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs $EP --out_dir=$OUT_DIR --test_batch_size $TEST_BATCH_SIZE --trainval ${TRAINVAL} --tcp_port=${PORT} --max_epochs ${MAX_EP} --pseudo_label ${PSEUDO_LABEL} --debug ${DEBUG} --pseu_train_ratio ${PSEUDO_TRAIN_RATIO} --double_test ${DOUBLE_TEST} --select_posprop ${SELECT_POSPROP}  --select_negprop ${SELECT_NEGPROP} --vis ${VIS} --pin_memory ${PIN_MEMORY} --workers ${WORKERS} --ins_da ${INST_DA}  --context ${CONTEXT} --fpn_only ${FPN_ONLY} --eval_last ${EVAL_LAST} --points_range ${RANGE}  --mcd ${MCD} --l_pow ${L_POW} --mcd_curve ${MCD_CURVE} --sync_bn $(optional $FIX_RANDOM_SEED --fix_random_seed) --draw_matrix ${DRAW_MATRIX}|& tee $OUTPUT_LOG
    else
        CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${GPUNUM} tools/traintest.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs $EP --out_dir=$OUT_DIR --test_batch_size $TEST_BATCH_SIZE --trainval ${TRAINVAL} --tcp_port=${PORT} --max_epochs ${MAX_EP} --pseudo_label ${PSEUDO_LABEL} --debug ${DEBUG} --pseu_train_ratio ${PSEUDO_TRAIN_RATIO} --double_test ${DOUBLE_TEST} --select_posprop ${SELECT_POSPROP}  --select_negprop ${SELECT_NEGPROP} --vis ${VIS} --pin_memory ${PIN_MEMORY} --workers ${WORKERS} --ins_da ${INST_DA}  --context ${CONTEXT} --fpn_only ${FPN_ONLY} --eval_last ${EVAL_LAST} --points_range ${RANGE}  --mcd ${MCD} --l_pow ${L_POW} --mcd_curve ${MCD_CURVE} $(optional $FIX_RANDOM_SEED --fix_random_seed) --draw_matrix ${DRAW_MATRIX}|& tee $OUTPUT_LOG
    fi

    # CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${NGPUS} tools/test.py --launcher pytorch --cfg_file ${CONFIG} --batch_size 1 --eval_all |& tee -a $TEST_LOG

else
    CUDA_VISIBLE_DEVICES=$GPU python -u tools/traintest.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs $EP --out_dir=$OUT_DIR --test_batch_size $TEST_BATCH_SIZE --trainval ${TRAINVAL} --tcp_port=${PORT} --max_epochs ${MAX_EP} --pseudo_label ${PSEUDO_LABEL} --debug ${DEBUG} --pseu_train_ratio ${PSEUDO_TRAIN_RATIO} --double_test ${DOUBLE_TEST} --select_posprop ${SELECT_POSPROP}  --select_negprop ${SELECT_NEGPROP} --vis ${VIS} --workers ${WORKERS} --context ${CONTEXT} --fpn_only ${FPN_ONLY} --eval_last ${EVAL_LAST} --points_range ${RANGE}  --mcd ${MCD} --l_pow ${L_POW} --mcd_curve ${MCD_CURVE} --sync_bn $(optional $FIX_RANDOM_SEED --fix_random_seed)|& tee $OUTPUT_LOG

    # CUDA_VISIBLE_DEVICES=$GPU python -u tools/test.py --cfg_file ${CONFIG} --batch_size 1 --eval_all |& tee -a $TEST_LOG
fi

cp $OUTPUT_LOG $OUT_DIR
cp $CONFIG $OUT_DIR