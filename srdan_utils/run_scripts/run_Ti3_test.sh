#!/bin/bash 
set -e 
# git stash
# git pull
#set -x

GPU=0
#,1
#,1,2,3
GPUNUM=1
#4
#2
PORT=$RANDOM

VER=second_fpn_nounet_noda_halfbost2halfsing_novelo_FPN_Up1Only_NoShare_Large
#second_a1_noda_halfbost2halfsing_novelo_FPN_Down1_NoShare_Large
CONTEXT=False
#second_halfbost2halfsing_novelo
# pv_second_halfbost2halfsing_novelo_da
#_pseudo
#
#minibost2minising_novelo
#bost2sing
#_pseudo
CONFIG=./tools/cfgs/nusc_models/${VER}.yaml

EP=50
MAX_EP=50
L_Ratio=1.0
BATCH_SIZE=2
#4
#$((${GPUNUM}))
TEST_BATCH_SIZE=2
DOUBLE_TEST=True
#$((${GPUNUM}))
PSEUDO_LABEL=False
PSEUDO_TRAIN_RATIO=0.5
#.2
DEBUG=False
VIS=False
WORKERS=0
PIN_MEMORY=True
INST_DA=False
SELECT_POSPROP=0.3
SELECT_NEGPROP=0.1
#${DA_W}daW_
MOD=nusc_dann_${PSEUDO_LABEL}pseudo_reinit_${PSEUDO_TRAIN_RATIO}ratio_${GPU}GPU_${BATCH_SIZE}Batch_${SELECT_POSPROP}:${SELECT_NEGPROP}PseuProp

TASK_DESC=$(date +'%d-%m-%Y')_${VER}_${MOD}
#$(date +'%d-%m-%Y')_${VER}_${MOD}
#$(date +'%d-%m-%Y')
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
cp -r pcdet $OUT_DIR

mkdir -p txt
cp run_Ti3_test.sh $OUT_DIR

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

	CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${GPUNUM} tools/traintest.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs $EP --out_dir=$OUT_DIR --test_batch_size $TEST_BATCH_SIZE --trainval ${TRAINVAL} --tcp_port=${PORT} --max_epochs ${MAX_EP} --pseudo_label ${PSEUDO_LABEL} --debug ${DEBUG} --pseu_train_ratio ${PSEUDO_TRAIN_RATIO} --double_test ${DOUBLE_TEST} --select_posprop ${SELECT_POSPROP}  --select_negprop ${SELECT_NEGPROP} --vis ${VIS} --pin_memory ${PIN_MEMORY} --workers ${WORKERS} --ins_da ${INST_DA}  --context ${CONTEXT}|& tee $OUTPUT_LOG

    # CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${NGPUS} tools/test.py --launcher pytorch --cfg_file ${CONFIG} --batch_size 1 --eval_all |& tee -a $TEST_LOG

else
    CUDA_VISIBLE_DEVICES=$GPU python -u tools/traintest.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs $EP --out_dir=$OUT_DIR --test_batch_size $TEST_BATCH_SIZE --trainval ${TRAINVAL} --tcp_port=${PORT} --max_epochs ${MAX_EP} --pseudo_label ${PSEUDO_LABEL} --debug ${DEBUG} --pseu_train_ratio ${PSEUDO_TRAIN_RATIO} --double_test ${DOUBLE_TEST} --select_posprop ${SELECT_POSPROP}  --select_negprop ${SELECT_NEGPROP} --vis ${VIS} --workers ${WORKERS} --context ${CONTEXT}|& tee $OUTPUT_LOG

    # CUDA_VISIBLE_DEVICES=$GPU python -u tools/test.py --cfg_file ${CONFIG} --batch_size 1 --eval_all |& tee -a $TEST_LOG
fi

cp $OUTPUT_LOG $OUT_DIR 
cp $CONFIG $OUT_DIR