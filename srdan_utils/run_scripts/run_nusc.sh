#!/bin/bash

GPU=1
GPUNUM=1
DA=True
SRC=boston
TGT=singapore

VER=pv_rcnn
CONFIG=/home/wzha8158/Dropbox/CVPR2021_LidarPerceptron/Home/OpenLidarPerceptron/tools/cfgs/nusc_models/${VER}.yaml

EVAL_EP=30
DA_W=0.3
L_Ratio=1.0
BATCH_SIZE=2

MOD=nusc

TASK_DESC=${VER}_${MOD}

OUT_DIR=/home/wzha8158/NewDet3D_Outputs/$TASK_DESC

CHECKPOINT=epoch_${TOTAL_EP}.pth
EVAL_LOC=${TGT}
EVAL_ALL=True

OUTPUT_LOG=txt/log_${TASK_DESC}.txt
TEST_LOG=txt/eval_log_${TASK_DESC}.txt

mkdir $OUT_DIR

if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

mkdir txt

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

# rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_NewDet3D/Home/Det3D ~/Dropbox/CVPR2021_NewDet3D/Ti5/
# rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_NewDet3D/Home/Det3D ~/Dropbox/CVPR2021_NewDet3D/Ti5/

# python setup.py build develop
pip uninstall pcdet -y
rm -rf pcdet.egg-info
python setup.py build develop

if (($GPUNUM > 1)); then
    set -x

    CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${GPUNUM} tools/train.py --launcher pytorch --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs 30 |& tee -a $OUTPUT_LOG

    CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --nproc_per_node=${NGPUS} tools/test.py --launcher pytorch --cfg_file ${CONFIG} --batch_size 1 --eval_all |& tee -a $TEST_LOG

else
    CUDA_VISIBLE_DEVICES=$GPU python -u tools/train.py --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs 30 |& tee -a $OUTPUT_LOG

    CUDA_VISIBLE_DEVICES=$GPU python -u tools/test.py --cfg_file ${CONFIG_FILE} --batch_size 1 --eval_all |& tee -a $TEST_LOG
fi







cp $OUTPUT_LOG $OUT_DIR 
cp $CONFIG $OUT_DIR
cp ${TEST_LOG} $OUT_DIR
cp runda.sh $OUT_DIR