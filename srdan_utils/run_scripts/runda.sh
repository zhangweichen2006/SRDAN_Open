#!/bin/bash

GPU=1
GPUNUM=1
DA=True
SRC=boston
TGT=singapore

VER=pv_rcnn
CONFIG=./tools/cfgs/kitti_models/${VER}.yaml

EVAL_EP=30
DA_W=0.3
L_Ratio=1.0
BATCH_SIZE=2

MOD=${DA_W}daw_${L_Ratio}lratio

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

CKP_FILE=$OUT_DIR/latest.pth
if test -f "$CKP_FILE"; then
    echo "pth model exist"
    CUDA_VISIBLE_DEVICES=$GPU python ./tools/train.py --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs 50 |& tee $OUTPUT_LOG
    #CUDA_VISIBLE_DEVICES=$GPU python -u ./tools/train.py $CONFIG --work_dir=$OUT_DIR \
    #--da=$DA --eval_location=$EVAL_LOC --resume_from=$CKP_FILE --da_w $DA_W --l_ratio=$L_Ratio |& tee $OUTPUT_LOG
    # -m torch.distributed.launch --nproc_per_node=$GPUNUM
else
    CUDA_VISIBLE_DEVICES=$GPU python ./tools/train.py --cfg_file ${CONFIG} --batch_size ${BATCH_SIZE} --epochs 50 |& tee $OUTPUT_LOG
    #CUDA_VISIBLE_DEVICES=$GPU python -u ./tools/train.py $CONFIG --work_dir=$OUT_DIR \
    #--da=$DA --eval_location=$EVAL_LOC --da_w $DA_W --l_ratio=$L_Ratio |& tee $OUTPUT_LOG
    # -m torch.distributed.launch --nproc_per_node=$GPUNUM  
fi


# CUDA_VISIBLE_DEVICES=$GPU python \
#     ./tools/dist_test_draw.py \
#     $CONFIG \
#     --work_dir=$OUT_DIR \
#     --checkpoint=$OUT_DIR/$CHECKPOINT \
#     --total_epoch=$EVAL_EP \
#     --eval_location=$EVAL_LOC \
#     --eval_all=$EVAL_ALL |& tee $TEST_LOG


cp $OUTPUT_LOG $OUT_DIR 
cp $CONFIG $OUT_DIR
cp ${TEST_LOG} $OUT_DIR
cp runda.sh $OUT_DIR