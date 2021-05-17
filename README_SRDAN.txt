## create virtual environment and activate
python3.6 -m venv ~/PCDet
source ~/PCDet/bin/activate

## install all requirements
pip install torch==1.1
pip install -r requirements.txt

## modify nuscenes library
cp modify_nusc_lib/splits.py ~/PCDet/lib/python3.6/site-packages/nuscenes/utils/splits.py
cp modify_nusc_lib/loaders.py ~/PCDet/lib/python3.6/site-packages/nuscenes/eval/detection/loaders.py

## create dataset (Take nuscenes as example) modify paths in line 810 and line 815 of nuscenes_dataset.py
python pcdet/datasets/nuscenes/nuscenes_dataset.py create_nuscenes_infos half_singapore
python pcdet/datasets/nuscenes/nuscenes_dataset.py create_nuscenes_dbinfos half_singapore
python pcdet/datasets/nuscenes/nuscenes_dataset.py create_nuscenes_infos boston
python pcdet/datasets/nuscenes/nuscenes_dataset.py create_nuscenes_dbinfos boston

## modify config 'run_main.sh' and run code. if running SRDAN boston -> singapore, change $VER in line 19 of run_main.sh to: second_da_nounet_bost2sing_novelo_PMA-0.1lr0.5_RCD-0.1lr0.5_FPN_Up1Down1_NoShare_Large_SRDAN_0.25loc and run bash
bash run_main.sh

## in case code crash use rerun code
bash rerun.sh main

## can use upload.sh and download.sh to synchorize the project with github
## rename current repo to myrepo ##
git remote rename {origin} myrepo
bash upload.sh {branch} {commit message}
bash download.sh {branch}