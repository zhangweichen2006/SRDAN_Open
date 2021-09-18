# OpenPCDet - SRDAN 

Code repo for **SRDAN**, Scale-aware and Range-aware Domain Adaptation Network for Cross-dataset 3D Object Detection (CVPR2021). The code is implemented based on [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet). Please see [README_SRDAN](README_SRDAN.txt) for more details.

Datasets used:
1. [Nuscenes](https://www.nuscenes.org/) (Real dataset, cross-scene)
2. [Astar3D](https://github.com/I2RDL2/ASTAR-3D)(Real dataset, day-to-night)
3. [Kitti](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) (Real dataset, synthetic-to-real)
4. [PreSil](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/projects/precise-synthetic-image-and-lidar-presil-dataset-autonomous) (Synthetic dataset, synthetic-to-real)

Please consider cite:\
@InProceedings{Zhang_2021_CVPR,\
&nbsp;&nbsp;author    = {Zhang, Weichen and Li, Wen and Xu, Dong},\
&nbsp;&nbsp;title     = {SRDAN: Scale-Aware and Range-Aware Domain Adaptation Network for Cross-Dataset 3D Object Detection},\
&nbsp;&nbsp;booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},\
&nbsp;&nbsp;month     = {June},\
&nbsp;&nbsp;year      = {2021},\
&nbsp;&nbsp;pages     = {6769-6779}\
}
