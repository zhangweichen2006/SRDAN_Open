import tqdm
import time
import pickle
import numpy as np
import torch
import os
from pcdet.utils import common_utils, box_utils
from pcdet.models import load_data_to_gpu
import torch.distributed as dist

import matplotlib.pyplot as plt

# from categorical_scatter import categorical_scatter_2d
# import sklearn.manifold as manifold
# from sklearn.decomposition import PCA as PCA
# sys.path.insert(0, './bhtsne')
# import bhtsne


# def get_world_size():
#     if not dist.is_available():
#         return 1
#     if not dist.is_initialized():
#         return 1
#     return dist.get_world_size()


# def synchronize():
#     """
#     Helper function to synchronize (barrier) among all processes when
#     using distributed training
#     """
#     if not dist.is_available():
#         return
#     if not dist.is_initialized():
#         return
#     world_size = get_world_size()
#     if world_size == 1:
#         return
#     dist.barrier()


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = common_utils.get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
    

def eval_one_epoch_nusc(cfg, model, dataloader, epoch_id, logger, distrib=False, save_to_file=False, result_dir=None, eval_location=None, epoch_point = [], map_dict={}, pseudo_label=False, vis=False, debug=False, test_id=0, t_mode=False, context=False, fpn_layers=[], map_dict_fpn={}, fpn_only=False, tsne=False, test_s_loader=None, map_dict_fuse={}, draw_matrix=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    # print("result_dir", result_dir)

    final_output_dir = result_dir / f'final_result' / 'data'
    final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    detections = {}
    cpu_device = torch.device("cpu")

    final_output_dir_fpn = {}
    detections_fpn = {}
    for layer in fpn_layers:
        final_output_dir_fpn[layer] = result_dir / f'final_result{layer}' / 'data'
        final_output_dir_fpn[layer].mkdir(parents=True, exist_ok=True)
        detections_fpn[layer] = {}

    if cfg.MODEL.get('FPN_FUSE_RES', True) and len(fpn_layers) > 0 and not fpn_only:
        rpn_fuse_res = True
        final_output_dir_fuse = result_dir / f'final_result_fuse' / 'data'
        final_output_dir_fuse.mkdir(parents=True, exist_ok=True)
        detections_fuse = {}
    else:
        rpn_fuse_res = False

    logger.info(f'*************** EPOCH {epoch_id} EVALUATION *****************')

    model.eval()

    if cfg.LOCAL_RANK == 0:
        if tsne:
            progress_bar_s = tqdm.tqdm(total=len(test_s_loader), leave=True, desc='eval', dynamic_ncols=True)
            progress_bar_t = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
            progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
        else:
            progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
        
    start_time = time.time()
    
    # save_points = np.zeros((len(dataloader), 4))

    if pseudo_label:
        pseudo_set = []
        # TRAIN:
        # gt_boxes: n*10, gt_importance: n, gt_classes:n,  points: m * 6,
        # use_lead_xyz: True, voxels: v*5, voxel_coords: v*4, voxel_num_points: v,
        # batch_size: btch

        # TEST:
        # points: m * 6,
        # use_lead_xyz: True, voxels: v*5, voxel_coords: v*4, voxel_num_points: v,
        # （no) gt_boxes: n*10, gt_importance: n, gt_classes:n,  
    if context:
        t_mode='dom_img_det'
    else:
        t_mode='det'

    # print("cfg", cfg)

    pseudo_dict_temp = {}
    num_box_list = np.array([])
    dist_list = np.array([])
    bevarea_list = np.array([])
    volume_list = np.array([])
    xy_list = np.array([])
    xyz_list = np.array([])
    
    num_box_list_fpn = {}
    dist_list_fpn = {}
    bevarea_list_fpn = {}
    volume_list_fpn = {}
    xy_list_fpn = {}
    xyz_list_fpn = {}
    
    if draw_matrix:
        model.eval()

        save_path = result_dir

        from matplotlib import pyplot
        matrix = model.module.dense_head.att_patch_layer.patch_matrix
        # print("matrix", matrix.shape)
        f_min, f_max = matrix.min(), matrix.max()
        matrix = (matrix - f_min) / (f_max - f_min)
        # print("matrix", matrix)
        matrix = matrix.cpu().detach().numpy().transpose(2,1,0).squeeze(-1)

        pyplot.imshow(matrix, cmap='RdYlBu')
        pyplot.savefig(os.path.join(save_path, f'matrix_vis.png'), bbox_inches='tight')
        pyplot.close()

        for i in fpn_layers:

            matrix_fpn = model.module.dense_head.att_patch_layer_fpn[i].patch_matrix
            f_min_fpn, f_max_fpn = matrix_fpn.min(), matrix_fpn.max()
            matrix_fpn = (matrix_fpn - f_min_fpn) / (f_max_fpn - f_min_fpn)
            matrix_fpn = matrix_fpn.cpu().detach().numpy().transpose(2,1,0).squeeze(-1)
            # print("matrix_fpn", matrix_fpn)

            pyplot.imshow(matrix_fpn, cmap='RdYlBu')
            # pyplot.show()
            pyplot.savefig(os.path.join(save_path, f'matrix_vis_fpn{i}.png'), bbox_inches='tight')
            pyplot.close()

    if tsne:
        tsne_source_feat = None
        logger.info('*************** TSNE VIS of EPOCH %s *****************' % epoch_id)

        Draw_set_S_x = []#np.array([])#.cuda()
        Draw_set_S_x_fpn3 = []#np.array([])#.cuda()

        Draw_set_S_x_fpn4 = []#np.array([])#.cuda()
        Draw_set_S_x_fpn5 = []#np.array([])#.cuda()

        Draw_set_T_x = []#np.array([])#.cuda()
        Draw_set_T_x_fpn3 = []#np.array([])#.cuda()

        Draw_set_T_x_fpn4 = []#np.array([])#.cuda()
        Draw_set_T_x_fpn5 = []#np.array([])#.cuda()

        for m, batch_dict in enumerate(test_s_loader):
            # print("batch_dict", batch_dict)
            # vis purpose
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                out_dict = model(batch_dict, t_mode='tsne')
                torch.cuda.empty_cache()
            # print("batch_dict s", out_dict['spatial_features_2d'])

            # for n in range(0,1):
            x_feat = out_dict[f'tsne_spatial_features_2d'].cpu().numpy()#.squeeze()
            # print(x_feat.shape)
            Draw_set_S_x.append(x_feat)

            x_feat2 = out_dict[f'tsne_spatial_features_2d_fpn5'].cpu().numpy()#.squeeze()
            Draw_set_S_x_fpn5.append(x_feat2)

            y_feat = out_dict[f'tsne_spatial_features_2d_fpn4'].cpu().numpy()#.squeeze()
            Draw_set_S_x_fpn4.append(y_feat)

            y_feat2 = out_dict[f'tsne_spatial_features_2d_fpn3'].cpu().numpy()#.squeeze()
            Draw_set_S_x_fpn3.append(y_feat2)

            torch.cuda.empty_cache()

            if cfg.LOCAL_RANK == 0:
                progress_bar_s.update()

        #############
        Draw_set_S_x = np.concatenate(Draw_set_S_x)
        Draw_set_S_x_fpn5 = np.concatenate(Draw_set_S_x_fpn5)

        Draw_set_S_x_fpn4 = np.concatenate(Draw_set_S_x_fpn4)
        Draw_set_S_x_fpn3 = np.concatenate(Draw_set_S_x_fpn3)

        print("Draw_set_S_x", Draw_set_S_x.shape)
        # print("Draw_set_S_x_fpn3", Draw_set_S_x_fpn3.shape)
        #############

        for n, batch_dict in enumerate(dataloader):
            # print("batch_dict", batch_dict)
            # vis purpose
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                out_dict = model(batch_dict, t_mode='tsne')
                torch.cuda.empty_cache()

            x_feat = out_dict[f'tsne_spatial_features_2d'].cpu().numpy()#.squeeze()
            Draw_set_T_x.append(x_feat)

            x_feat2 = out_dict[f'tsne_spatial_features_2d_fpn5'].cpu().numpy()#.squeeze()
            Draw_set_T_x_fpn5.append(x_feat2)

            y_feat = out_dict[f'tsne_spatial_features_2d_fpn4'].cpu().numpy()#.squeeze()
            Draw_set_T_x_fpn4.append(y_feat)

            y_feat2 = out_dict[f'tsne_spatial_features_2d_fpn3'].cpu().numpy()#.squeeze()
            Draw_set_T_x_fpn3.append(y_feat2)

            torch.cuda.empty_cache()

            if cfg.LOCAL_RANK == 0:
                progress_bar_t.update()
        
        common_utils.synchronize()

        all_Draw_set_S_x = all_gather(Draw_set_S_x)
        all_Draw_set_S_x_fpn5 = all_gather(Draw_set_S_x_fpn5)
        all_Draw_set_S_x_fpn4 = all_gather(Draw_set_S_x_fpn4)
        all_Draw_set_S_x_fpn3 = all_gather(Draw_set_S_x_fpn3)

        all_Draw_set_T_x = all_gather(Draw_set_T_x)
        all_Draw_set_T_x_fpn5 = all_gather(Draw_set_T_x_fpn5)
        all_Draw_set_T_x_fpn4 = all_gather(Draw_set_T_x_fpn4)
        all_Draw_set_T_x_fpn3 = all_gather(Draw_set_T_x_fpn3)
        # np.save('save_points.npy', save_points) 

        # predictions = {}
        # for p in all_predictions:
        #     predictions.update(p)


        #############
        Draw_set_T_x = np.concatenate(Draw_set_T_x)

        Draw_set_T_x_fpn5 = np.concatenate(Draw_set_T_x_fpn5)
        Draw_set_T_x_fpn4 = np.concatenate(Draw_set_T_x_fpn4)
        Draw_set_T_x_fpn3 = np.concatenate(Draw_set_T_x_fpn3)
        
        print("Draw_set_T_x", Draw_set_T_x.shape)
        # print("Draw_set_T_x_fpn3", Draw_set_T_x_fpn3.shape)
        ####################
        # if cfg.LOCAL_RANK == 0:
        Draw_set_ST_x = np.concatenate((Draw_set_S_x, Draw_set_T_x))
        Draw_set_ST_x_fpn5 = np.concatenate((Draw_set_S_x_fpn5, Draw_set_T_x_fpn5))
        Draw_set_ST_x_fpn4 = np.concatenate((Draw_set_S_x_fpn4, Draw_set_T_x_fpn4))
        Draw_set_ST_x_fpn3 = np.concatenate((Draw_set_S_x_fpn3, Draw_set_T_x_fpn3))

        # Draw_set_S_x_ori5 = np.concatenate((Draw_set_S_x, Draw_set_S_x_fpn4))
        # # Draw_set_S_x_fpn43 = np.concatenate((Draw_set_S_x_fpn4, Draw_set_S_x_fpn3))

        # Draw_set_T_x_ori5 = np.concatenate((Draw_set_T_x, Draw_set_T_x_fpn4))
        # Draw_set_T_x_fpn43 = np.concatenate((Draw_set_T_x_fpn4, Draw_set_T_x_fpn5))

        print("Draw_set_ST_x", Draw_set_ST_x.shape)
        # print("Draw_set_S_x_ori5", Draw_set_S_x_ori5.shape)

        np.save(result_dir/'Draw_set_ST_x.npy', Draw_set_ST_x) 
        np.save(result_dir/'Draw_set_ST_x_fpn3.npy', Draw_set_ST_x_fpn3) 
        np.save(result_dir/'Draw_set_ST_x_fpn4.npy', Draw_set_ST_x_fpn4) 
        np.save(result_dir/'Draw_set_ST_x_fpn5.npy', Draw_set_ST_x_fpn5) 
        # np.save(result_dir/'Draw_set_S_x_ori5.npy', Draw_set_S_x_ori5) 
        # # np.save(result_dir/'Draw_set_S_x_fpn43.npy', Draw_set_S_x_fpn43) 
        # np.save(result_dir/'Draw_set_T_x_ori5.npy', Draw_set_T_x_ori5) 
        # np.save(result_dir/'Draw_set_T_x_fpn43.npy', Draw_set_T_x_fpn43) 

        ########################
        # return None, None

    for layer in fpn_layers:
        num_box_list_fpn[layer] = np.array([])
        dist_list_fpn[layer] = np.array([])
        bevarea_list_fpn[layer] = np.array([])
        volume_list_fpn[layer] = np.array([])
        xy_list_fpn[layer] = np.array([])
        xyz_list_fpn[layer] = np.array([])

    for j, batch_dict in enumerate(dataloader):
        # print("batch_dict", batch_dict)
        # vis purpose
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            if rpn_fuse_res and not fpn_only:
                pred_dicts, ret_dict, pred_dicts_fuse, ret_dict_fuse, pred_dicts_fpn, ret_dict_fpn = model(batch_dict, t_mode=t_mode)
            elif len(fpn_layers) > 0 and not fpn_only:
                pred_dicts, ret_dict, pred_dicts_fpn, ret_dict_fpn = model(batch_dict, t_mode=t_mode)
            elif fpn_only:
                pred_dicts_fpn, ret_dict_fpn = model(batch_dict, t_mode=t_mode)
            else:
                pred_dicts, ret_dict = model(batch_dict, t_mode=t_mode)

            torch.cuda.empty_cache()
        disp_dict = {}
        
        if not fpn_only:
            batch_size = len(pred_dicts)
        else:
            batch_size = len(pred_dicts_fpn[fpn_layers[0]])
        
        for i in range(batch_size):
            # predict results
            if not fpn_only:
                token = batch_dict["metadata"][i]["token"]
                for k, v in pred_dicts[i].items():
                    if k not in [
                        "metadata",
                    ]:  
                        pred_dicts[i][k] = v.to(cpu_device)

                pred_dicts[i]["metadata"] = batch_dict["metadata"][i]
                pred_dicts[i]['box3d_lidar'] = pred_dicts[i].pop(f'pred_boxes')
                pred_dicts[i]['scores'] = pred_dicts[i].pop(f'pred_scores')
                pred_dicts[i]['label_preds'] = pred_dicts[i].pop(f'pred_labels') 
                
                detections.update(
                    {token: pred_dicts[i],}
                )

                dist, bevarea, volume, xy, xyz = box_utils.get_box_range_bev_area_volume(pred_dicts[i]['box3d_lidar'])
                # print("pred_dicts[i]['box3d_lidar']", pred_dicts[i]['box3d_lidar'])

                num_box_list = np.concatenate((num_box_list, np.array([pred_dicts[i]['box3d_lidar'].shape[0]])))
                dist_list = np.concatenate((dist_list, dist))
                bevarea_list = np.concatenate((bevarea_list, bevarea))
                volume_list = np.concatenate((volume_list, volume))
                xy_list = np.concatenate((xy_list, xy))
                xyz_list = np.concatenate((xyz_list, xyz))
# ref_boxes
            # predict results
            if rpn_fuse_res:
                token = batch_dict["metadata"][i]["token"]
                for k, v in pred_dicts_fuse[i].items():
                    if k not in [
                        "metadata",
                    ]:  
                        pred_dicts_fuse[i][k] = v.to(cpu_device)

                pred_dicts_fuse[i]["metadata"] = batch_dict["metadata"][i]
                pred_dicts_fuse[i]['box3d_lidar'] = pred_dicts_fuse[i].pop(f'pred_boxes_fuse')
                pred_dicts_fuse[i]['scores'] = pred_dicts_fuse[i].pop(f'pred_scores_fuse')
                pred_dicts_fuse[i]['label_preds'] = pred_dicts_fuse[i].pop(f'pred_labels_fuse') 
                
                detections_fuse.update(
                    {token: pred_dicts_fuse[i],}
                )

            ####### for FPN layers #######
            for layer in fpn_layers:
                suffix = f'_fpn{layer}'

                token = batch_dict["metadata"][i]["token"]
                for k, v in pred_dicts_fpn[layer][i].items():
                    if k not in [
                        "metadata",
                    ]:  
                        pred_dicts_fpn[layer][i][k] = v.to(cpu_device)

                pred_dicts_fpn[layer][i]["metadata"] = batch_dict["metadata"][i]
                pred_dicts_fpn[layer][i]['box3d_lidar'] = pred_dicts_fpn[layer][i].pop(f'pred_boxes{suffix}')
                pred_dicts_fpn[layer][i]['scores'] = pred_dicts_fpn[layer][i].pop(f'pred_scores{suffix}')
                pred_dicts_fpn[layer][i]['label_preds'] = pred_dicts_fpn[layer][i].pop(f'pred_labels{suffix}')
                
                detections_fpn[layer].update(
                    {token: pred_dicts_fpn[layer][i],}
                )

                dist_fpn, bevarea_fpn, volume_fpn, xy_fpn, xyz_fpn = box_utils.get_box_range_bev_area_volume(pred_dicts_fpn[layer][i]['box3d_lidar'])
                
                num_box_list_fpn[layer] = np.concatenate((num_box_list_fpn[layer], np.array([pred_dicts_fpn[layer][i]['box3d_lidar'].shape[0]])))
                # print("dist_list_fpn[layer]", dist_list_fpn[layer].shape, dist_list_fpn[layer])
                # print("dist_fpn", dist_fpn.shape, dist_fpn)
                dist_list_fpn[layer] = np.concatenate((dist_list_fpn[layer], dist_fpn))
                bevarea_list_fpn[layer] = np.concatenate((bevarea_list_fpn[layer], bevarea_fpn))
                volume_list_fpn[layer] = np.concatenate((volume_list_fpn[layer], volume_fpn))
                xy_list_fpn[layer] = np.concatenate((xy_list_fpn[layer], xy_fpn))
                xyz_list_fpn[layer] = np.concatenate((xyz_list_fpn[layer], xyz_fpn))

                # print(f"pred_dicts_fpn[{layer}][i]['box3d_lidar']", pred_dicts_fpn[layer][i]['box3d_lidar'])
                
                
                # print("pred_dicts_fpn[i]",pred_dicts_fpn[layer][i])

            # visualization
            if vis:
                from visual_utils import visualize_utils as V
                import mayavi.mlab as mlab

                V.draw_scenes(
                    points=batch_dict['points'][:, 1:], 
                    gt_boxes=batch_dict['gt_boxes'][i],
                    ref_boxes=pred_dicts[i]['box3d_lidar'],
                    ref_scores=pred_dicts[i]['scores'], ref_labels=pred_dicts[i]['label_preds']
                )
                # print("vis saved")
                    # [],
                mlab.show(stop=True)
                mlab.savefig(filename=f'test_{i}.png')

            # pseudo_label
            if pseudo_label:
                # pseudo_dict_temp['points'] = batch_dict[f'points_single{i}'].to(cpu_device)
                # pseudo_dict_temp['voxels'] = batch_dict[f'voxels_single{i}'].to(cpu_device)
                # pseudo_dict_temp['voxel_num_points'] = batch_dict[f'voxel_num_points_single{i}'].to(cpu_device)
                # pseudo_dict_temp['voxel_coords'] = batch_dict[f'voxel_coords_single{i}'].to(cpu_device)
                # pseudo_dict_temp['use_lead_xyz'] = batch_dict['use_lead_xyz'][i].to(cpu_device)
                pseudo_dict_temp['metadata'] = pred_dicts[i]["metadata"]
                pseudo_dict_temp['pseudo_boxes'] = pred_dicts[i]['box3d_lidar'].numpy()
                pseudo_dict_temp['pseudo_importance'] = pred_dicts[i]['scores'].numpy()
                pseudo_dict_temp['pseudo_classes'] = pred_dicts[i]['label_preds'].numpy()
                pseudo_set.append(pseudo_dict_temp)
                
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

        # if vis:
        #     break

        # if debug:
        #     break
    # print("cls", torch.sort(cls_preds)[0])
    # print("cls fpn", torch.sort(cls_preds_fpn)[0])

    common_utils.synchronize()

    all_predictions = all_gather(detections)

    all_predictions_fpn = {}
    for layer in fpn_layers:
        all_predictions_fpn[layer] = all_gather(detections_fpn[layer])

    if rpn_fuse_res:
        all_predictions_fuse = all_gather(detections_fuse)
    # np.save('save_points.npy', save_points) 

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    # print('predictions', predictions)
    predictions_fpn = {}
    for layer in fpn_layers:
        predictions_fpn[layer] = {}
        for p in all_predictions_fpn[layer]:
            predictions_fpn[layer].update(p)

    if rpn_fuse_res:
        predictions_fuse = {}
        for p in all_predictions_fuse:
            predictions_fuse.update(p)

    # np.save('predictions.npy', predictions)
    # if args.eval_location=='None' or args.eval_location is None:

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    # print("dataset.root_path", dataset.root_path)

    print("tsne", tsne)
    if tsne:
        tsne_source_feat = None
        logger.info('*************** TSNE VIS of EPOCH %s *****************' % epoch_id)

        for m, batch_dict in enumerate(test_s_loader):
            # print("batch_dict", batch_dict)
            # vis purpose
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                batch_dict = model(batch_dict, t_mode='tsne')
                torch.cuda.empty_cache()
            # print("batch_dict s", batch_dict['spatial_features_2d'].shape)

        for n, batch_dict in enumerate(dataloader):
            # print("batch_dict", batch_dict)
            # vis purpose
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                batch_dict = model(batch_dict, t_mode='tsne')
                torch.cuda.empty_cache()
                
            # print("batch_dict t", batch_dict['spatial_features_2d'].shape)

            torch.cuda.empty_cache()
        
    model.train()
    
    # save_points_np = np.load('save_points.npy')
    # print("len(predictions.keys())",len(predictions.keys()))

    if cfg.LOCAL_RANK == 0:# and not debug

        # with open(os.path.join(result_dir, "prediction.pkl"), "wb") as f:
        #     pickle.dump(predictions, f)

        # with open(os.path.join(result_dir, 'prediction.pkl'), 'rb') as f:
        #     predictions = pickle.load(f)
        # print("predictions", predictions)

        if not fpn_only:
            result_dict, _ = dataset.evaluation(predictions, output_dir=final_output_dir, class_names=class_names, eval_location=eval_location)

            if eval_location == None:
                out_dict = None
                pseudo_set = None
            else:
                mAP={}
                for k, v in result_dict["results"].items():
                    print(f"Evaluation {k}: {v}")
                
                out_dict = result_dict['detail']['eval.nusc']
                for k in out_dict.keys():
                    mAP[k] = float(result_dict["results"]["nusc"].split(': ')[-1])
                
                # print("mAP", mAP)

                for k in out_dict.keys():
                    if k in map_dict.keys():
                        map_dict[k][0].append(out_dict[k]['dist@0.5']*100)
                        map_dict[k][1].append(out_dict[k]['dist@1.0']*100)
                        map_dict[k][2].append(out_dict[k]['dist@2.0']*100)
                        map_dict[k][3].append(out_dict[k]['dist@4.0']*100)
                        map_dict[k][4].append(mAP[k]*100)
                    else:
                        map_dict[k] = [[out_dict[k]['dist@0.5']*100], [out_dict[k]['dist@1.0']*100], [out_dict[k]['dist@2.0']*100], [out_dict[k]['dist@4.0']*100], [mAP[k]*100]]
                # print("map_dict", map_dict)
                # print("epoch_point", epoch_point)
                # print('class_names', class_names)
                draw_graph(final_output_dir, epoch_point, map_dict, class_names, test_id=test_id) #,'pedestrian'

                # draw_distribution(final_output_dir, num_box_list, dist_list, bevarea_list, volume_list, xy_list, xyz_list)

            if rpn_fuse_res:
                result_dict_fuse, _ = dataset.evaluation(predictions_fuse, output_dir=final_output_dir_fuse, class_names=class_names, eval_location=eval_location)

                if eval_location is not None:
                    for k, v in result_dict_fuse["results"].items():
                        print(f"Evaluation {k}: {v}")
                    
                    mAP_fuse = {}
                    out_dict_fuse = result_dict_fuse['detail']['eval.nusc']
                    for k in out_dict_fuse.keys():
                        mAP_fuse[k] = float(result_dict_fuse["results"]["nusc"].split(': ')[-1])
                        
                    for k in out_dict_fuse.keys():
                        if k in map_dict_fuse.keys():
                            map_dict_fuse[k][0].append(out_dict_fuse[k]['dist@0.5']*100)
                            map_dict_fuse[k][1].append(out_dict_fuse[k]['dist@1.0']*100)
                            map_dict_fuse[k][2].append(out_dict_fuse[k]['dist@2.0']*100)
                            map_dict_fuse[k][3].append(out_dict_fuse[k]['dist@4.0']*100)
                            map_dict_fuse[k][4].append(mAP_fuse[k]*100)
                        else:
                            map_dict_fuse[k] = [[out_dict_fuse[k]['dist@0.5']*100], [out_dict_fuse[k]['dist@1.0']*100], [out_dict_fuse[k]['dist@2.0']*100], [out_dict_fuse[k]['dist@4.0']*100], [mAP_fuse[k]*100]]
                    # print("map_dict", map_dict)
                    # print("epoch_point", epoch_point)
                    # print('class_names', class_names)
                    draw_graph(final_output_dir_fuse, epoch_point, map_dict_fuse, class_names, test_id=test_id) #,'pedestrian'

            

        if eval_location == None:
            out_dict_fpn = None
        else:

            for layer in fpn_layers:
                result_dict_fpn, _ = dataset.evaluation(predictions_fpn[layer], output_dir=final_output_dir_fpn[layer], class_names=class_names, eval_location=eval_location)

                print(f"############## FPN Layer {layer}#############")
                mAP_fpn = {}
                for k, v in result_dict_fpn["results"].items():
                    print(f"Evaluation {k}: {v}")
                
                out_dict_fpn = result_dict_fpn['detail']['eval.nusc']

                for k in out_dict_fuse.keys():
                    mAP_fpn[k] = float(result_dict_fpn["results"]["nusc"].split(': ')[-1])
                    
                for k in out_dict_fpn.keys():
                    if k in map_dict_fpn[layer].keys():
                        map_dict_fpn[layer][k][0].append(out_dict_fpn[k]['dist@0.5']*100)
                        map_dict_fpn[layer][k][1].append(out_dict_fpn[k]['dist@1.0']*100)
                        map_dict_fpn[layer][k][2].append(out_dict_fpn[k]['dist@2.0']*100)
                        map_dict_fpn[layer][k][3].append(out_dict_fpn[k]['dist@4.0']*100)
                        map_dict_fpn[layer][k][4].append(mAP_fpn[k]*100)
                    else:
                        map_dict_fpn[layer][k] = [[out_dict_fpn[k]['dist@0.5']*100], [out_dict_fpn[k]['dist@1.0']*100], [out_dict_fpn[k]['dist@2.0']*100], [out_dict_fpn[k]['dist@4.0']*100], [mAP_fpn[k]*100]]
                # print("map_dict", map_dict)
                # print("epoch_point", epoch_point)
                draw_graph(final_output_dir_fpn[layer], epoch_point, map_dict_fpn[layer], class_names, test_id=test_id)

                # draw_distribution(final_output_dir_fpn[layer], num_box_list_fpn[layer], dist_list_fpn[layer], bevarea_list_fpn[layer], volume_list_fpn[layer], xy_list_fpn[layer], xyz_list_fpn[layer])

        logger.info('****************Evaluation done.*****************')

        if fpn_only:
            if pseudo_label:
                return out_dict_fpn, None#pseudo_set_fpn
            return out_dict_fpn, None
        if pseudo_label:
            return out_dict, pseudo_set
        return out_dict, None
    else:
        if fpn_only:
            return {}, None
        if pseudo_label:
            return {}, pseudo_set
        return {}, None

def draw_graph(out_dir, epoch_point, res_dict, classes=[], test_id=0, kitti=False):
    ######### draw ###########
    print("draw classes", classes)
    # print('test_id', )
    for c in classes:
        fig, ax = plt.subplots()

        if kitti:
            ax.plot(epoch_point, res_dict[c][0], 'k', label='Easy',color='r')
            ax.plot(epoch_point, res_dict[c][1], 'k', label='Moderate',color='g')
            ax.plot(epoch_point, res_dict[c][2], 'k', label='Hard',color='b')
            ax.annotate('Last Epoch MAP: %0.4f' % (res_dict[c][1][-1]), xy=(1.05, 0.1), xycoords='axes fraction', size=14)
        else:
            ax.plot(epoch_point, res_dict[c][0], 'k', label='MAP 0.5',color='r')
            ax.plot(epoch_point, res_dict[c][1], 'k', label='MAP 1.0',color='g')
            ax.plot(epoch_point, res_dict[c][2], 'k', label='MAP 2.0',color='b')
            ax.plot(epoch_point, res_dict[c][3], 'k', label='MAP 4.0',color='c')
            ax.plot(epoch_point, res_dict[c][4], 'k', label='MAP',color='black')

            ax.annotate('Last Epoch Overall MAP: %0.4f' % (res_dict[c][4][-1]), xy=(1.05, 0.1), xycoords='axes fraction', size=14)

        # Now add the legend with some customizations.
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., shadow=True)

        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')

        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')

        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width

        fig.text(0.5, 0.02, 'EPOCH', ha='center')
        fig.text(0.02, 0.5, 'ACCURACY', va='center', rotation='vertical')

        save_path = out_dir

        if kitti:
            plt.savefig(os.path.join(save_path, f'map_epoch_{c}_{test_id}_{str(res_dict[c][1][-1])}.png'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, f'map_epoch_{c}_{test_id}_{str(res_dict[c][4][-1])}.png'), bbox_inches='tight')
        
        fig.clf()

        plt.clf()

def draw_distribution(out_dir, num_box_list, dist_list, bevarea_list, volume_list, xy_list, xyz_list):

    draw_dict = {"num_box": num_box_list, "dist": dist_list, "bevarea": bevarea_list, "volume": volume_list, "xy": xy_list, "xyz": xyz_list}

    for key, val in draw_dict.items():
        # print("val", val)
        mu=np.mean(val)   
        sigma =np.std(val)  
        # num=1000

        # plt.annotate('Mean: %0.4f' % (mu), xy=(1.05, 0.1), xycoords='axes fraction', size=14)
        # plt.annotate('STD: %0.4f' % (sigma), xy=(1.05, 0.1), xycoords='axes fraction', size=14)

        # plt.plot(np.linspace(0,150,num=150), val, color='orange')

        # plt.savefig(os.path.join(out_dir, f'{key}.png'), bbox_inches='tight')

        # plt.clf()

        # plt.annotate(f'Mean: {mu:0.2f}', xy=(1.05, 0.1), size=14)
        # plt.annotate(f'STD: {sigma:0.2f}', xy=(1.05, 0.2), size=14)

        normal_data = val

        count, bins, ignored = plt.hist(normal_data, 20)

        # print("count, bins", count, bins)

        if (sigma * np.sqrt(2 * np.pi)) != 0:
            plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='orange')

        plt.savefig(os.path.join(out_dir, f'{key}_normal.png'), bbox_inches='tight')

        plt.clf()

def draw_overall_graph(out_dir, epoch_point, res_dict, res_dict2={}, kitti=False):
    ######### draw ###########
    # print("draw classes", classes)
    classes = res_dict.keys()
    for c in classes:
        fig, ax = plt.subplots()

        ax.plot(epoch_point, res_dict[c][0], 'k', label='MAP 0.5',color='r')
        ax.plot(epoch_point, res_dict[c][1], 'k', label='MAP 1.0',color='g')
        ax.plot(epoch_point, res_dict[c][2], 'k', label='MAP 2.0',color='b')
        if not kitti:
            ax.plot(epoch_point, res_dict[c][3], 'k', label='MAP 4.0',color='y')

        if res_dict2 != {}:
            ax.plot(epoch_point, res_dict2[c][0], 'k', label='2nd Test MAP 0.5',color='lightcoral')
            ax.plot(epoch_point, res_dict2[c][1], 'k', label='2nd Test MAP 1.0',color='lime')
            ax.plot(epoch_point, res_dict2[c][2], 'k', label='2nd Test MAP 2.0',color='cornflowerblue')
            if not kitti:
                ax.plot(epoch_point, res_dict2[c][3], 'k', label='2nd Test MAP 4.0',color='gold')

            if not kitti:
                ax.annotate('Last Epoch MAP: %0.4f, %0.4f' % (res_dict[c][2][-1], res_dict2[c][2][-1]), xy=(1.05, 0.1), xycoords='axes fraction', size=14)
            else:
                ax.annotate('Last Epoch MAP: %0.4f, %0.4f' % (res_dict[c][1][-1], res_dict2[c][1][-1]), xy=(1.05, 0.1), xycoords='axes fraction', size=14)

        # Now add the legend with some customizations.
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., shadow=True)

        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')

        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')

        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width

        fig.text(0.5, 0.02, 'EPOCH', ha='center')
        fig.text(0.02, 0.5, 'ACCURACY', va='center', rotation='vertical')

        save_path = out_dir

        plt.savefig(os.path.join(save_path, f'map_epoch_{c}_all.png'), bbox_inches='tight')
        
        fig.clf()

        plt.clf()
    
    print("draw:", os.path.join(save_path, f'map_epoch_{c}_all.png'))

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, distrib=False, save_to_file=False, result_dir=None, epoch_point=[], map_dict={}, pseudo_label=False, vis=False, debug=False, test_id=0, t_mode='test_det', context=False, fpn_layers=[], map_dict_fpn={}, fpn_only=False, tsne=False, test_s_loader=None, map_dict_fuse={}, draw_matrix=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    # if save_to_file:
    final_output_dir = result_dir / f'final_result'
    final_output_dir.mkdir(parents=True, exist_ok=True)
    # else:
    #     final_output_dir = None

    if cfg.MODEL.get('FPN_FUSE_RES', True) and len(fpn_layers) > 0 and not fpn_only:
        rpn_fuse_res = True
        final_output_dir_fuse = result_dir / f'final_result_fuse'
        final_output_dir_fuse.mkdir(parents=True, exist_ok=True)
        detections_fuse = {}
    else:
        rpn_fuse_res = False

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    det_annos_fuse = []

    # print("final_output_dir", final_output_dir, epoch_point, map_dict, class_names, test_id)
    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    # print("distrib", distrib)
    # if distrib:
    #     num_gpus = torch.cuda.device_count()
    #     local_rank = cfg.LOCAL_RANK % num_gpus
    #     model = torch.nn.parallel.DistributedDataParallel(
    #             model,
    #             device_ids=[local_rank],
    #             broadcast_buffers=False
    #     )
    if context:
        t_mode='dom_img_det'
    else:
        t_mode='det'
        
    model.eval()

    if draw_matrix:

        from matplotlib import pyplot
        matrix = model.module.dense_head.att_patch_layer.patch_matrix
        # print("matrix", matrix.shape)
        f_min, f_max = matrix.min(), matrix.max()
        matrix = (matrix - f_min) / (f_max - f_min)
        # print("matrix", matrix)
        matrix = matrix.cpu().detach().numpy().transpose(2,1,0).squeeze(-1)
        pyplot.imshow(matrix, cmap='RdYlBu')
        save_path = result_dir
        pyplot.savefig(os.path.join(save_path, f'matrix_vis.png'), bbox_inches='tight')
        pyplot.close()
        for i in fpn_layers:

            matrix_fpn = model.module.dense_head.att_patch_layer_fpn[i].patch_matrix
            f_min_fpn, f_max_fpn = matrix_fpn.min(), matrix_fpn.max()
            matrix_fpn = (matrix_fpn - f_min_fpn) / (f_max_fpn - f_min_fpn)
            matrix_fpn = matrix_fpn.cpu().detach().numpy().transpose(2,1,0).squeeze(-1)
            # print("matrix_fpn", matrix_fpn)

            pyplot.imshow(matrix_fpn, cmap='RdYlBu')
            # pyplot.show()
            pyplot.savefig(os.path.join(save_path, f'matrix_vis_fpn{i}.png'), bbox_inches='tight')
            pyplot.close()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        # print("batch_dict", batch_dict)
        with torch.no_grad():
            # pred_dicts, ret_dict = model(batch_dict)
            if rpn_fuse_res and not fpn_only:
                pred_dicts, ret_dict, pred_dicts_fuse, ret_dict_fuse, pred_dicts_fpn, ret_dict_fpn = model(batch_dict, t_mode=t_mode)
            elif len(fpn_layers) > 0 and not fpn_only:
                pred_dicts, ret_dict, pred_dicts_fpn, ret_dict_fpn = model(batch_dict, t_mode=t_mode)
            elif fpn_only:
                pred_dicts_fpn, ret_dict_fpn = model(batch_dict, t_mode=t_mode)
            else:
                pred_dicts, ret_dict = model(batch_dict, t_mode=t_mode)

            torch.cuda.empty_cache()
            # print("pred_dicts, ret_dict", pred_dicts, ret_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        # print("ret_dict", ret_dict)
        # print("pred_dicts", pred_dicts)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos

        if rpn_fuse_res and not fpn_only:
            disp_dict_fuse = {}
            # print("ret_dict_fuse", ret_dict_fuse)
            # print("pred_dicts_fuse", pred_dicts_fuse)
            statistics_info(cfg, ret_dict_fuse, metric, disp_dict_fuse)
            # statistics_info(cfg, ret_dict, metric, disp_dict)
            annos_fuse = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts_fuse, class_names,
                output_path=final_output_dir_fuse if save_to_file else None, suffix='_fuse'
            )
            det_annos_fuse += annos_fuse
        
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if distrib:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

        if rpn_fuse_res:
            det_annos_fuse = common_utils.merge_results_dist(det_annos_fuse, len(dataset), tmpdir=result_dir / 'tmpdir')


    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    model.train()

    if cfg.LOCAL_RANK == 0:
        ret_dict = {}
        if distrib:
            for key, val in metric[0].items():
                for k in range(1, world_size):
                    metric[0][key] += metric[k][key]
            metric = metric[0]

        gt_num_cnt = metric['gt_num']
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
            ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

        total_pred_objects = 0
        for anno in det_annos:
            total_pred_objects += anno['name'].__len__()
        logger.info('Average predicted number of objects(%d samples): %.3f'
                    % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

        # with open(result_dir / 'result.pkl', 'wb') as f:
        #     pickle.dump(det_annos, f)

        
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names, output_dir=result_dir
        )
        # print("result_str", result_str)
        # print("result_dict", result_dict)
        # for k, v in result_dict.items():
        #     print(f"Evaluation {k}: {v}")
        for category in class_names:
            if category in map_dict.keys():
                map_dict[category][0].append(result_dict[f'{category}_3d/easy'])
                map_dict[category][1].append(result_dict[f'{category}_3d/moderate'])
                map_dict[category][2].append(result_dict[f'{category}_3d/hard'])
            else:
                map_dict[category] = [[result_dict[f'{category}_3d/easy']], [result_dict[f'{category}_3d/moderate']], [result_dict[f'{category}_3d/hard']]]

            # print("map_dict", map_dict)
        draw_graph(final_output_dir, epoch_point, map_dict, class_names, test_id=test_id, kitti=True) #,
        # eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC

        if rpn_fuse_res:
            result_str, result_dict = dataset.evaluation(
                det_annos_fuse, class_names, output_dir=final_output_dir_fuse
            )
            for category in class_names:
                if category in map_dict_fuse.keys():
                    map_dict_fuse[category][0].append(result_dict[f'{category}_3d/easy'])
                    map_dict_fuse[category][1].append(result_dict[f'{category}_3d/moderate'])
                    map_dict_fuse[category][2].append(result_dict[f'{category}_3d/hard'])
                else:
                    map_dict_fuse[category] = [[result_dict[f'{category}_3d/easy']], [result_dict[f'{category}_3d/moderate']], [result_dict[f'{category}_3d/hard']]]

                # print("map_dict", map_dict)
            draw_graph(final_output_dir_fuse, epoch_point, map_dict_fuse, class_names, test_id=test_id, kitti=True) #,
            # eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC

        logger.info(result_str)
        ret_dict.update(result_dict)

        logger.info('Result is save to %s' % result_dir)
        logger.info('****************Evaluation done.*****************')
        
        return ret_dict, None #pseudo_set
    else:
        return {}, None

if __name__ == '__main__':
    pass