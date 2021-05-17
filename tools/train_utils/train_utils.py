import glob
import os

import torch
import tqdm
import datetime
import pickle
import math
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader, build_pseudo_dataloader
from eval_utils import eval_utils
import numpy as np
import torch.distributed as dist


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg, rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, debug=False, range_inv=False, mcd=False):

    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    # print("train?")
    # print("total_it_each_epoch", total_it_each_epoch)
    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
        # print("batch", batch)

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train_' + key, val, accumulated_iter)
        
        if debug:
            break

    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_one_epoch_da(model, optimizer, optimizer_dom, train_loader, source_loader, target_loader, model_func, lr_scheduler, dom_lr_scheduler, accumulated_iter, optim_cfg, rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, epoch=0, total_epochs=30, pseudo=False, debug=False, ins_da=False, context=False, range_inv=False, l_pow=10, mcd=False, mcd_curve=False, optimizer_mcd=None, dom_atten=False, optimizer_dom2=None, dom_lr_scheduler2=None, dom_reg=False, atten_w_optimizer=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
        # source_iter = iter(source_loader)
        target_iter = iter(target_loader)

    disp_dict_all = {}

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
    
    if optimizer_dom2 is not None:
        if dom_reg:
            two_opt = False
        else:
            two_opt = True
    else:
        two_opt = False

    for cur_it in range(total_it_each_epoch):
        # train batch and source batch 
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            # print('new train iters')

        # try:
        #     batch_src = next(source_iter)
        # except StopIteration:
        #     source_iter = iter(source_loader)
        #     batch_src = next(source_iter)
        #     # print('new src iters')

        try:
            batch_tgt = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            batch_tgt = next(target_iter)
            # print('new tgt iters')
        
        lr_scheduler.step(accumulated_iter)
        dom_lr_scheduler.step(accumulated_iter)
        if two_opt:
            dom_lr_scheduler2.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('learning_rate', cur_lr, accumulated_iter)

        model.train()

        ########### det ############
        # print("train batch_dict", batch)
        # print("batch_dict['points']", batch['points'].shape) 
        # print("batch_dict['gt_boxes']", batch['gt_boxes'].shape) 
        # print("batch_dict['voxels']", batch['voxels'].shape)
       
        ########### dann dom ##############
        l_ratio = 1
        p= (epoch*total_it_each_epoch+cur_it)/(total_it_each_epoch*total_epochs)
        if mcd_curve:
            l = math.sin((epoch + 1)/total_epochs * math.pi/2 )
        else:
            l= (2. / (1. + np.exp(-l_pow * p))) - 1


        # # -------- for src voxel range DA -------
        if range_inv:
            # print("range inv src")
            optimizer_dom.zero_grad()

            range_loss_src, tb_dict_src, disp_dict = model_func(model, batch, t_mode='dom_range', l=l)

            # print("img dom_loss_src", dom_loss_src.item())

            range_loss_src.backward()
            # clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer_dom.step()
            
            # -------- for tgt voxel range DA -------
            # optimizer_dom.zero_grad()
            # range_loss_tgt, tb_dict_tgt, disp_dict = model_func(model, batch_tgt, t_mode='dom_range', l=l)

            # # print("img dom_loss_tgt", dom_loss_tgt.item())

            # range_loss_tgt.backward()
            # # clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            # optimizer_dom.step()

        # # -------- for src image-level DA -------
        # print("mcd", mcd)
        # print()

        if not mcd or (mcd and context):
            # print("dom img src")
            optimizer_dom.zero_grad()
            if atten_w_optimizer is not None:
                atten_w_optimizer.zero_grad()

            dom_loss_src, tb_dict_src, disp_dict = model_func(model, batch, t_mode='dom_img_src', l=l)

            if atten_w_optimizer is not None:
                disp_dict_all.update(disp_dict)
                dom_loss_src.backward()
                # clip_grad_norm_(model.parameters(), 30) #optim_cfg.GRAD_NORM_CLIP
                optimizer_dom.step()
                atten_w_optimizer.step()
                # att_w_dom_loss_src = dom_loss_src #* 100#00000000000
                # att_w_dom_loss_src.backward()
                # clip_grad_norm_(model.parameters(), 20) #optim_cfg.GRAD_NORM_CLIP
            else:
                disp_dict_all.update(disp_dict)
                dom_loss_src.backward()
                # clip_grad_norm_(model.parameters(), 30) #optim_cfg.GRAD_NORM_CLIP
                optimizer_dom.step()

            if two_opt:
                optimizer_dom2.zero_grad()
                dom_loss_src2, tb_dict_src2, disp_dict2 = model_func(model, batch_tgt, t_mode='dom_img_src_diffdom', l=l)

                disp_dict_all.update(disp_dict)
                dom_loss_src2.backward()
                optimizer_dom2.step()
        
            if dom_reg:
                optimizer_dom2.zero_grad()
                dom_loss_src2, tb_dict_src2, disp_dict2 = model_func(model, batch_tgt, t_mode='dom_img_src_reg', l=l)

                disp_dict_all.update(disp_dict)
                dom_loss_src2.backward()
                optimizer_dom2.step()

            # -------- for tgt image-level DA -------
            # print("dom img tgt")
            optimizer_dom.zero_grad()
            if atten_w_optimizer is not None:
                atten_w_optimizer.zero_grad()
            dom_loss_tgt, tb_dict_tgt, disp_dict = model_func(model, batch_tgt, t_mode='dom_img_tgt', l=l)
            
            if atten_w_optimizer is not None:
                disp_dict_all.update(disp_dict)
                dom_loss_tgt.backward()
                # clip_grad_norm_(model.parameters(), 30) #optim_cfg.GRAD_NORM_CLIP
                optimizer_dom.step()
                atten_w_optimizer.step()
                # print(":atten_w_optimizer", atten_w_optimizer)
                # att_w_dom_loss_tgt = dom_loss_tgt #* 100#00000000000
                # att_w_dom_loss_tgt.backward()
                # clip_grad_norm_(model.parameters(), 20) #optim_cfg.GRAD_NORM_CLIP
            else:
                disp_dict_all.update(disp_dict)
                dom_loss_tgt.backward()
                # clip_grad_norm_(model.parameters(), 30) #optim_cfg.GRAD_NORM_CLIP
                optimizer_dom.step()

            if two_opt:
                optimizer_dom2.zero_grad()
                dom_loss_tgt2, tb_dict_tgt2, disp_dict2 = model_func(model, batch_tgt, t_mode='dom_img_tgt_diffdom', l=l)

                disp_dict_all.update(disp_dict)
                dom_loss_tgt2.backward()
                optimizer_dom2.step()

            if dom_reg:
                optimizer_dom2.zero_grad()
                dom_loss_tgt2, tb_dict_tgt2, disp_dict2 = model_func(model, batch_tgt, t_mode='dom_img_tgt_reg', l=l)

                disp_dict_all.update(disp_dict)
                dom_loss_tgt2.backward()
                optimizer_dom2.step()
        # -------- for src instance-level DA -------
        
        if ins_da:
            # print("dom ins src")
            optimizer_dom.zero_grad()

            dom_loss_src_ins, tb_dict, disp_dict = model_func(model, batch, t_mode='dom_ins_src', l=l)
            disp_dict_all.update(disp_dict)

            # print("ins dom_loss_src", dom_loss_src.item())
            dom_loss_src_ins.backward()
            # clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer_dom.step()
            
            # -------- for tgt instance-level -------
            # print("dom ins tgt")
            optimizer_dom.zero_grad()

            dom_loss_tgt_ins, tb_dict, disp_dict = model_func(model, batch_tgt, t_mode='dom_ins_tgt', l=l)
            disp_dict_all.update(disp_dict)

            dom_loss_tgt_ins.backward()
            # clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer_dom.step()
            
        ####################################
        if not context and not dom_atten:
            # print("det src")
            optimizer.zero_grad()
            loss, tb_dict, disp_dict = model_func(model, batch, t_mode='det')
            disp_dict_all.update(disp_dict)
            loss.backward()
            # clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()

            if mcd:
                # print("mcd det tgt")
                optimizer.zero_grad()
                loss_tgt, tb_dict, disp_dict = model_func(model, batch, t_mode='mcd_tgt')
                disp_dict_all.update(disp_dict)
                loss_tgt.backward()
                # clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                optimizer.step()

        else:
            # print("det context src")
            optimizer.zero_grad()
            loss, tb_dict, disp_dict = model_func(model, batch, t_mode='dom_img_det')
            disp_dict_all.update(disp_dict)
            loss.backward()
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()

            if mcd:
                # print("mcd det context tgt")
                optimizer.zero_grad()
                loss_tgt, tb_dict, disp_dict = model_func(model, batch, t_mode='dom_img_det_tgt')
                disp_dict_all.update(disp_dict)
                loss_tgt.backward()
                clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                optimizer.step()#optimizer_mcd
            
        accumulated_iter += 1
        if ins_da:
            disp_dict.update({'loss': loss.item(), 'dom_loss_img': (dom_loss_src.item()+dom_loss_tgt.item())*0.5, 'dom_loss_ins': (dom_loss_src_ins.item()+dom_loss_tgt_ins.item())*0.5, 'lr': cur_lr})
        else:
            if mcd:
                disp_dict.update({'loss': loss.item(), 'loss_tgt': loss_tgt.item(), 'lr': cur_lr})
            else:
                disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
                
        
        # log to console and tensorboard
        if rank == 0:
            # disp_dict_all.update(disp_dict)
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict_all)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train_loss', loss, accumulated_iter)
                tb_log.add_scalar('learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train_' + key, val, accumulated_iter)

        # if cur_it > 10:
        if debug:
            break

    if rank == 0:
        pbar.close()
    return accumulated_iter



def train_one_epoch_da_pseudo(model, optimizer, pseudo_loader, model_func, lr_scheduler, accumulated_pseudo_iter, optim_cfg, rank, tbar, tb_log=None, leave_pbar=False, epoch=0, total_epochs=30, debug=False, context=False):

    total_it_each_epoch = len(pseudo_loader)
    dataloader_iter = iter(pseudo_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        # train batch and source batch 
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(pseudo_loader)
            batch = next(dataloader_iter)
            # print('new train iters')
        # lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        # if tb_log is not None:
        #     tb_log.add_scalar('learning_rate', cur_lr, accumulated_iter)

        model.train()

        ########### det ############
        optimizer.zero_grad()

        # print("pseudo batch", batch)
        # print("pseudo batch", [(i, batch[i].shape) for i in batch.keys() if 'batch' not in i])
        if context:
            t_mode = 'dom_img_det_pseudo'
        else:
            t_mode = 'pseudo'
        
        loss, tb_dict, disp_dict = model_func(model, batch, t_mode=t_mode)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        
        ####################################

        accumulated_pseudo_iter += 1
        disp_dict.update({'pseudo_loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_pseudo_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('pseudo_train_loss', loss, accumulated_pseudo_iter)
                tb_log.add_scalar('pseudo_learning_rate', cur_lr, accumulated_pseudo_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('pseudo_train_' + key, val, accumulated_pseudo_iter)

        if debug:
            break

    if rank == 0:
        pbar.close()
    return accumulated_pseudo_iter


def eval_single_ckpt(model, test_loader, ckpt, eval_output_dir, logger, epoch_id, distrib=False, cfg=None, load_ckpt=True, epoch_point=[], map_dict={}, pseudo_label=False, vis=False, debug=False, test_id=0, context=False, fpn_layers=[], map_dict_fpn={}, fpn_only=False, check_da=False, tsne=False, test_s_loader=None, map_dict_fuse={}, draw_matrix=False):
    # load checkpoint
    if load_ckpt: 
        print("load from checkpoint")
        model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=distrib)
        model.cuda()
    # start evaluation

    if check_da:
        if cfg.get('DATA_CONFIG_TEST', None) is not None:
            test_cfg = cfg.DATA_CONFIG_TEST
        else:
            test_cfg = cfg.DATA_CONFIG_TARGET
    else:
        test_cfg = cfg.DATA_CONFIG

    if test_loader.dataset.dataset_cfg.DATASET in ['NuscenesDataset']:
        out, pseudo_set = eval_utils.eval_one_epoch_nusc(
            cfg, model, test_loader, epoch_id, logger, distrib=distrib,
            result_dir=eval_output_dir, eval_location=test_cfg.EVAL_LOC, epoch_point=epoch_point, map_dict=map_dict, pseudo_label=pseudo_label, vis=vis, debug=debug, test_id=test_id, context=context, fpn_layers=fpn_layers, 
            map_dict_fpn=map_dict_fpn, fpn_only=fpn_only, tsne=tsne, test_s_loader=test_s_loader, draw_matrix=draw_matrix
        )
        return out, pseudo_set
    elif test_loader.dataset.dataset_cfg.DATASET in ['WaymoDataset']:
        out, pseudo_set = eval_utils.eval_one_epoch_nusc(
            cfg, model, test_loader, epoch_id, logger, distrib=distrib,
            result_dir=eval_output_dir, epoch_point=epoch_point, map_dict=map_dict, pseudo_label=pseudo_label, vis=vis, debug=debug, test_id=test_id, context=context, fpn_layers=fpn_layers, 
            map_dict_fpn=map_dict_fpn, fpn_only=fpn_only, tsne=tsne, test_s_loader=test_s_loader, draw_matrix=draw_matrix
        )
        return out, pseudo_set
    else:
        out, pseudo_set = eval_utils.eval_one_epoch(
            cfg, model, test_loader, epoch_id, logger, distrib=distrib,
            result_dir=eval_output_dir, epoch_point=epoch_point, map_dict=map_dict, pseudo_label=pseudo_label, vis=vis, debug=debug, test_id=test_id, context=context, fpn_layers=fpn_layers, 
            map_dict_fpn=map_dict_fpn, fpn_only=fpn_only, tsne=tsne, test_s_loader=test_s_loader, draw_matrix=draw_matrix
        )
        return out, pseudo_set

    return None, None
    
    # print('data', data)


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, lr_scheduler_dom=None, lr_warmup_scheduler_dom=None, ckpt_save_interval=1, max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, test=False, test_output_dir=None,out_logger=None,cfg=None,test_loader=None, da=False, dom_optimizer=None, source_loader=None, target_loader=None, dom_optim_cfg=None, da_w=0.1, l_ratio=1, max_epochs=0, pseudo_label=False, pseu_train_ratio=1.0, debug=False, vis=False,  pseudo_batch_size=1, pseudo_workers=1, start_pseudo_iter=0, distrib=False, double_test=False, select_prop=0.25, ins_da=False,context=False, pseu_bg_threshold=0.3, pseu_bg_default_w=0.5, select_posprop=0, select_negprop=0, fpn_layers=[], fpn_only=False, eval_last=False, range_inv=False, mcd=False, l_pow=10, mcd_curve=False, mcd_optimizer=None, dom_atten=False,tsne=False,test_s_loader=None, dom_optimizer2=None, dom_optim_cfg2=None, 
                lr_scheduler_dom2=None, lr_warmup_scheduler_dom2=None, dom_reg=False, draw_matrix=False, atten_w_optimizer=None):
    accumulated_iter = start_iter
    epoch_point = []
    epoch_point2 = []
    map_dict = {}
    map_dict2 = {}
    result_dict = {}
    result_dict2 = {}
    epoch_point_fpn = {}
    map_dict_fpn = {}
    map_dict_fuse = {}
    # result_dict_fpn = {}
    if len(fpn_layers) != 0:
        for i in fpn_layers:
            epoch_point_fpn[i] = []
            map_dict_fpn[i] = {}
            # result_dict_fpn[i] = {}

    if pseudo_label:
        pseudo_set = []
        accumulated_pseudo_iter = start_pseudo_iter

    if max_epochs == 0:
        max_epochs = total_epochs

    is_kitti = 'NuscenesDataset' not in test_loader.dataset.dataset_cfg.DATASET

    if eval_last:
        epoch_point.append(start_epoch)
        trained_epoch = start_epoch
        # ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
        # ckpt_name_pth = Path(f'{ckpt_name}.pth')

        # print("ckpt_save_dir", ckpt_save_dir)
        output_dir = ckpt_save_dir / '..' / 'eval'
        # print("output_dir", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_output_dir = output_dir / 'eval_last'
        # eval_output_dir = output_dir / ('epoch_%s' % trained_epoch)

        _, pseudo_set = eval_single_ckpt(model, test_loader, None, eval_output_dir, out_logger, trained_epoch, distrib=distrib, cfg=cfg, load_ckpt=False, epoch_point=epoch_point, map_dict=map_dict, pseudo_label=pseudo_label, vis=vis, debug=debug, test_id=0, context=context, fpn_layers=fpn_layers, map_dict_fpn=map_dict_fpn, fpn_only=fpn_only, check_da=da, tsne=tsne, test_s_loader=test_s_loader, draw_matrix=draw_matrix) #ckpt_name_pth, 

        common_utils.synchronize()
        
        # if pseudo_label:
        #     print("pseudo_set len", len(pseudo_set))
        #     print("test len", len(test_loader))

        torch.cuda.empty_cache()

        return


    with tqdm.trange(start_epoch, max_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            epoch_point.append(cur_epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            #######################
            # train one epoch
            #######################

            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
                cur_scheduler_dom = lr_warmup_scheduler_dom
                cur_scheduler_dom2 = lr_warmup_scheduler_dom2
            else:
                cur_scheduler = lr_scheduler
                cur_scheduler_dom = lr_scheduler_dom
                cur_scheduler_dom2 = lr_scheduler_dom2
            
            if da:
                accumulated_iter = train_one_epoch_da(
                    model, optimizer, dom_optimizer, 
                    train_loader, source_loader, target_loader, model_func,
                    lr_scheduler=cur_scheduler,
                    dom_lr_scheduler=cur_scheduler_dom,
                    accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                    rank=rank, tbar=tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == total_epochs),
                    total_it_each_epoch=total_it_each_epoch,
                    dataloader_iter=dataloader_iter,epoch=cur_epoch,total_epochs=total_epochs, debug=debug,
                    ins_da=ins_da, context=context, range_inv=range_inv, mcd=mcd, l_pow=l_pow, mcd_curve=mcd_curve,
                    optimizer_mcd=mcd_optimizer, dom_atten=dom_atten, optimizer_dom2=dom_optimizer2,
                    dom_lr_scheduler2=lr_scheduler_dom2, dom_reg=dom_reg, atten_w_optimizer=atten_w_optimizer
                )
            else:
                accumulated_iter = train_one_epoch(
                    model, optimizer, train_loader, model_func,
                    lr_scheduler=cur_scheduler,
                    accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                    rank=rank, tbar=tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == total_epochs),
                    total_it_each_epoch=total_it_each_epoch,
                    dataloader_iter=dataloader_iter,
                    debug=debug, range_inv=range_inv, mcd=mcd
                )

            torch.cuda.empty_cache()

            # # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                ckpt_name_pth = Path(f'{ckpt_name}.pth')
                # print('ckpt_name', ckpt_name)
                
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter, dom_optimizer=dom_optimizer), filename=ckpt_name
                )

            #######################
            # test one epoch
            #######################
            trained_epoch = cur_epoch + 1
            if test:
                # print("ckpt_save_dir", ckpt_save_dir)
                output_dir = ckpt_save_dir / '..' / 'eval'
                # print("output_dir", output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                eval_output_dir = output_dir 
                eval_output_dir = eval_output_dir / ('epoch_%s' % trained_epoch)

                result_dict, pseudo_set = eval_single_ckpt(model, test_loader, None, eval_output_dir, out_logger, trained_epoch, distrib=distrib, cfg=cfg, load_ckpt=False, epoch_point=epoch_point, map_dict=map_dict, pseudo_label=pseudo_label, vis=vis, debug=debug, test_id=0, context=context, fpn_layers=fpn_layers, map_dict_fpn=map_dict_fpn, fpn_only=fpn_only, tsne=tsne, test_s_loader=test_s_loader, draw_matrix=draw_matrix) #ckpt_name_pth, 

                common_utils.synchronize()
                
                # if pseudo_label:
                #     print("pseudo_set len", len(pseudo_set))
                #     print("test len", len(test_loader))

                torch.cuda.empty_cache()
                
            #######################
            # pseudo preprocess one epoch
            #######################
            pre_epochs = int(pseu_train_ratio * total_epochs)

            # print("cur_epoch", cur_epoch)
            # print("pre_epochs", pre_epochs)
            # balanced way - focal loss
            if pseudo_label and cur_epoch > pre_epochs:
                print("#################### TRAIN PSEUDO ####################")
                # sort pseudo_set top Pos:0%-25% Neg:100%-75%? (120 - 20) / (100 - 20)
                # max_pseudo_ratio 0.25
                if select_posprop == 0:
                    select_posprop = select_prop
                
                if select_negprop == 0:
                    select_negprop = select_prop

                pseudo_pos_progress = (cur_epoch - pre_epochs) * select_posprop / (total_epochs - pre_epochs) 
                pseudo_neg_progress = (cur_epoch - pre_epochs) * select_negprop / (total_epochs - pre_epochs)

                # print("pseudo_pos_progress", pseudo_pos_progress)
                # print("pseudo_neg_progress", pseudo_neg_progress)

                all_confidence = np.array([])

                for i in range(len(pseudo_set)):
                    all_confidence = np.append(all_confidence, pseudo_set[i]['pseudo_importance'], 0)

                # 0 - 0.25*100
                pseudo_progress_select_neg = int(pseudo_neg_progress * len(all_confidence)) 
                pseudo_progress_select_pos = int(pseudo_pos_progress * len(all_confidence))
                # 0.75*100 - 1
                # pseudo_progress_select_pos = -pseudo_progress_select_neg -1
                #int((1-pseudo_progress) * len(all_confidence)) - 1

                Sorted_confidence = np.sort(all_confidence)

                if len(Sorted_confidence) > 0:
                    neg_thresh = Sorted_confidence[pseudo_progress_select_neg-1]
                    pos_thresh = Sorted_confidence[-pseudo_progress_select_pos-1]
                else:
                    neg_thresh = -1
                    pos_thresh = -1

                def label_cond(x, neg_thresh, pos_thresh, pred_label):
                    if x <= neg_thresh and x <= pseu_bg_threshold:
                        return 0
                    elif x > pos_thresh:
                        return pred_label
                    else:
                        return -1
                def weights_cond(x, neg_thresh, pos_thresh, pred_label):
                    if x <= neg_thresh and x <= pseu_bg_threshold:
                        return pseu_bg_default_w - x
                    elif x > pos_thresh:
                        return x
                    else:
                        return 0

                for n in range(len(pseudo_set)):
                    pseudo_set[n]['pseudo_classes_filtered'] \
                        = np.array([label_cond(pseudo_set[n]['pseudo_importance'][i], 
                                    neg_thresh, pos_thresh, 
                                    pseudo_set[n]['pseudo_classes'][i]) 
                            for i in range(len(pseudo_set[n]['pseudo_importance']))])
                        
                    pseudo_set[n]['pseudo_weights'] \
                    = np.array([weights_cond(pseudo_set[n]['pseudo_importance'][i], 
                                neg_thresh, pos_thresh, 
                                pseudo_set[n]['pseudo_classes'][i]) 
                        for i in range(len(pseudo_set[n]['pseudo_importance']))])
                
                # print("pseudo_set[n]['pseudo_classes_filtered']", pseudo_set[n]['pseudo_classes_filtered'])
                # some 1 some 0 some -1

                #######################
                # pseudo train one epoch
                #######################
                
                # if pseudo_label and cur_epoch > pre_epochs:

                # init pseudo data loader
                # print("pseudo_set", pseudo_set)
                pseudo_workers=0

                pseudo_set, pseudo_loader, pseudo_sampler = build_pseudo_dataloader(
                    dataset_cfg=cfg.DATA_CONFIG_TARGET,
                    class_names=cfg.CLASS_NAMES,
                    batch_size=pseudo_batch_size,
                    dist=distrib, workers=pseudo_workers,
                    logger=out_logger,
                    root_path=Path(cfg.DATA_CONFIG_TARGET.DATA_PATH),
                    training=True,
                    pseudo_set=pseudo_set
                )

                if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                    cur_scheduler = lr_warmup_scheduler
                else:
                    cur_scheduler = lr_scheduler

                total_it_pseu_each_epoch = len(pseudo_loader)
                pseudo_dataloader_iter = iter(pseudo_loader)

                accumulated_pseudo_iter = train_one_epoch_da_pseudo(
                    model, optimizer, 
                    pseudo_loader, model_func,
                    cur_scheduler,
                    accumulated_pseudo_iter, optim_cfg,
                    rank, tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == total_epochs),
                    epoch=cur_epoch, total_epochs=total_epochs,
                    debug=debug, context=context
                )

                if double_test:
                    output_dir = ckpt_save_dir / '..' / 'eval'
                    # print("output_dir", output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    eval_output_dir = output_dir / 'eval'
                    eval_output_dir = eval_output_dir / ('epoch_%s' % trained_epoch)

                    epoch_point2.append(cur_epoch)

                    _, _ = eval_single_ckpt(model, test_loader, None, eval_output_dir, out_logger, trained_epoch, distrib=distrib, cfg=cfg, load_ckpt=False, epoch_point=epoch_point2, map_dict=map_dict2, pseudo_label=pseudo_label, vis=vis, debug=debug, test_id=1, context=context, fpn_layers=fpn_layers, fpn_only=fpn_only, tsne=tsne, test_s_loader=test_s_loader, draw_matrix=draw_matrix) #ckpt_name_pth, 

                    common_utils.synchronize()

                    torch.cuda.empty_cache()
            else:
                epoch_point2.append(cur_epoch)
                ##### no results for test 2 #####
                

                for k in cfg.CLASS_NAMES:
                # for k in ['car']:
                    if k in map_dict2.keys():
                        map_dict2[k][0].append(0)
                        map_dict2[k][1].append(0)
                        map_dict2[k][2].append(0)
                        if not is_kitti:
                            map_dict2[k][3].append(0)
                    else:
                        if not is_kitti:
                            map_dict2[k] = [[0], [0], [0], [0]]
                        else:
                            map_dict2[k] = [[0], [0], [0]]

            if rank == 0:
                draw_epoch_dir = ckpt_save_dir / '..'
                if double_test and not debug:
                    eval_utils.draw_overall_graph(draw_epoch_dir, res_dict=map_dict, res_dict2=map_dict2, epoch_point=epoch_point, kitti=is_kitti)#, classes=['car'])
                

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None, da=False, dom_optimizer=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    dom_optim_state = dom_optimizer.state_dict() if dom_optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version} #'dom_optimizer':dom_optimizer, 


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    if False and 'dom_optimizer_state' in state:
        dom_optimizer_state = state['dom_optimizer_state']
        state.pop('dom_optimizer_state', None)
        dom_optimizer_filename = '{}_dom_optim.pth'.format(filename)
        torch.save({'dom_optimizer_state': dom_optimizer_state}, dom_optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
