import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from pcdet.config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
import torch.distributed as dist

from pathlib import Path
import argparse
import datetime
import glob
def str2bool(v):
      return v.lower() in ("yes", "true", "t", "1")

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=30, required=False, help='number of epochs to train for')
    parser.add_argument('--max_epochs', type=int, default=40, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--out_dir', type=str, default='.')
    parser.add_argument('--test_batch_size', type=int, default=1, help='')
    parser.add_argument('--trainval', type=str2bool, default=False, help='')
    parser.add_argument('--pseudo_label', type=str2bool, default=False, help='')
    parser.add_argument('--debug', type=str2bool, default=False, help='')
    parser.add_argument('--vis', type=str2bool, default=False, help='')
    parser.add_argument('--pin_memory', type=str2bool, default=True, help='')
    parser.add_argument('--double_test', type=str2bool, default=False, help='')
    parser.add_argument('--ins_da', type=str2bool, default=False, help='')
    parser.add_argument('--context', type=str2bool, default=True, help='')
    parser.add_argument('--draw_matrix', type=str2bool, default=False, help='')


    parser.add_argument('--select_prop', type=float, default=0.25, help='')
    parser.add_argument('--select_posprop', type=float, default=0.3, help='')
    parser.add_argument('--select_negprop', type=float, default=0.1, help='')

    parser.add_argument('--pseu_bg_default_w', type=float, default=0.5, help='')    
    parser.add_argument('--pseu_bg_threshold', type=float, default=0.25, help='')
    parser.add_argument('--pseu_train_ratio', type=float, default=1.0, help='')
    
    parser.add_argument('--points_range', type=str2bool, default=False)
    parser.add_argument('--eval_last', type=str2bool, default=False)
    parser.add_argument('--fpn_only', type=str2bool, default=False)
    parser.add_argument('--range_inv', type=str2bool, default=False)
    parser.add_argument('--mcd', type=str2bool, default=False)
    parser.add_argument('--mcd_curve', type=str2bool, default=False)
    parser.add_argument('--dom_atten', type=str2bool, default=False)
    parser.add_argument('--tsne', type=str2bool, default=False)

    parser.add_argument('--voxel_attention', type=str2bool, default=False)
    

    
    parser.add_argument(
        "--da_w",
        type=float,
        default=0.1,
        help="domain adaptation",
    )
    parser.add_argument(
        "--l_ratio",
        type=float,
        default=1.0,
        help="domain adaptation",
    )
    parser.add_argument(
        "--l_pow",
        type=float,
        default=10,
        help="domain adaptation l under pow",
    )

    args = parser.parse_args()

    print("args", args)
    print("cfg", cfg)
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'+'/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
    else:
        print("args.batch_size, args.tcp_port, args.local_rank", args.batch_size, args.tcp_port, args.local_rank)
        args.batch_size, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.batch_size, args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True
    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    print('args fix_random_seed', args.fix_random_seed)

    output_dir = Path(args.out_dir)
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        total_gpus = dist.get_world_size()
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------

    check_da = cfg.DATA_CONFIG.get('DA', False) == True
    
    logger.info(f'Using Domain Adaptation: {check_da}')

    logger.info('before build_dataloader')
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
        training=True,
        pin_memory=args.pin_memory,
        points_range=args.points_range
    )
    
    
    if check_da:
        source_set, source_loader, source_sampler = train_set, train_loader, train_sampler
        # build_dataloader(
        #     dataset_cfg=cfg.DATA_CONFIG,
        #     class_names=cfg.CLASS_NAMES,
        #     batch_size=args.batch_size,
        #     dist=dist_train, workers=args.workers,
        #     logger=logger,
        #     root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
        #     training=True,
        #     pin_memory=args.pin_memory
        # ) #args.workers,
        target_set, target_loader, target_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG_TARGET,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_train, workers=args.workers,
            logger=logger,
            root_path=Path(cfg.DATA_CONFIG_TARGET.DATA_PATH),
            training=True,
            pin_memory=args.pin_memory,
            points_range=args.points_range
        )#args.workers,
    else:
        source_set, source_loader, source_sampler = None, None, None
        target_set, target_loader, target_sampler = None, None, None

    single_collate_fn = True if args.pseudo_label else False

    if check_da:
        if cfg.get('DATA_CONFIG_TEST', None) is not None:
            test_cfg = cfg.DATA_CONFIG_TEST
        else:
            test_cfg = cfg.DATA_CONFIG_TARGET
    else:
        test_cfg = cfg.get('DATA_CONFIG_TARGET', cfg.DATA_CONFIG)
        # test_cfg = cfg.DATA_CONFIG
    
    print("test_cfg", test_cfg)

    test_set, test_loader, test_sampler = build_dataloader(
        dataset_cfg=test_cfg,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.test_batch_size,
        dist=dist_train, workers=0,
        logger=logger, 
        root_path=Path(test_cfg.get('DATA_PATH','')),
        training=False,
        vis=args.vis,
        single_collate_fn=single_collate_fn,
        pin_memory=args.pin_memory,
        points_range=args.points_range
    )

    if args.tsne:
        test_s_config = cfg.DATA_CONFIG_SOURCE_TEST
        test_s_set, test_s_loader, test_s_sampler = build_dataloader(
            dataset_cfg=test_s_config,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.test_batch_size,
            dist=dist_train, workers=0,
            logger=logger, 
            root_path=Path(test_s_config.DATA_PATH),
            training=False,
            vis=args.vis,
            single_collate_fn=single_collate_fn,
            pin_memory=args.pin_memory,
            points_range=args.points_range
        )
    else:
        test_s_loader = None
    
    logger.info('after build_dataloader')
    
    # disable velo
    check_nusc = False#cfg.DATA_CONFIG.DATASET == 'NuscenesDataset'

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set, nusc=check_nusc)
    
    # test_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set, nusc=check_nusc)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    if check_da:
        dom_optimizer = build_optimizer(model, cfg.DOM_OPTIMIZATION, dom_opt=True,feat_lr_ratio=cfg.DOM_OPTIMIZATION.FEAT_LR_RATIO) 
    else:
        dom_optimizer = None
    
    if cfg.MODEL.get('DENSE_HEAD', {}).get('DIFF_DOM_OPT', False) or cfg.MODEL.get('DENSE_HEAD', {}).get('TWO_DOM_REG', False):
        two_dom = True
        dom_optimizer2 = build_optimizer(model, cfg.DOM_OPTIMIZATION2, dom_opt=True,feat_lr_ratio=cfg.DOM_OPTIMIZATION2.FEAT_LR_RATIO) 
        if cfg.MODEL.DENSE_HEAD.get('TWO_DOM_REG', False):
            dom_reg = True
        else:
            dom_reg = False
    else:
        two_dom = False
        dom_reg = False
        dom_optimizer2 = None

    if args.mcd:
        mcd_optimizer = build_optimizer(model, cfg.OPTIMIZATION, mcd_opt=True,      
                                        feat_lr_ratio=0) 
    else:
        mcd_optimizer = None

    if args.voxel_attention:
        atten_w_optimizer = build_optimizer(model, cfg.ATT_W_OPTIMIZATION, atten_opt=True, use_att=True) 
    else:
        atten_w_optimizer = None

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)

    # check ckpt exist
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger, dom_optimizer=dom_optimizer)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist, optimizer=optimizer, dom_optimizer=dom_optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=True)
    logger.info(model)

    total_iters_each_epoch = len(train_loader) if not args.merge_all_iters_to_one_epoch else len(train_loader) // args.epochs
    
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    if check_da:
        lr_scheduler_dom, lr_warmup_scheduler_dom = build_scheduler(
            dom_optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs,
            last_epoch=last_epoch, optim_cfg=cfg.DOM_OPTIMIZATION
        )
        dom_opt_cfg = cfg.DOM_OPTIMIZATION
        if two_dom:
            lr_scheduler_dom2, lr_warmup_scheduler_dom2 = build_scheduler(
                dom_optimizer2, total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs,
                last_epoch=last_epoch, optim_cfg=cfg.DOM_OPTIMIZATION2
            )
            dom_opt_cfg2 = cfg.DOM_OPTIMIZATION2
        else:
            lr_scheduler_dom2, lr_warmup_scheduler_dom2 = None, None
            dom_opt_cfg2 = None
    else:
        lr_scheduler_dom, lr_warmup_scheduler_dom = None, None
        lr_scheduler_dom2, lr_warmup_scheduler_dom2 = None, None

        dom_opt_cfg = None
        dom_opt_cfg2 = None

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    num_fpn_up = cfg.MODEL.get('FPN_UP_LAYERS',0)
    num_fpn_down = cfg.MODEL.get('FPN_DOWN_LAYERS',0)
    num_fpn_downup = cfg.MODEL.get('FPN_DOWNUP_LAYERS',0)
    fpn_layers = [str(3 - l) for l in range(num_fpn_up)] + [str(4 + l) for l in range(num_fpn_downup+1) if num_fpn_downup > 0] + [str(4 + 1 + l) for l in range(num_fpn_down) if num_fpn_down > 0]
    print("l_pow", args.l_pow)
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        lr_scheduler_dom=lr_scheduler_dom,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        lr_warmup_scheduler_dom=lr_warmup_scheduler_dom,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        test=args.trainval,
        eval_last=args.eval_last,
        test_output_dir=args.out_dir,
        out_logger=logger,
        cfg=cfg,
        test_loader=test_loader,
        da=check_da,
        dom_optimizer=dom_optimizer, 
        source_loader=source_loader, 
        target_loader=target_loader,  
        dom_optim_cfg=dom_opt_cfg,
        da_w=args.da_w,
        l_ratio=args.l_ratio,
        max_epochs=args.max_epochs,
        pseudo_label=args.pseudo_label,
        pseu_train_ratio=args.pseu_train_ratio,
        debug=args.debug,
        vis=args.vis,
        pseudo_batch_size=args.batch_size,
        pseudo_workers=args.workers,
        distrib=dist_train,
        double_test=args.double_test,
        ins_da=args.ins_da,
        context=args.context,
        pseu_bg_threshold=args.pseu_bg_threshold,
        pseu_bg_default_w=args.pseu_bg_default_w,
        select_prop=args.select_prop,
        select_posprop=args.select_posprop,
        select_negprop=args.select_negprop,
        fpn_layers=fpn_layers,
        fpn_only=args.fpn_only,
        range_inv=args.range_inv,
        l_pow=args.l_pow,
        mcd=args.mcd,
        mcd_curve=args.mcd_curve,
        dom_atten=args.dom_atten,
        mcd_optimizer=mcd_optimizer,
        tsne=args.tsne,
        test_s_loader=test_s_loader,
        dom_optimizer2=dom_optimizer2,
        dom_optim_cfg2=dom_opt_cfg2,
        lr_scheduler_dom2=lr_scheduler_dom2,
        lr_warmup_scheduler_dom2=lr_warmup_scheduler_dom2,
        dom_reg=dom_reg,
        draw_matrix=args.draw_matrix,
        atten_w_optimizer=atten_w_optimizer
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
