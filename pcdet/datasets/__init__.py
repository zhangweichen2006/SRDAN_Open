import torch
from torch.utils.data import DataLoader
from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuscenesDataset
from .astar3d.astar3d_dataset import AStar3DDataset
from .presil.presil_dataset import PreSILDataset
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuscenesDataset': NuscenesDataset,
    'AStar3DDataset':AStar3DDataset,
    'PreSILDataset':PreSILDataset,
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, vis=False, single_collate_fn=False, pin_memory=True, points_range=False):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
        points_range=points_range,
        vis=vis
    )
    #

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    # if single_collate_fn:
    #     fn = dataset.collate_batch_pseudo
    # else:
    #     fn = dataset.collate_batch
    fn = dataset.collate_batch

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=fn,
        drop_last=False, sampler=sampler, timeout=0
    )

    # logger.info(iter(dataloader).next())
    # print("item0", iter(dataloader).next()[0])

    return dataset, dataloader, sampler

def build_pseudo_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, pseudo_set=None, pin_memory=True):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
        pseudo=True,
        pseudo_set=pseudo_set,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None


    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler
