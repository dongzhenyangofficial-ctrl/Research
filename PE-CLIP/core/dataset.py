from datasets.ve8 import VE8Dataset
import paddle
# from paddle.io import DataLoader
class DataLoader(paddle.io.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=None,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False

        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
        if sampler is not None:
            self.batch_sampler.sampler = sampler

def get_ve8(opt, subset, transforms):
    spatial_transform, temporal_transform, target_transform, masking_generator = transforms
    return VE8Dataset(opt.video_path,
                      opt.audio_path,
                      opt.annotation_path,
                      subset,
                      opt.fps,
                      spatial_transform,
                      temporal_transform,
                      target_transform,
                      masking_generator,
                      need_audio=True)


def get_training_set(opt, spatial_transform, temporal_transform, target_transform, masking_generator):
    if opt.dataset == 've8':
        transforms = [spatial_transform, temporal_transform, target_transform, masking_generator]
        return get_ve8(opt, 'training', transforms)
    else:
        raise Exception


def get_validation_set(opt, spatial_transform, temporal_transform, target_transform, masking_generator):
    if opt.dataset == 've8':
        transforms = [spatial_transform, temporal_transform, target_transform, masking_generator]
        return get_ve8(opt, 'validation', transforms)
    else:
        raise Exception


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    if opt.dataset == 've8':
        transforms = [spatial_transform, temporal_transform, target_transform]
        return get_ve8(opt, 'validation', transforms)
    else:
        raise Exception


def get_data_loader(opt, dataset, shuffle, batch_size=0):
    batch_size = opt.batch_size if batch_size == 0 else batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt.n_threads,
        pin_memory=True,
        drop_last=opt.dl
    )
