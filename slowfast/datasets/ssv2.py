#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import numpy as np
import os
import random
from itertools import chain as chain
import torch
import torch.utils.data
from torchvision import transforms
from iopath.common.file_io import g_pathmgr

import slowfast.utils.logging as logging

from . import autoaugment as autoaugment
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ssv2(torch.utils.data.Dataset):
    """
    Something-Something v2 (SSV2) video loader. Construct the SSV2 video loader,
    then sample clips from the videos. For training and validation, a single
    clip is randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Load Something-Something V2 data (frame paths, labels, etc. ) to a given
        Dataset object. The dataset could be downloaded from Something-Something
        official website (https://20bn.com/datasets/something-something).
        Please see datasets/DATASET.md for more information about the data format.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries for reading frames from disk.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Something-Something V2".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Something-Something V2 {}...".format(mode))
        self._construct_loader()

        self.label_dict = {}
        for c in range(max(self._labels)):
            indexes = [index for (index, item) in enumerate(self._labels) if item == c]
            if len(indexes) > 0:
                self.label_dict[c] = indexes            

        self.class_balancing = False


    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # Loading label names.
        with g_pathmgr.open(
            os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR,
                "something-something-v2-labels.json",
            ),
            "r",
        ) as f:
            label_dict = json.load(f)

        # Loading labels.
        label_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            "something-something-v2-{}.json".format(
                "train" if self.mode == "train" else "validation"
            ),
        )
        with g_pathmgr.open(label_file, "r") as f:
            label_json = json.load(f)

        self._video_names = []
        self._labels = []
        for video in label_json:
            video_name = video["id"]
            template = video["template"]
            template = template.replace("[", "")
            template = template.replace("]", "")
            label = int(label_dict[template])
            self._video_names.append(video_name)
            self._labels.append(label)

        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            "{}.csv".format("train" if self.mode == "train" else "val"),
        )
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos, _ = utils.load_image_lists(
            path_to_file, self.cfg.DATA.PATH_PREFIX
        )

        # assert len(self._path_to_videos) == len(self._video_names), (
        #     len(self._path_to_videos),
        #     len(self._video_names),
        # )

        # From dict to list.
        new_paths, new_labels = [], []
        for index in range(len(self._video_names)):
            if self._video_names[index] in self._path_to_videos:
                new_paths.append(self._path_to_videos[self._video_names[index]])
                new_labels.append(self._labels[index])

        self._labels = new_labels
        self._path_to_videos = new_paths

        # Extend self when self._num_clips > 1 (during testing).
        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )
        )
        self._labels = list(
            chain.from_iterable([[x] * self._num_clips for x in self._labels])
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(len(self._path_to_videos))
                ]
            )
        )
        logger.info(
            "Something-Something V2 dataloader constructed "
            " (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def get_seq_frames(self, index):
        """
        Given the video index, return the list of sampled frame indexes.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        num_frames = self.cfg.DATA.NUM_FRAMES
        video_length = len(self._path_to_videos[index])

        seg_size = float(video_length - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.mode == "train":
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        return seq


    def set_class_balancing(self):
        self.class_balancing = True

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """

        # change index if class balancing:
        if self.class_balancing:
            action_class, action_class_indexes = random.choice(list(self.label_dict.items()))
            index = random.choice(action_class_indexes)



        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        label = self._labels[index]

        seq = self.get_seq_frames(index)

        # print(label, self._path_to_videos[index][0])

        frames = torch.as_tensor(
            utils.retry_load_images(
                [self._path_to_videos[index][frame] for frame in seq],
                self._num_retries,
            )
        )

        if self.cfg.DATA.USE_RAND_AUGMENT and self.mode in ["train"]:
            # Transform to PIL Image
            frames = [transforms.ToPILImage()(frame.squeeze().numpy()) for frame in frames]

            # Perform RandAugment
            img_size_min = crop_size
            auto_augment_desc = "rand-m20-mstd0.5-inc1"
            aa_params = dict(
                translate_const=int(img_size_min * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in self.cfg.DATA.MEAN]),
            )
            seed = random.randint(0, 100000000)
            frames = [autoaugment.rand_augment_transform(
                auto_augment_desc, aa_params, seed)(frame) for frame in frames]

            # To Tensor: T H W C
            frames = [torch.tensor(np.array(frame)) for frame in frames]
            frames = torch.stack(frames)

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        use_random_resize_crop = self.cfg.DATA.USE_RANDOM_RESIZE_CROPS
        if use_random_resize_crop:
            if self.mode in ["train", "val"]:
                frames = transform.random_resize_crop_video(
                    frames, crop_size, 
                    scale=(0.05, 1.0), 
                    interpolation_mode="bilinear")
                frames, _ = transform.horizontal_flip(0.5, frames)
            else:
                assert len({min_scale, max_scale, crop_size}) == 1
                frames, _ = transform.random_short_side_scale_jitter(
                    frames, min_scale, max_scale
                )
                frames, _ = transform.uniform_crop(
                    frames, crop_size, spatial_sample_index)
        else:
            # Perform data augmentation.
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

        # T H W C -> T C H W.
        if self.mode in ["train", "val"]:
            frames = frames.permute(1, 0, 2, 3)
            frames = utils.frames_augmentation(
                frames, 
                colorjitter=self.cfg.DATA.COLORJITTER,
                use_grayscale=self.cfg.DATA.GRAYSCALE,
                use_gaussian=self.cfg.DATA.GAUSSIAN
            )

        frames = utils.pack_pathway_output(self.cfg, frames)

        
        vid_folder = os.path.split(self._path_to_videos[index][0])[0]
        vid_id = os.path.split(vid_folder)[-1]

        metadata = {"vid_id": vid_id}

        return frames, label, index, metadata

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
