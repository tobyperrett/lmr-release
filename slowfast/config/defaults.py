#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

from . import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "kinetics"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)

# Stop every epoch after this many iterations. Useful for debugging.
_C.TRAIN.MAX_ITERS_PER_EPOCH = 9999999999
_C.TRAIN.MAX_ITERS_PER_EPOCH_VAL = 9999999999
_C.TRAIN.MAX_ITERS_PER_EPOCH_TEST = 9999999999

_C.TRAIN.EXPERT_LOSS_ALPHA = 0.9

_C.TRAIN.SEP_REP_CLS_EPOCH = 9999
_C.TRAIN.CLS_BAL_EPOCH = 9999
_C.TRAIN.FREEZE_REP_EPOCH = 9999
_C.TRAIN.RESET_CLS_EPOCH = 9999

_C.TRAIN.RECONSTRUCT = False
_C.TRAIN.RECONSTRUCT_BANK_SIZE = 0
_C.TRAIN.RECONSTRUCT_CONTRIBUTION = "no-fst"
_C.TRAIN.RECONSTRUCT_APPLICATION = "fst"
_C.TRAIN.RECONSTRUCT_WEIGHT = 0.5
_C.TRAIN.RECONSTRUCT_ATTENTION_TYPE = "static"
_C.TRAIN.RECONSTRUCT_FST_THRESHOLD = 20
_C.TRAIN.RECONSTRUCT_L = 0.6
_C.TRAIN.RECONSTRUCT_H = 0.001
_C.TRAIN.RECONSTRUCT_D = 0.25

# _C.TRAIN.FRAMESTACK = False
# _C.TRAIN.FRAMESTACK_RATIO = 0.5

_C.TRAIN.FEATURE_MIXUP_PROB = 0.0

_C.TRAIN.DESCRIPTION = ""

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# Shuffle frames durin testing 
_C.TEST.SHUFFLE_FRAMES = False

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"

# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# ---------------------------------------------------------------------------- #
# X3D  options
# See https://arxiv.org/abs/2004.04730 for details about X3D Networks.
# ---------------------------------------------------------------------------- #
_C.X3D = CfgNode()

# Width expansion factor.
_C.X3D.WIDTH_FACTOR = 1.0

# Depth expansion factor.
_C.X3D.DEPTH_FACTOR = 1.0

# Bottleneck expansion factor for the 3x3x3 conv.
_C.X3D.BOTTLENECK_FACTOR = 1.0  #

# Dimensions of the last linear layer before classificaiton.
_C.X3D.DIM_C5 = 2048

# Dimensions of the first 3x3 conv layer.
_C.X3D.DIM_C1 = 12

# Whether to scale the width of Res2, default is false.
_C.X3D.SCALE_RES2 = False

# Whether to use a BatchNorm (BN) layer before the classifier, default is false.
_C.X3D.BN_LIN5 = False

# Whether to use channelwise (=depthwise) convolution in the center (3x3x3)
# convolution operation of the residual blocks.
_C.X3D.CHANNELWISE_3x3x3 = True

# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slowfast"

# Model name
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 400

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slow", "x3d", "fast"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"


# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# If True, use randaugment augmentation
_C.DATA.USE_RAND_AUGMENT = False

# If > 0.0, use random erasing augmentation
_C.DATA.RE_PROB = 0.0

# If True, use repeated aug
_C.DATA.USE_REPEATED_AUG = False

# If True, use color jitter augmentation
_C.DATA.COLORJITTER = False

# If True, use grayscale augmentation
_C.DATA.GRAYSCALE = False

# If True, use gaussian augmentation
_C.DATA.GAUSSIAN = False

# If True, use gaussian augmentation
_C.DATA.USE_RANDOM_RESIZE_CROPS = False


# ---------------------------------------------------------------------------- #
# Mixup options
# ---------------------------------------------------------------------------- #

_C.MIXUP = CfgNode()

# mixup alpha, mixup enabled if > 0. (default: 0.8)
_C.MIXUP.MIXUP_ALPHA = 0.0

# cutmix alpha, cutmix enabled if > 0. (default: 1.0)
_C.MIXUP.CUTMIX_ALPHA = 0.0

# cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)
_C.MIXUP.CUTMIX_MINMAX = None

# Probability of performing mixup or cutmix when either/both is enabled
_C.MIXUP.MIXUP_PROB = 1.0

# Probability of switching to cutmix when both mixup and cutmix enabled
_C.MIXUP.MIXUP_SWITCH_PROB = 0.5

# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.MIXUP.MIXUP_MODE = "batch"


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# Use Mixed Precision Training
_C.SOLVER.USE_MIXED_PRECISION = False

# If > 0.0, use label smoothing
_C.SOLVER.SMOOTHING = 0.0

# Clip Grad
_C.SOLVER.CLIP_GRAD = None

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = False

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# Distributed backend.
_C.USE_SBATCH = True

# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CfgNode()

# Number of epochs for data loading benchmark.
_C.BENCHMARK.NUM_EPOCHS = 5

# Log period in iters for data loading benchmark.
_C.BENCHMARK.LOG_PERIOD = 100

# If True, shuffle dataloader for epoch during benchmark.
_C.BENCHMARK.SHUFFLE = True


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()

# Whether enable video detection.
_C.DETECTION.ENABLE = False

# Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
_C.DETECTION.ALIGNED = True

# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7


# -----------------------------------------------------------------------------
# AVA Dataset options
# -----------------------------------------------------------------------------
_C.AVA = CfgNode()

# Directory path of frames.
_C.AVA.FRAME_DIR = "/mnt/fair-flash3-east/ava_trainval_frames.img/"

# Directory path for files of frame lists.
_C.AVA.FRAME_LIST_DIR = (
    "/mnt/vol/gfsai-flash3-east/ai-group/users/haoqifan/ava/frame_list/"
)

# Directory path for annotation files.
_C.AVA.ANNOTATION_DIR = (
    "/mnt/vol/gfsai-flash3-east/ai-group/users/haoqifan/ava/frame_list/"
)

# Filenames of training samples list files.
_C.AVA.TRAIN_LISTS = ["train.csv"]

# Filenames of test samples list files.
_C.AVA.TEST_LISTS = ["val.csv"]

# Filenames of box list files for training. Note that we assume files which
# contains predicted boxes will have a suffix "predicted_boxes" in the
# filename.
_C.AVA.TRAIN_GT_BOX_LISTS = ["ava_train_v2.2.csv"]
_C.AVA.TRAIN_PREDICT_BOX_LISTS = []

# Filenames of box list files for test.
_C.AVA.TEST_PREDICT_BOX_LISTS = ["ava_val_predicted_boxes.csv"]

# This option controls the score threshold for the predicted boxes to use.
_C.AVA.DETECTION_SCORE_THRESH = 0.9

# If use BGR as the format of input frames.
_C.AVA.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.AVA.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.AVA.TRAIN_PCA_JITTER_ONLY = True

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.AVA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.AVA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# Whether to do horizontal flipping during test.
_C.AVA.TEST_FORCE_FLIP = False

# Whether to use full test set for validation split.
_C.AVA.FULL_TEST_ON_VAL = False

# The name of the file to the ava label map.
_C.AVA.LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt"

# The name of the file to the ava exclusion.
_C.AVA.EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"

# The name of the file to the ava groundtruth.
_C.AVA.GROUNDTRUTH_FILE = "ava_val_v2.2.csv"

# Backend to process image, includes `pytorch` and `cv2`.
_C.AVA.IMG_PROC_BACKEND = "cv2"

# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.
_C.MULTIGRID.SHORT_CYCLE = False
# Short cycle additional spatial dimensions relative to the default crop size.
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]

_C.MULTIGRID.LONG_CYCLE = False
# (Temporal, Spatial) dimensions relative to the default shape.
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5 ** 0.5),
    (0.5, 0.5 ** 0.5),
    (0.5, 1),
    (1, 1),
]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0

# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False
# Provide path to prediction results for visualization.
# This is a pickle file of [prediction_tensor, label_tensor]
_C.TENSORBOARD.PREDICTIONS_PATH = ""
# Path to directory for tensorboard logs.
# Default to to cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = ""
# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
# This file must be provided to enable plotting confusion matrix
# by a subset or parent categories.
_C.TENSORBOARD.CLASS_NAMES_PATH = ""

# Path to a json file for categories -> classes mapping
# in the format {"parent_class": ["child_class1", "child_class2",...], ...}.
_C.TENSORBOARD.CATEGORIES_PATH = ""

# Config for confusion matrices visualization.
_C.TENSORBOARD.CONFUSION_MATRIX = CfgNode()
# Visualize confusion matrix.
_C.TENSORBOARD.CONFUSION_MATRIX.ENABLE = False
# Figure size of the confusion matrices plotted.
_C.TENSORBOARD.CONFUSION_MATRIX.FIGSIZE = [8, 8]
# Path to a subset of categories to visualize.
# File contains class names separated by newline characters.
_C.TENSORBOARD.CONFUSION_MATRIX.SUBSET_PATH = ""

# Config for histogram visualization.
_C.TENSORBOARD.HISTOGRAM = CfgNode()
# Visualize histograms.
_C.TENSORBOARD.HISTOGRAM.ENABLE = False
# Path to a subset of classes to plot histograms.
# Class names must be separated by newline characters.
_C.TENSORBOARD.HISTOGRAM.SUBSET_PATH = ""
# Visualize top-k most predicted classes on histograms for each
# chosen true label.
_C.TENSORBOARD.HISTOGRAM.TOPK = 10
# Figure size of the histograms plotted.
_C.TENSORBOARD.HISTOGRAM.FIGSIZE = [8, 8]

# Config for layers' weights and activations visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS = CfgNode()

# If False, skip model visualization.
_C.TENSORBOARD.MODEL_VIS.ENABLE = False

# If False, skip visualizing model weights.
_C.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS = False

# If False, skip visualizing model activations.
_C.TENSORBOARD.MODEL_VIS.ACTIVATIONS = False

# If False, skip visualizing input videos.
_C.TENSORBOARD.MODEL_VIS.INPUT_VIDEO = False


# List of strings containing data about layer names and their indexing to
# visualize weights and activations for. The indexing is meant for
# choosing a subset of activations outputed by a layer for visualization.
# If indexing is not specified, visualize all activations outputed by the layer.
# For each string, layer name and indexing is separated by whitespaces.
# e.g.: [layer1 1,2;1,2, layer2, layer3 150,151;3,4]; this means for each array `arr`
# along the batch dimension in `layer1`, we take arr[[1, 2], [1, 2]]
_C.TENSORBOARD.MODEL_VIS.LAYER_LIST = []
# Top-k predictions to plot on videos
_C.TENSORBOARD.MODEL_VIS.TOPK_PREDS = 1
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.COLORMAP = "Pastel2"
# Config for visualization video inputs with Grad-CAM.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM = CfgNode()
# Whether to run visualization using Grad-CAM technique.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE = True
# CNN layers to use for Grad-CAM. The number of layers must be equal to
# number of pathway(s).
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST = []
# If True, visualize Grad-CAM using true labels for each instances.
# If False, use the highest predicted class.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.USE_TRUE_LABEL = False
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.COLORMAP = "viridis"

# Config for visualization for wrong prediction visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.WRONG_PRED_VIS = CfgNode()
_C.TENSORBOARD.WRONG_PRED_VIS.ENABLE = False
# Folder tag to origanize model eval videos under.
_C.TENSORBOARD.WRONG_PRED_VIS.TAG = "Incorrectly classified videos."
# Subset of labels to visualize. Only wrong predictions with true labels
# within this subset is visualized.
_C.TENSORBOARD.WRONG_PRED_VIS.SUBSET_PATH = ""


# ---------------------------------------------------------------------------- #
# Demo options
# ---------------------------------------------------------------------------- #
_C.DEMO = CfgNode()

# Run model in DEMO mode.
_C.DEMO.ENABLE = False

# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
_C.DEMO.LABEL_FILE_PATH = ""

# Specify a camera device as input. This will be prioritized
# over input video if set.
# If -1, use input video instead.
_C.DEMO.WEBCAM = -1

# Path to input video for demo.
_C.DEMO.INPUT_VIDEO = ""
# Custom width for reading input video data.
_C.DEMO.DISPLAY_WIDTH = 0
# Custom height for reading input video data.
_C.DEMO.DISPLAY_HEIGHT = 0
# Path to Detectron2 object detection model configuration,
# only used for detection tasks.
_C.DEMO.DETECTRON2_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# Path to Detectron2 object detection model pre-trained weights.
_C.DEMO.DETECTRON2_WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
# Threshold for choosing predicted bounding boxes by Detectron2.
_C.DEMO.DETECTRON2_THRESH = 0.9
# Number of overlapping frames between 2 consecutive clips.
# Increase this number for more frequent action predictions.
# The number of overlapping frames cannot be larger than
# half of the sequence length `cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE`
_C.DEMO.BUFFER_SIZE = 0
# If specified, the visualized outputs will be written this a video file of
# this path. Otherwise, the visualized outputs will be displayed in a window.
_C.DEMO.OUTPUT_FILE = ""
# Frames per second rate for writing to output video file.
# If not set (-1), use fps rate from input file.
_C.DEMO.OUTPUT_FPS = -1
# Input format from demo video reader ("RGB" or "BGR").
_C.DEMO.INPUT_FORMAT = "BGR"
# Draw visualization frames in [keyframe_idx - CLIP_VIS_SIZE, keyframe_idx + CLIP_VIS_SIZE] inclusively.
_C.DEMO.CLIP_VIS_SIZE = 10
# Number of processes to run video visualizer.
_C.DEMO.NUM_VIS_INSTANCES = 2

# Path to pre-computed predicted boxes
_C.DEMO.PREDS_BOXES = ""
# Whether to run in with multi-threaded video reader.
_C.DEMO.THREAD_ENABLE = False
# Take one clip for every `DEMO.NUM_CLIPS_SKIP` + 1 for prediction and visualization.
# This is used for fast demo speed by reducing the prediction/visualiztion frequency.
# If -1, take the most recent read clip for visualization. This mode is only supported
# if `DEMO.THREAD_ENABLE` is set to True.
_C.DEMO.NUM_CLIPS_SKIP = 0
# Path to ground-truth boxes and labels (optional)
_C.DEMO.GT_BOXES = ""
# The starting second of the video w.r.t bounding boxes file.
_C.DEMO.STARTING_SECOND = 900
# Frames per second of the input video/folder of images.
_C.DEMO.FPS = 30
# Visualize with top-k predictions or predictions above certain threshold(s).
# Option: {"thres", "top-k"}
_C.DEMO.VIS_MODE = "thres"
# Threshold for common class names.
_C.DEMO.COMMON_CLASS_THRES = 0.7
# Theshold for uncommon class names. This will not be
# used if `_C.DEMO.COMMON_CLASS_NAMES` is empty.
_C.DEMO.UNCOMMON_CLASS_THRES = 0.3
# This is chosen based on distribution of examples in
# each classes in AVA dataset.
_C.DEMO.COMMON_CLASS_NAMES = [
    "watch (a person)",
    "talk to (e.g., self, a person, a group)",
    "listen to (a person)",
    "touch (an object)",
    "carry/hold (an object)",
    "walk",
    "sit",
    "lie/sleep",
    "bend/bow (at the waist)",
]
# Slow-motion rate for the visualization. The visualized portions of the
# video will be played `_C.DEMO.SLOWMO` times slower than usual speed.
_C.DEMO.SLOWMO = 1

# ---------------------------------------------------------------------------- #
# VIT options
# ---------------------------------------------------------------------------- #
_C.VIT = CfgNode()

# Patch-size spatial to tokenize input
_C.VIT.PATCH_SIZE = 16

# Patch-size temporal to tokenize input
_C.VIT.PATCH_SIZE_TEMP = 2

# Number of input channels
_C.VIT.CHANNELS = 3

# Embedding dimension
_C.VIT.EMBED_DIM = 768

# Depth of transformer: number of layers
_C.VIT.DEPTH = 12

# number of attention heads
_C.VIT.NUM_HEADS = 12

# expansion ratio for MLP
_C.VIT.MLP_RATIO = 4

# add bias to QKV projection layer
_C.VIT.QKV_BIAS = True

# video input
_C.VIT.VIDEO_INPUT = True

# temporal resolution i.e. number of frames
_C.VIT.TEMPORAL_RESOLUTION = 8

# use MLP classification head
_C.VIT.USE_MLP = False

# Dropout rate for
_C.VIT.DROP = 0.0

# Stochastic drop rate
_C.VIT.DROP_PATH = 0.0

# Dropout rate for MLP head
_C.VIT.HEAD_DROPOUT = 0.0

# Dropout rate for positional embeddings
_C.VIT.POS_DROPOUT = 0.0

# Dropout rate 
_C.VIT.ATTN_DROPOUT = 0.0

# Activation for head
_C.VIT.HEAD_ACT = "tanh"

# Use IM pretrained weights
_C.VIT.IM_PRETRAINED = True

# Pretrained weights type
_C.VIT.PRETRAINED_WEIGHTS = "vit_1k"

# Type of position embedding
_C.VIT.POS_EMBED = "separate"

# Self-Attention layer
_C.VIT.ATTN_LAYER = "trajectory"

# Flag to use original trajectory attn code
_C.VIT.USE_ORIGINAL_TRAJ_ATTN_CODE = True

# Approximation type
_C.VIT.APPROX_ATTN_TYPE = "none"

# Approximation Dimension
_C.VIT.APPROX_ATTN_DIM = 128

_C.VIT.NUM_EXPERT_BLOCKS = 0
_C.VIT.NUM_EXPERTS = 1



# -----------------------------------------------------------------------------
# EPIC-KITCHENS Dataset options
# -----------------------------------------------------------------------------
_C.EPICKITCHENS = CfgNode()

# Path to Epic-Kitchens RGB data directory
_C.EPICKITCHENS.VISUAL_DATA_DIR = "/raid/local_scratch/txp48-wwp01/frames"

# Path to Epic-Kitchens Annotation directory
_C.EPICKITCHENS.ANNOTATIONS_DIR = "/jmain02/home/J2AD001/wwp01/txp48-wwp01/epic-kitchens-100-annotations"

# List of EPIC-100 TRAIN files
_C.EPICKITCHENS.TRAIN_LIST = "EPIC_100_train.pkl"

# List of EPIC-100 VAL files
_C.EPICKITCHENS.VAL_LIST = "EPIC_100_validation.pkl"

# List of EPIC-100 TEST files
_C.EPICKITCHENS.TEST_LIST = "EPIC_100_validation.pkl"

# Testing split
_C.EPICKITCHENS.TEST_SPLIT = "validation"

# Use Train + Val
_C.EPICKITCHENS.TRAIN_PLUS_VAL = False

# Verb and noun weights.
_C.EPICKITCHENS.VERB_NOUN_ALPHA = 0.5

# Add custom config with default values.
custom_config.add_custom_config(_C)

_C.EPICKITCHENS.LT_TASK = "verb"


_C.EPICKITCHENS.TRAIN_VERB_COUNTS = [14848, 12225, 6927, 4870, 3483, 3016, 2293, 1742, 1861, 1595, 1574, 1652, 1065, 783, 737, 570, 367, 408, 371, 372, 336, 346, 340, 266, 267, 216, 232, 199, 195, 208, 199, 132, 147, 156, 173, 143, 123, 140, 108, 139, 116, 93, 91, 77, 100, 72, 69, 87, 93, 64, 80, 51, 68, 65, 61, 73, 55, 66, 75, 60, 53, 56, 43, 52, 45, 38, 55, 42, 40, 42, 26, 18, 33, 32, 36, 28, 34, 25, 21, 26, 11, 10, 12, 8, 11, 13, 14, 14, 11, 8, 2, 6, 4, 1, 3, 2, 2]
_C.EPICKITCHENS.TRAIN_NOUN_COUNTS = [3617, 2351, 2457, 2312, 2210, 2178, 1737, 1688, 1639, 1523, 1534, 1434, 1220, 1057, 1085, 981, 925, 998, 928, 919, 841, 910, 857, 835, 848, 884, 725, 689, 758, 642, 483, 562, 608, 497, 511, 513, 509, 347, 527, 532, 451, 437, 453, 450, 427, 411, 400, 346, 397, 439, 400, 371, 361, 385, 389, 312, 255, 340, 311, 245, 266, 234, 249, 291, 222, 203, 289, 261, 175, 167, 233, 211, 200, 184, 163, 236, 240, 166, 216, 181, 190, 207, 130, 178, 163, 139, 181, 127, 158, 148, 146, 146, 131, 154, 100, 137, 142, 149, 146, 95, 121, 141, 109, 125, 111, 103, 97, 94, 97, 120, 103, 78, 35, 98, 85, 90, 105, 59, 64, 86, 95, 73, 72, 57, 76, 80, 58, 52, 60, 68, 76, 63, 60, 65, 63, 52, 77, 44, 61, 70, 47, 59, 68, 41, 43, 46, 46, 65, 43, 47, 50, 61, 28, 47, 30, 40, 46, 43, 57, 58, 48, 49, 54, 35, 53, 33, 53, 27, 50, 31, 29, 35, 49, 48, 34, 35, 33, 31, 32, 24, 9, 25, 22, 29, 33, 39, 27, 24, 35, 33, 27, 27, 14, 12, 5, 29, 17, 17, 19, 27, 10, 8, 24, 19, 19, 0, 18, 25, 16, 21, 14, 19, 36, 23, 7, 7, 22, 11, 21, 21, 18, 20, 14, 18, 7, 16, 18, 16, 18, 18, 18, 17, 15, 17, 16, 12, 8, 7, 9, 14, 0, 14, 12, 7, 13, 9, 7, 10, 12, 10, 7, 5, 11, 4, 7, 10, 6, 9, 9, 9, 4, 9, 5, 8, 8, 6, 7, 6, 6, 7, 0, 2, 7, 6, 5, 0, 6, 5, 3, 5, 5, 5, 3, 4, 4, 2, 0, 3, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 2]

_C.EPICKITCHENS.VERB_DIST_SPLITS = [
    [0, 1, 2, 3, 4, 5, 6, 8, 7, 11, 9, 10, 12, 13, 14, 15, 17, 19, 18],
    [16, 21, 22, 20, 24, 23, 26, 25, 29, 27, 30, 28, 34, 33, 32, 35, 37, 39, 31, 36, 40, 38, 44, 41, 48, 42, 47, 50, 43, 58, 55, 45, 46, 52, 57, 53, 49, 54, 59, 61, 56, 66, 60, 63, 51, 64, 62, 67, 69, 68, 65, 74, 76, 72, 73, 75, 70, 79, 77, 78],
    [71, 86, 87, 85, 82, 80, 84, 88, 81, 83, 89, 91, 92, 94, 90, 95, 96]
]

_C.EPICKITCHENS.NOUN_DIST_SPLITS = [
    [0, 2, 1, 3, 4, 5, 6, 7, 8, 10, 9, 11, 12, 14, 13, 17, 15, 18, 16, 19, 21, 25, 22, 24, 20, 23, 28, 26, 27, 29, 32, 31, 39, 38, 35, 34, 36, 33, 30, 42, 40, 43, 49, 41, 44, 45, 46, 50, 48, 54, 53, 51, 52, 37, 47, 57, 55, 58, 63, 66, 60, 67, 56, 62, 59, 76, 75, 61, 70, 64, 78, 71, 81, 65, 72, 80, 73, 79, 86, 83, 68, 69, 77, 74, 84, 88, 93, 97, 89, 90, 91, 98, 96, 101, 85, 95, 92],
    [98, 96, 101, 85, 95, 92, 82, 87, 103, 100, 109, 104, 102, 116, 105, 110, 94, 113, 106, 108, 99, 120, 107, 115, 119, 114, 125, 111, 136, 124, 130, 121, 122, 139, 129, 142, 133, 147, 118, 131, 134, 138, 151, 128, 132, 117, 141, 126, 159, 123, 158, 162, 164, 166, 127, 135, 150, 168, 161, 172, 160, 173, 140, 149, 153, 145, 146, 156, 137, 144, 148, 157, 143, 155, 185, 212, 112, 163, 171, 175, 188, 174, 165, 176, 184, 189, 178, 169, 177, 154, 170, 183, 195, 152, 167, 186, 190, 191, 199, 181, 207, 179, 187, 202, 213, 182, 216, 209, 218, 219, 221],
    [198, 203, 204, 211, 206, 220, 223, 226, 228, 229, 230, 196, 197, 231, 233, 208, 225, 227, 234, 232, 192, 210, 222, 239, 241, 244, 193, 235, 242, 248, 217, 252, 200, 247, 249, 255, 180, 238, 245, 257, 258, 259, 261, 201, 236, 263, 264, 214, 215, 224, 237, 243, 246, 250, 254, 266, 269, 272, 256, 265, 267, 268, 273, 276, 194, 251, 262, 274, 277, 279, 280, 281, 253, 260, 283, 284, 278, 282, 287, 271, 285, 291, 292, 293, 297, 298, 299, 205, 240, 270, 275, 286, 288, 289, 290, 294, 295]
]

_C.SSV2 = CfgNode()
_C.SSV2.TRAIN_COUNTS = [1162, 837, 497, 487, 603, 1068, 2727, 449, 783, 890, 903, 982, 1122, 273, 972, 1738, 1459, 991, 1781, 1403, 1260, 689, 783, 669, 348, 225, 317, 1673, 1593, 1501, 1430, 1594, 1076, 639, 687, 680, 1520, 1563, 464, 501, 910, 881, 907, 2741, 853, 3170, 1253, 980, 790, 1195, 671, 91, 185, 291, 217, 1287, 2075, 679, 153, 873, 239, 274, 314, 320, 311, 796, 908, 1473, 1547, 523, 314, 697, 1044, 1204, 1391, 669, 337, 322, 169, 408, 717, 1204, 924, 915, 725, 462, 1555, 1587, 290, 548, 493, 292, 199, 2949, 2724, 526, 351, 573, 962, 1687, 1874, 1452, 955, 1103, 1204, 837, 2188, 2031, 396, 3284, 507, 156, 1608, 364, 1766, 368, 361, 591, 609, 733, 982, 903, 1255, 936, 837, 1678, 832, 922, 709, 1638, 1213, 327, 462, 448, 1100, 1390, 111, 162, 266, 747, 1071, 415, 400, 1936, 1032, 1379, 2275, 1032, 1699, 1736, 1285, 2254, 1254, 943, 774, 875, 604, 928, 441, 326, 1471, 471, 748, 183, 2058, 867, 1029, 1018, 851, 240, 724, 2426, 840, 539]
_C.SSV2.DIST_SPLITS = [
    [109, 45, 93, 43, 6, 94, 171, 146, 151, 106, 56, 164, 107, 143],
    [100, 18, 114, 15, 149, 148, 99, 125, 27, 129, 112, 31, 28, 87, 37, 86, 68, 36, 29, 67, 160, 16, 101, 30, 19, 74, 135, 145, 55, 150, 20, 122, 152, 46, 130, 73, 81, 104, 49, 0, 12, 103, 134, 32, 140, 5, 72, 144, 147, 166, 167, 17, 11, 120, 47, 14, 98, 102, 153, 123, 157, 82, 127, 83, 40, 66, 42, 10, 121, 9, 41, 155, 59, 165, 44, 168, 172, 1, 105, 124, 126, 65, 48, 8, 22, 154, 162, 139, 119, 84, 170, 80, 128, 71, 21, 34, 35, 50, 57, 23, 33, 75, 118],
    [4, 156, 97, 117, 89, 173, 69, 95, 39, 110, 2, 90, 3, 38, 161, 7, 85, 132, 133, 141, 158, 79, 108, 142, 96, 113, 115, 116, 24, 76, 131, 159, 26, 62, 63, 70, 77, 53, 61, 64, 88, 91, 13, 25, 54, 60, 92, 138, 169, 51, 52, 58, 78, 111, 136, 137],
]


def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
