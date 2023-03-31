#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pickle
import pprint
from timm.data import Mixup
import torch
import torch.nn.functional as F
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

from slowfast.config.defaults import get_cfg
import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import TrainMeter, ValMeter, EPICTrainMeter, EPICValMeter
from slowfast.utils.multigrid import MultigridSchedule
from timm.utils import NativeScaler

from einops import rearrange


from inspect import signature

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, 
    writer=None, loss_scaler=None, loss_fun=None, mixup_fn=None, ap=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()


    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (inputs, labels, index, meta) in enumerate(train_loader):

        if cur_iter > cfg.TRAIN.MAX_ITERS_PER_EPOCH:
            break

        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if not isinstance(val[i], (str,)):
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

 

        if mixup_fn is not None:
            if cfg.TRAIN.DATASET == "Epickitchens":
                labels['verb'] = labels['verb'].cuda()
                labels['noun'] = labels['noun'].cuda()

                if cfg.EPICKITCHENS.LT_TASK == "verb":
                    inputs, labels['verb'] = mixup_fn(inputs[0], labels['verb'])
                elif cfg.EPICKITCHENS.LT_TASK == "noun":
                    inputs, labels['noun'] = mixup_fn(inputs[0], labels['noun'])
                else:
                    raise NotImplementedError()
                inputs = [inputs]

            else:
                labels = labels.cuda()
                inputs, labels = mixup_fn(inputs[0], labels)
                inputs = [inputs]
            


        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.SOLVER.USE_MIXED_PRECISION):
            
            preds, labels = model(inputs, labels=labels)

            # print(torch.max(labels))


            if mixup_fn is None:
                if isinstance(labels, (dict,)):
                    labels = {k: v.cuda() for k, v in labels.items()}
                else:
                    labels = labels.cuda()
            global_step = data_size * cur_epoch + cur_iter

            if isinstance(labels, (dict,)) and cfg.TRAIN.DATASET == "Epickitchens":

                if len(labels['verb'].shape) > len(labels['noun'].shape):
                    labels['noun'] = F.one_hot(labels['noun'], num_classes=300)
                elif len(labels['noun'].shape) > len(labels['verb'].shape):
                    labels['verb'] = F.one_hot(labels['verb'], num_classes=97)


###
                # if mixup_fn is None:
                #     if len(labels['verb'].shape) > 1:
                #         labels['verb'] = labels['verb'].long()
                #     if len(labels['noun'].shape) > 1:
                #         labels['noun'] = labels['noun'].long()

                # Compute the loss.
                if cfg.VIT.NUM_EXPERTS > 1:
                    loss_verb = loss_fun(preds[0], labels['verb'], 'verb')
                    loss_noun = loss_fun(preds[1], labels['noun'], 'noun')
                elif "label_set" in signature(loss_fun).parameters:    
                    loss_verb = loss_fun(preds[0][1], labels['verb'], 'verb')
                    loss_noun = loss_fun(preds[1][1], labels['noun'], 'noun')
                else:
                    # print(labels['verb'].shape, labels['noun'].shape)
                    loss_verb = loss_fun(preds[0][1], labels['verb'])
                    loss_noun = loss_fun(preds[1][1], labels['noun'])
                loss = cfg.EPICKITCHENS.VERB_NOUN_ALPHA * loss_verb + (1.0 - cfg.EPICKITCHENS.VERB_NOUN_ALPHA) * loss_noun
            else:

###
                # if mixup_fn is None:
                #     if len(labels.shape) > 1:
                #         labels = labels.long()
                if cfg.VIT.NUM_EXPERTS > 1:
                    loss = loss_fun(preds, labels)
                else:
                    loss = loss_fun(preds[1], labels)



        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        if cfg.SOLVER.USE_MIXED_PRECISION: # Mixed Precision Training
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=cfg.SOLVER.CLIP_GRAD,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            # Update the parameters.
            optimizer.step()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None

            if isinstance(labels, (dict,)) and cfg.TRAIN.DATASET == "Epickitchens":
                if cfg.TRAIN.FEATURE_MIXUP_PROB or cfg.MIXUP.MIXUP_ALPHA:
                    labels['verb'] = torch.argmax(labels['verb'], dim=-1)
                    labels['noun'] = torch.argmax(labels['noun'], dim=-1)

                if len(labels['verb'].shape) > len(labels['noun'].shape):
                    labels['noun'] = torch.argmax(labels['noun'], dim=-1)

                if len(labels['noun'].shape) > len(labels['verb'].shape):
                    labels['verb'] = torch.argmax(labels['verb'], dim=-1)

                # Compute the verb accuracies.
                verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(
                    preds[0][0], labels['verb'], (1, 5))
                verb_lt, verb_lt_cor, verb_lt_count, _, _ = metrics.lt_accuracies(preds[0][0], labels['verb'], cfg.EPICKITCHENS.TRAIN_VERB_COUNTS, cfg.EPICKITCHENS.VERB_DIST_SPLITS)

                # Gther all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_verb, verb_top1_acc, verb_top5_acc, verb_lt, verb_lt_cor, verb_lt_count = du.all_reduce(
                        [loss_verb, verb_top1_acc, verb_top5_acc, verb_lt, verb_lt_cor, verb_lt_count]
                    )


                # Copy the stats from GPU to CPU (sync point).
                loss_verb, verb_top1_acc, verb_top5_acc = (
                    loss_verb.item(),
                    verb_top1_acc.item(),
                    verb_top5_acc.item(),
                )
                # verb_lt = [v.item() for v in verb_lt]
                verb_lt = verb_lt.cpu().detach().numpy()
                verb_lt_cor = verb_lt_cor.cpu().detach().numpy()
                verb_lt_count = verb_lt_count.cpu().detach().numpy()

                # Compute the noun accuracies.
                noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(
                    preds[1][0], labels['noun'], (1, 5))
                noun_lt, noun_lt_cor, noun_lt_count, _, _ = metrics.lt_accuracies(preds[1][0], labels['noun'], cfg.EPICKITCHENS.TRAIN_NOUN_COUNTS, cfg.EPICKITCHENS.NOUN_DIST_SPLITS)

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_noun, noun_top1_acc, noun_top5_acc, noun_lt, noun_lt_cor, noun_lt_count = du.all_reduce(
                        [loss_noun, noun_top1_acc, noun_top5_acc, noun_lt, noun_lt_cor, noun_lt_count]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_noun, noun_top1_acc, noun_top5_acc = (
                    loss_noun.item(),
                    noun_top1_acc.item(),
                    noun_top5_acc.item(),
                )
                # noun_lt = [n.item() for n in noun_lt]
                noun_lt = noun_lt.cpu().detach().numpy()
                noun_lt_cor = noun_lt_cor.cpu().detach().numpy()
                noun_lt_count = noun_lt_count.cpu().detach().numpy()

                # Compute the action accuracies.
                action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
                    (preds[0][0], preds[1][0]),
                    (labels['verb'], labels['noun']),
                    (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, action_top1_acc, action_top5_acc = du.all_reduce(
                        [loss, action_top1_acc, action_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, action_top1_acc, action_top5_acc = (
                    loss.item(),
                    action_top1_acc.item(),
                    action_top5_acc.item(),
                )



                # Update and log stats.
                train_meter.update_stats(
                    (verb_top1_acc, noun_top1_acc, action_top1_acc),
                    (verb_top5_acc, noun_top5_acc, action_top5_acc),
                    (verb_lt, noun_lt, verb_lt_cor, noun_lt_cor, verb_lt_count, noun_lt_count),
                    (loss_verb, loss_noun, loss),
                    lr, inputs[0].size(0) * cfg.NUM_GPUS
                )
            else:
                if cfg.TRAIN.FEATURE_MIXUP_PROB or cfg.MIXUP.MIXUP_ALPHA:
                    labels = torch.argmax(labels, dim=-1)

                num_topks_correct = metrics.topks_correct(preds[0], labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds[0].size(0)) * 100.0 for x in num_topks_correct
                ]

                lt, lt_cor, lt_count, _, _ = metrics.lt_accuracies(preds[0], labels, cfg.SSV2.TRAIN_COUNTS, cfg.SSV2.DIST_SPLITS)

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err, lt, lt_cor, lt_count = du.all_reduce(
                        [loss, top1_err, top5_err, lt, lt_cor, lt_count]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )
                lt = lt.cpu().detach().numpy()
                lt_cor = lt_cor.cpu().detach().numpy()
                lt_count = lt_count.cpu().detach().numpy()

                # Update and log stats.
                train_meter.update_stats(
                    top1_err,
                    top5_err,
                    (lt, lt_cor, lt_count),
                    loss,
                    lr,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),
                )
            
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )
                if isinstance(labels, (dict,)) and cfg.TRAIN.DATASET == "Epickitchens":
                    writer.add_scalars(
                        {
                            "Train/verb_top1_acc": verb_top1_acc,
                            "Train/verb_top5_acc": verb_top5_acc,
                            "Train/noun_top1_acc": noun_top1_acc,
                            "Train/noun_top5_acc": noun_top5_acc,
                            "Train/action_top1_acc": action_top1_acc,
                            "Train/action_top5_acc": action_top5_acc,
                            "Train/LT_verb": verb_lt,
                            "Train/LT_noun": noun_lt,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )
                else:
                    writer.add_scalars(
                        {
                            "Train/Top1_err": top1_err if top1_err is not None else 0.0,
                            "Train/Top5_err": top5_err if top5_err is not None else 0.0,
                            # "Train/LT": lt,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()



    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

    return None

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cur_iter > cfg.TRAIN.MAX_ITERS_PER_EPOCH_VAL:
            break

        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if isinstance(labels, (dict,)):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if not isinstance(val[i], (str,)):
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.SOLVER.USE_MIXED_PRECISION):
            
            preds, _ = model(inputs)
            if isinstance(labels, (dict,)) and cfg.TRAIN.DATASET == "Epickitchens":
                # Compute the verb accuracies.
                verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(
                    preds[0], labels['verb'], (1, 5))

                verb_lt, verb_lt_cor, verb_lt_count, _, _ = metrics.lt_accuracies(preds[0], labels['verb'], cfg.EPICKITCHENS.TRAIN_VERB_COUNTS, cfg.EPICKITCHENS.VERB_DIST_SPLITS)

                # Gther all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    verb_top1_acc, verb_top5_acc, verb_lt, verb_lt_cor, verb_lt_count = du.all_reduce(
                        [verb_top1_acc, verb_top5_acc, verb_lt, verb_lt_cor, verb_lt_count]
                    )

                # Copy the errors from GPU to CPU (sync point).
                verb_top1_acc, verb_top5_acc = verb_top1_acc.item(), verb_top5_acc.item()
                verb_lt = verb_lt.cpu().detach().numpy()
                verb_lt_cor = verb_lt_cor.cpu().detach().numpy()
                verb_lt_count = verb_lt_count.cpu().detach().numpy()                

                # Compute the noun accuracies.
                noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(
                    preds[1], labels['noun'], (1, 5))
                
                noun_lt, noun_lt_cor, noun_lt_count, _, _ = metrics.lt_accuracies(preds[1], labels['noun'], cfg.EPICKITCHENS.TRAIN_NOUN_COUNTS, cfg.EPICKITCHENS.NOUN_DIST_SPLITS)

                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    noun_top1_acc, noun_top5_acc, noun_lt, noun_lt_cor, noun_lt_count = du.all_reduce(
                        [noun_top1_acc, noun_top5_acc, noun_lt, noun_lt_cor, noun_lt_count]
                    )

                # Copy the errors from GPU to CPU (sync point).
                noun_top1_acc, noun_top5_acc = noun_top1_acc.item(), noun_top5_acc.item()
                noun_lt = noun_lt.cpu().detach().numpy()
                noun_lt_cor = noun_lt_cor.cpu().detach().numpy()
                noun_lt_count = noun_lt_count.cpu().detach().numpy() 

                # Compute the action accuracies.
                action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
                    (preds[0], preds[1]),
                    (labels['verb'], labels['noun']),
                    (1, 5))

                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    action_top1_acc, action_top5_acc = du.all_reduce([action_top1_acc, action_top5_acc])

                # Copy the errors from GPU to CPU (sync point).
                action_top1_acc, action_top5_acc = action_top1_acc.item(), action_top5_acc.item()

                val_meter.iter_toc()
                
                # Update and log stats.
                val_meter.update_stats(
                    (verb_top1_acc, noun_top1_acc, action_top1_acc),
                    (verb_top5_acc, noun_top5_acc, action_top5_acc),
                    (verb_lt, noun_lt, verb_lt_cor, noun_lt_cor, verb_lt_count, noun_lt_count),
                    inputs[0].size(0) * cfg.NUM_GPUS
                )
                
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {
                            "Val/verb_top1_acc": verb_top1_acc,
                            "Val/verb_top5_acc": verb_top5_acc,
                            "Val/noun_top1_acc": noun_top1_acc,
                            "Val/noun_top5_acc": noun_top5_acc,
                            "Val/action_top1_acc": action_top1_acc,
                            "Val/action_top5_acc": action_top5_acc,
                        },
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                
                lt, lt_cor, lt_count, _, _ = metrics.lt_accuracies(preds, labels, cfg.SSV2.TRAIN_COUNTS, cfg.SSV2.DIST_SPLITS)

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err, lt, lt_cor, lt_count = du.all_reduce([top1_err, top5_err, lt, lt_cor, lt_count])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()
                lt = lt.cpu().detach().numpy()
                lt_cor = lt_cor.cpu().detach().numpy()
                lt_count = lt_count.cpu().detach().numpy()


                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    (lt, lt_cor, lt_count),
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
        all_labels = [
            label.clone().detach() for label in val_meter.all_labels
        ]
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(
            preds=all_preds, labels=all_labels, global_step=cur_epoch
        )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO and cfg.DATA.INPUT_TYPE == 'rgb':
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Mixed Precision Training Scaler
    if cfg.SOLVER.USE_MIXED_PRECISION:
        loss_scaler = NativeScaler()
    else:
        loss_scaler = None

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, loss_scaler=loss_scaler)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.TRAIN.DATASET == 'Epickitchens':
        train_meter = EPICTrainMeter(len(train_loader), cfg)
        val_meter = EPICValMeter(len(val_loader), cfg)
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    
    mixup_fn = None
    mixup_active = cfg.MIXUP.MIXUP_ALPHA > 0 or cfg.MIXUP.CUTMIX_ALPHA > 0 or cfg.MIXUP.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=cfg.MIXUP.MIXUP_ALPHA, 
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA, 
            cutmix_minmax=cfg.MIXUP.CUTMIX_MINMAX,
            prob=cfg.MIXUP.MIXUP_PROB, 
            switch_prob=cfg.MIXUP.MIXUP_SWITCH_PROB, 
            mode=cfg.MIXUP.MIXUP_MODE,
            label_smoothing=cfg.SOLVER.SMOOTHING, 
            num_classes=cfg.MODEL.NUM_CLASSES
        )

    # Explicitly declare reduction to mean.

    if cfg.MIXUP.MIXUP_ALPHA > 0.:
        # smoothing is handled with mixup label transform
        loss_fun = losses.get_loss_func("soft_target_cross_entropy")()
    elif cfg.SOLVER.SMOOTHING > 0.0:
        loss_fun = losses.get_loss_func("label_smoothing_cross_entropy")(
            smoothing=cfg.SOLVER.SMOOTHING)
    else:
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)
        if "cfg" in signature(loss_fun).parameters:
            loss_fun = loss_fun(cfg)
        elif "reduction " in signature(loss_fun).parameters:
            loss_fun = loss_fun(reduction="mean")
        else:
            loss_fun = loss_fun()

    if cfg.VIT.NUM_EXPERTS > 1:
        loss_fun = losses.ExpertLoss(cfg, loss_fun)

    if cfg.TRAIN.DATASET == "Epickitchens":
        ap = torch.zeros(97)
    elif cfg.TRAIN.DATASET == "Ssv2":
        ap = torch.zeros(cfg.MODEL.NUM_CLASSES)
    else: 
        ap = None

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        if cur_epoch == cfg.TRAIN.CLS_BAL_EPOCH:
            print("Epoch {}: CLS_BAL".format(cur_epoch))
            train_loader.dataset.set_class_balancing()
        if cur_epoch == cfg.TRAIN.FREEZE_REP_EPOCH:
            print("Epoch {}: FREEZE_REP".format(cur_epoch))
            model.module.freeze_representation()
        if cur_epoch == cfg.TRAIN.RESET_CLS_EPOCH:
            print("Epoch {}: RESET_CLS".format(cur_epoch))
            model.module.reset_classifier()

        # Train for one epoch.
        ap = train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer, 
            loss_scaler=loss_scaler, loss_fun=loss_fun, mixup_fn=mixup_fn, ap=ap)

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, 
                loss_scaler=loss_scaler)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()
