#-------------------------------------#
#       Train code
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (Loss, ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
Training your own target detection model must need to pay attention to the following points:
1, double-check before training to see if their format meets the requirements, the library requires the dataset format for the VOC format, you need to prepare the content of the input images and labels
   The input image is a .jpg image, no need to fix the size, it will be automatically resized before passing into the training.
   The grayscale image will be automatically converted to RGB image for training, no need to modify it.
   If the input image has a non-jpg suffix, you need to batch convert it to jpg and then start training.

   The label is in .xml format, and there will be information about the target to be detected in the file, and the label file corresponds to the input image file.

2, the size of the loss value is used to determine whether the convergence, it is more important to have a convergence trend, that is, the verification set loss continues to decline, if the verification set loss basically does not change, the model basically converged.
   The exact size of the loss value doesn't mean much, large and small only lies in the way the loss is calculated, not close to 0 to be good. If you want to make the loss look good, you can go directly to the corresponding loss function and divide by 10000.
   Loss values during training will be saved in the logs folder in the loss_%Y_%m_%d_%H_%M_%S folder
   
3, the trained weights file is saved in the logs folder, each training generation (Epoch) contains a number of training steps (Step), each training step (Step) for a gradient descent.
   If you just train a few Step is not saved, the concept of Epoch and Step to run through it.
'''
if __name__ == "__main__":
    #---------------------------------#
    # Cuda Whether to use Cuda
    # No GPU can be set to False
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    # Seed is used to fix the random seed
    # So that the same result can be obtained for each independent training session
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    # distributed is used to specify whether or not to use single multi-card distributed operation
    # Terminal commands are only supported in Ubuntu. CUDA_VISIBLE_DEVICES is used to specify the graphics card under Ubuntu.
    # DP mode is used to invoke all graphics cards by default under Windows. and DDP is not supported.
    # DP mode:
    # Set distributed = False.
    # Type CUDA_VISIBLE_DEVICES=0,1 in the terminal python train.py
    # DDP mode:
    # Set distributed = True
    # Type CUDA_VISIBLE_DEVICES=0,1 in terminal python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    # sync_bn Whether to use sync_bn, DDP mode multi-card available
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    # fp16 whether to train with mixed precision
    # Reduces video memory by about half, requires pytorch 1.7.1 or higher
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    # classes_path points to the txt under model_data, which is related to your own training dataset.
    # Make sure to modify classes_path before training so that it corresponds to your own dataset.
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/cls_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    # pretrain model path
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = ''
    #------------------------------------------------------#
    #   input_shape
    #------------------------------------------------------#
    input_shape     = [640, 640]
    #------------------------------------------------------#
    # The version of yolov8 used by phi.
    # n : corresponds to yolov8_n
    # s : corresponds to yolov8_s
    # m : corresponds to yolov8_m
    # l : corresponds to yolov8_l
    # x : corresponds to yolov8_x
    #------------------------------------------------------#
    phi             = 's'
    #----------------------------------------------------------------------------------------------------------------------------#
    # pretrained Whether to use the pretrained weights of the backbone network, here the weights of the backbone are used, so they are loaded during model construction.
    # If model_path is set, the weights of the backbone do not need to be loaded and the value of pretrained is meaningless.
    # If model_path is not set, pretrained = True, at which point only the trunk is loaded to start training.
    # If model_path is not set, pretrained = False, Freeze_Train = Fasle, at this point training starts from 0 and there is no process of freezing the trunk.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = True
    #------------------------------------------------------------------#
    #   mosaic data augmentation
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    #------------------------------------------------------------------#
    #   label_smoothing
    #------------------------------------------------------------------#
    label_smoothing     = 0

    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 32
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 16
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    Freeze_Train        = True
    #------------------------------------------------------------------#
    # Other training parameters: learning rate, optimizer, learning rate drop related to
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    # Maximum learning rate for Init_lr model
    # Min_lr model's minimum learning rate, defaults to 0.01 of maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    # optimizer_type the type of optimizer to use, optional adam, sgd
    # Init_lr=1e-3 is recommended when using Adam optimizer.
    # Init_lr=1e-3 is recommended when using the SGD optimizer.
    # momentum The momentum parameter is used internally by the optimizer.
    # weight_decay weight_decay to prevent overfitting
    # adam will cause weight_decay error, recommend setting to 0 when using adam.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 10
    #------------------------------------------------------------------#
    num_workers         = 4

    #------------------------------------------------------#
    # train_annotation_path train image paths and labels
    # val_annotation_path Validate image paths and labels
    #------------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    seed_everything(seed)
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    class_names, num_classes = get_classes(classes_path)

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)  
            dist.barrier()
        else:
            download_weights(phi)

    #------------------------------------------------------#
    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)

    if model_path != '':

        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")


    #----------------------#
    yolo_loss = Loss(model)

    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:

            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #----------------------------#
    ema = ModelEMA(model_train)
    

    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

        #----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small to continue training, please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print(
                "\n\033[1;33;44m[Warning] It is recommended to set the total training step size above %d when using the %s optimizer. \033[0m" % (
                optimizer_type, wanted_step))
            print(
                "\033[1;33;44m[Warning] The total training data volume for this run is %d, Unfreeze_batch_size is %d, a total of %d Epochs are trained, and the total training step size is calculated to be %d. \033[0m" % (
                num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print(
                "\033[1;33;44m[Warning] Since the total training step is %d, which is less than the recommended total step size of %d, it is recommended to set the total generation to %d. \033[0m" % (
                total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False

        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training, please expand the dataset.")

        if ema:
            ema.updates     = epoch_step * Init_Epoch

        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training, please expand the dataset.")
                    
                if ema:
                    ema.updates     = epoch_step * epoch

                if distributed:
                    batch_size  = batch_size // ngpus_per_node
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag   = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
            
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
