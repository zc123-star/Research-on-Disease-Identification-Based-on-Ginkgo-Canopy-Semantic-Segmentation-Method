import time
import os
import datetime

import torch

from src import multi_unet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from my_dataset import DriveDataset
import transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # File used to save coco_info
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_root = args.data_path
    # check data root
    if os.path.exists(os.path.join(data_root, "DRIVE")) is False:
        raise FileNotFoundError("DRIVE dose not in path:'{}'.".format(data_root))

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn, drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    print("Creating model")
    # create model num_classes equal background + foreground classes
    model = create_model(num_classes=num_classes)
    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Create a learning rate update strategy, here it is updated every step (not every epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True)

    # If the resume parameter is passed, which is the address of the last training weights, training will continue from the previous parameters.
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # Load the previously saved weight file (including the optimizer and learning rate schedule)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        return

    best_dice = 0.
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")

        # Write operations are performed only in the main process.
        if args.rank in [-1, 0]:
            # write into txt
            with open(results_file, "a") as f:
                # Record the train_loss, learning rate (lr), and various metrics of the validation set for each epoch.
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n" \
                             f"dice coefficient: {dice:.3f}\n"
                f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        if args.output_dir:
            # Perform the weight-saving operation only on the master node
            save_file = {'model': model_without_ddp.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'lr_scheduler': lr_scheduler.state_dict(),
                         'args': args,
                         'epoch': epoch}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()

            if args.save_best is True:
                save_on_master(save_file,
                               os.path.join(args.output_dir, 'best_model.pth'))
            else:
                save_on_master(save_file,
                               os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='./', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best weights')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
