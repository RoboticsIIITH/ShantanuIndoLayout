from __future__ import with_statement
import argparse
import os
import numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from datasets.habitat import HabitatDataset
import utils.logger as logger
import torch.backends.cudnn as cudnn
import cv2

from networks.stereo_depth.anynet.models.anynet import AnyNet

parser = argparse.ArgumentParser(description='Anynet train on Habitat')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/habitat',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--train_split_file', type=str, default=None)
parser.add_argument('--val_split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--save_eval_output', action='store_true')


args = parser.parse_args()

if args.save_eval_output and args.test_bsize != 1:
    raise Exception("Set test batch size to 1 when dumping evaluation output")

def main():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = AnyNet(args)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            mk, uk = model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')
    cudnn.benchmark = True
    start_full_time = time.time()

    with open(args.val_split_file, 'r') as f:
        val_filepaths = f.read().splitlines()

    TestImgLoader = torch.utils.data.DataLoader(
        HabitatDataset(args.datapath, val_filepaths),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if args.evaluate:
        test(TestImgLoader, model, log)
        return
    
    with open(args.train_split_file, 'r') as f:
        train_filepaths = f.read().splitlines()

    TrainImgLoader = torch.utils.data.DataLoader(
        HabitatDataset(args.datapath, train_filepaths),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        adjust_learning_rate(optimizer, epoch)

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = args.save_path + f'/checkpoint_{epoch}.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

        if epoch % 1 ==0:
            test(TestImgLoader, model, log)

    test(TestImgLoader, model, log)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, inputs in enumerate(dataloader):
        imgL = inputs[("color_aug", 'l', 0)].float().cuda()
        imgL = imgL.reshape((-1, imgL.shape[-3], imgL.shape[-2], imgL.shape[-1]))
        imgR = inputs[("color_aug", 'r', 0)].float().cuda()
        imgR = imgR.reshape((-1, imgR.shape[-3], imgR.shape[-2], imgR.shape[-1]))

        depth_L = inputs["depth_gt"].float().cuda()
        depth_L = depth_L.reshape((-1, depth_L.shape[-2], depth_L.shape[-1]))

        ## Todo: Change this depth to disp conversion.
        # Using -1 for inf,-inf to filter it using mask.
        disp_L = torch.nan_to_num((320 * 0.2)/depth_L, -1, -1) 

        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        outputs, _, _ = model(imgL, imgR)

        if args.with_spn:
            if epoch >= args.start_epoch_for_spn:
                num_out = len(outputs)
            else:
                num_out = len(outputs) - 1
        else:
            num_out = len(outputs)

        outputs = [torch.squeeze(output, 1) for output in outputs]
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        if batch_idx % args.print_freq:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)


def test(dataloader, model, log):

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, inputs in enumerate(dataloader):
        imgL = inputs[("color_aug", 'l', 0)].float().cuda()
        imgL = imgL.reshape((-1, imgL.shape[-3], imgL.shape[-2], imgL.shape[-1]))
        imgR = inputs[("color_aug", 'r', 0)].float().cuda()
        imgR = imgR.reshape((-1, imgR.shape[-3], imgR.shape[-2], imgR.shape[-1]))

        depth_L = inputs["depth_gt"].float().cuda()
        depth_L = depth_L.reshape((-1, depth_L.shape[-2], depth_L.shape[-1]))

        ## Todo: Change this depth to disp conversion.
        # Using -1 for inf,-inf to filter it using mask in error_estimating
        disp_L = torch.nan_to_num((320 * 0.2)/depth_L, -1, -1)

        with torch.no_grad():
            outputs, _, _ = model(imgL, imgR)

            if args.save_eval_output:
                line = inputs['filename'][0].split()
                outdir = os.path.join(args.save_path, 'eval_output', os.path.basename(line[0]))
                os.makedirs(outdir, exist_ok=True)
                outpath = os.path.join(outdir, '{}.png'.format(line[1]))
                img = torch.nan_to_num((320 * 0.2)/outputs[-1], 0, 0).squeeze().cpu().detach().numpy()
                img = (img * 65535/10).astype(numpy.uint16)
                cv2.imwrite(outpath, img)

            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())

        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])

        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error = ' + info_str)


def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
