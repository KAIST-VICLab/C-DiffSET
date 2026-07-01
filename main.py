"""Entry point for C-DiffSET training.

Configuration is provided through a YAML file (``--config``); any CLI flag not
present in the YAML falls back to the defaults defined below. Example::

    python main.py --config configs/spacenet_eps_conf.yaml
"""

import argparse
import os
import random
import sys
import traceback

import numpy as np
import torch
import yaml

from train import Trainer
from utils import Report


class YamlAction(argparse.Action):
    """argparse Action that parses an inline YAML string into a dict."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, yaml.safe_load(values))


def get_parser():
    parser = argparse.ArgumentParser(
        description='C-DiffSET: SAR-to-EO Image Translation with Stable Diffusion Models')
    parser.add_argument('--work-dir', default='./work_dir',
                        help='folder for logs, checkpoints, and result images')
    parser.add_argument('--config', default='./config/test.yaml',
                        help='path to the YAML configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')

    # logging / evaluation
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log-iter', type=int, default=100,
                        help='interval (in iterations) for printing training logs')
    parser.add_argument('--save-iter', type=int, default=1,
                        help='interval (in iterations) for saving checkpoints')
    parser.add_argument('--save-epoch', type=int, default=0,
                        help='interval (in epochs) for saving checkpoints')
    parser.add_argument('--eval-epoch', type=int, default=5,
                        help='interval (in epochs) for running validation')

    # feeder
    parser.add_argument('--feeder', default='feeders.feeder', help='data loader class path')
    parser.add_argument('--num-worker', type=int, default=4, help='number of data-loader workers')
    parser.add_argument('--train-feeder-args', action=YamlAction, default=dict(),
                        help='keyword arguments for the training feeder')
    parser.add_argument('--test-feeder-args', action=YamlAction, default=dict(),
                        help='keyword arguments for the test feeder')

    # model
    parser.add_argument('--model', default=None, help='(unused placeholder)')
    parser.add_argument('--prediction-type', type=str, default='epsilon')
    parser.add_argument('--loss-diff-weight', type=float, default=1.0)
    parser.add_argument('--loss-cond-weight', type=float, default=1.0)
    parser.add_argument('--loss-reg-weight', type=float, default=0.2)

    # optimization
    parser.add_argument('--gpu', type=int, default=0, help='GPU index (single-GPU training)')
    parser.add_argument('--optimizer', default='AdamW', help='optimizer type')
    parser.add_argument('--lr-scheduler', default='cosine', help='LR scheduler type')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--num-iter', type=int, default=1, help='total number of iterations')
    parser.add_argument('--num-warmup', type=int, default=1, help='number of warmup iterations')
    parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1, help='test batch size')
    parser.add_argument('--mixed-precision', type=str, default=None, choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--pretrained-model-name-or-path', type=str,
                        default='Manojb/stable-diffusion-2-1-base')
    parser.add_argument('--accelerator-path', type=str,
                        default='Manojb/stable-diffusion-2-1-base',
                        help='path to the pretrained (SAR-conditioned) U-Net safetensors')

    # test
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--val-len', type=int, default=10)
    return parser


def import_class(import_str):
    """Import ``module.ClassName`` given as a dotted string."""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)'
                          % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Unsupported value encountered.')


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(args):
    Feeder = import_class(args.feeder)
    data_loader = dict()
    data_loader['train'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.train_feeder_args),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_worker, drop_last=True)
    data_loader['test'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.test_feeder_args),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_worker, drop_last=False)
    return data_loader


def train(args):
    global_step = 0
    train_log = Report(args.work_dir, type='train')
    val_log = Report(args.work_dir, type='val')
    data_loader = load_data(args)

    trainer = Trainer(args=args, data_loader=data_loader)

    best_psnr = 0
    best_epoch = 0
    total_epoch = args.num_iter // len(data_loader['train']) + 1
    for epoch in range(0, total_epoch):
        train_log.write(f'========= Epoch {epoch + 1} of {total_epoch} =========')
        global_step = trainer.train(train_log, global_step)

        if args.save_epoch and (epoch + 1) % args.save_epoch == 0:
            trainer.save_checkpoint(epoch + 1)

        if (global_step > args.num_iter * 0.8) or (epoch + 1) % args.eval_epoch == 0:
            psnr = trainer.val(val_log, epoch + 1, args.val_len)
            if psnr > best_psnr:
                best_psnr = psnr
                best_epoch = epoch + 1
                trainer.save_best_model()
            val_log.write(f'Best PSNR: {best_psnr:.6f}\tBest Epoch: {best_epoch}')


if __name__ == '__main__':
    parser = get_parser()

    # Merge YAML config into argparse defaults.
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_args = yaml.safe_load(f)
        keys = vars(p).keys()
        for k in default_args.keys():
            if k not in keys:
                print('WRONG ARG: {}'.format(k))
                assert k in keys
        parser.set_defaults(**default_args)

    args = parser.parse_args()
    init_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.phase == 'train':
        train(args)
