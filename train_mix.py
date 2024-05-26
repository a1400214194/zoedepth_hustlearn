# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

from zoedepth.utils.misc import count_parameters, parallelize
from zoedepth.utils.config import get_config
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.trainers.builder import get_trainer
from zoedepth.models.builder import build_model
from zoedepth.data.data_mono import MixedNYUKITTI
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch
import numpy as np
from pprint import pprint
import argparse
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"

#目的是确保在相同的随机种子下，每次运行模型训练代码时得到相同的结果
def fix_random_seed(seed: int):
    """
    Fix random seed for reproducibility
    Args:
        seed (int): random seed
    """
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#根据配置文件中的设置，加载预训练的模型权重
def load_ckpt(config, model, checkpoint_dir="./checkpoints", ckpt_type="best"):
    import glob
    import os

    from zoedepth.models.model_io import load_wts

    if hasattr(config, "checkpoint"):
        checkpoint = config.checkpoint
    elif hasattr(config, "ckpt_pattern"):
        pattern = config.ckpt_pattern
        matches = glob.glob(os.path.join(
            checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
        if not (len(matches) > 0):
            raise ValueError(f"No matches found for the pattern {pattern}")

        checkpoint = matches[0]

    else:
        return model
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model

#实现了模型训练的整个流程，从模型的构建、加载预训练权重、数据加载器的创建、训练器的选择，到模型训练的过程
def main_worker(gpu, ngpus_per_node, config):
    try:
        fix_random_seed(43)#调用fix_random_seed函数，设置随机种子为43，以确保训练过程的可复现性。

        config.gpu = gpu#将当前GPU的索引赋值给配置文件config的gpu属性，以便指定在哪个GPU上运行模型。

        model = build_model(config)#构建模型，根据配置文件config中的设定，创建相应的深度学习模型。
        model = load_ckpt(config, model)#加载预训练的权重到模型中，根据配置文件
        model = parallelize(config, model)#并行化处理模型，如果需要在多个GPU上运行模型，则将模型进行数据并行化。

        total_params = f"{round(count_parameters(model)/1e6,2)}M"#计算模型的总参数量，并保存为字符串格式。
        config.total_params = total_params#将计算得到的模型总参数量保存到配置文件config中。
        print(f"Total parameters : {total_params}")

        train_loader = MixedNYUKITTI(config, "train").data#创建训练集和测试集的数据加载器，用于加载训练和测试数据。
        test_loader = MixedNYUKITTI(config, "online_eval").data

        trainer = get_trainer(config)(
            config, model, train_loader, test_loader, device=config.gpu)

        trainer.train()#开始训练模型。
    finally:
        import wandb
        wandb.finish()# 结束运行并清理wandb的资源


if __name__ == '__main__':
    mp.set_start_method('forkserver')#设置多进程启动方式为'forkserver'

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="synunet")
    parser.add_argument("-d", "--dataset", type=str, default='mix')
    parser.add_argument("--trainer", type=str, default=None)

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    overwrite_kwargs["model"] = args.model
    if args.trainer is not None:
        overwrite_kwargs["trainer"] = args.trainer

    config = get_config(args.model, "train", args.dataset, **overwrite_kwargs)
    # git_commit()
    if config.use_shared_dict:
        shared_dict = mp.Manager().dict()
    else:
        shared_dict = None
    config.shared_dict = shared_dict

    config.batch_size = config.bs
    config.mode = 'train'
    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace(
            '[', '').replace(']', '')
        nodes = node_str.split(',')

        config.world_size = len(nodes)
        config.rank = int(os.environ['SLURM_PROCID'])
        # config.save_dir = "/ibex/scratch/bhatsf/videodepth/checkpoints"

    except KeyError as e:
        # We are NOT using SLURM
        config.world_size = 1
        config.rank = 0
        nodes = ["127.0.0.1"]

    if config.distributed:#如果配置中开启了分布式训练

        print(config.rank)
        port = np.random.randint(15000, 15025)
        config.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(config.dist_url)
        config.dist_backend = 'nccl'
        config.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node
    print("Config:")
    pprint(config)
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = 0
        main_worker(config.gpu, ngpus_per_node, config)
