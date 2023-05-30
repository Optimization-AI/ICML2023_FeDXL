import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--T0', type=int, default=5000)
parser.add_argument('--numStages', type=int, default=10000)
parser.add_argument('--B1', type=int, default=1)
parser.add_argument('--B2', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.1) # initial learning rate
parser.add_argument('--gamma', type=float, default=0.2)
parser.add_argument('--test_freq', type=int, default=200)
parser.add_argument('--test_batchsize', type=int, default=128)
parser.add_argument('--test_batches', type=int, default=100) # total used in testing" test_batchsize * test_batches
parser.add_argument('--save_freq', type=int, default=10000)
parser.add_argument('--I', type=int, default=2)

parser.add_argument('--numGPU', type=int, default=1)
parser.add_argument('--total_iter', type=int, default=10000)
parser.add_argument('--neg_keep_ratio', type=float, default=0.2)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--master_addr', type=str)
parser.add_argument('--max_fpr', type=float, default=0.3)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--lamb', type=float, default=1)
parser.add_argument('--pos_size', type=int, default=10000)
parser.add_argument('--neg_size', type=int, default=50000)
parser.add_argument('--data_split', type=int, default=4)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--data', type=str, default="Cifar10")
parser.add_argument('--tau', type=float, default=1)
parser.add_argument('--OPAUC', type=int, default=1)
parser.add_argument('--ce', type=int, default=0)
parser.add_argument('--central', type=int, default=0)
parser.add_argument('--margin', type=float, default=1)
parser.add_argument('--num_cluster', type=int, default=2)
parser.add_argument('--random_seed', type=int, default=1234)
parser.add_argument('--server', type=str, default="faster")
parser.add_argument('--pos_class', type=int, default=1)
parser.add_argument('--class_index', type=int, default=4)
parser.add_argument('--noise_std', type=float, default=0.01)


args = parser.parse_args()
