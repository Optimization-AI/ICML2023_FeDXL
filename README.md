# FeDXL: Provable Federated Learning for Deep X-Risk Optimization  [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/abs/2210.14396)

This is the official implementation of the paper "**FeDXL: Provable Federated Learning for Deep X-Risk Optimization**" published on **ICML2023**. 


How to run 
---------
If you are using a cluster with SLURM scheduler: 
```
sbatch run.slurm
```
otherwise, use
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=0 --master_addr='YOUR IP' --master_port=8888 \
            main.py --T0=5000 --lr=0.1 --I=32 --total_iter=10000
```

Reference
---------
This is an implementation of the following paper:
```
@article{guo2022fedx,
  title={FedX: Federated Learning for Compositional Pairwise Risk Optimization},
  author={Guo, Zhishuai and Jin, Rong and Luo, Jiebo and Yang, Tianbao},
  journal={arXiv preprint arXiv:2210.14396},
  year={2022}
}
```
