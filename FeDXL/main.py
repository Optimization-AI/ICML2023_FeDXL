import os
import torch
import torch.distributed as dist
import torchvision
from torch.multiprocessing import Process

from torchvision import transforms
import sys
from datetime import datetime
import numpy as np
from arguments import args
import copy
import time
from sklearn.cluster import KMeans

from sklearn import metrics
import pandas as pd
import random
from noise import AddGaussianNoise
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

time_spent_on_training = 0
time_spent_list = list()
iteration_list = list()
train_auc_3_list = list()
val_auc_3_list = list()
test_auc_3_list = list()
train_auc_5_list = list()
val_auc_5_list = list()
test_auc_5_list = list()
train_auc_10_list = list()
val_auc_10_list = list()
test_auc_10_list = list()
comm_r_list = list()

# Model Averaging
def average_model(model, size):
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size
    

class dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, trans=None):
        self.x = inputs
        self.y = targets
        self.trans=trans
        self.size = dist.get_world_size()

    def __len__(self):
        #return self.x.shape[0]
        return self.x.size()[0]

    def __getitem__(self, idx):
        #size = dist.get_world_size()
        if self.trans == None:
            return (self.x[idx], self.y[idx], idx)
        else:
            return (self.trans(self.x[idx]), self.y[idx], idx) 
     
        
def AUC(label, scores, max_fpr):
    label[label==-1] = False
    label[label==1] = True
    return metrics.roc_auc_score(label, scores, max_fpr=max_fpr)

def train(rank, size):
    torch.cuda.set_device(rank%torch.cuda.device_count())
    print("local rank:"+str(args.local_rank))
    use_average = False

    # same random seed on all machines to guarantee non-overlap data partition between machines
    # random seed will be reset after the data partition 
    np.random.seed(1234) 
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234) 
    random.seed(1234)

    if args.ce==1 and args.OPAUC==0:
        args.beta = 1
        CE = torch.nn.BCELoss()
   
    import medmnist 
    from medmnist import INFO, Evaluator
    root = ''
    info = INFO[args.data]
    DataClass = getattr(medmnist, info['python_class'])
    eval_train_dataset = DataClass(split='train', download=True, root=root)
    val_dataset = DataClass(split='val', download=True, root=root)
    test_dataset = DataClass(split='test', download=True, root=root)

    eval_train_data = eval_train_dataset.imgs 
    eval_train_labels = eval_train_dataset.labels[:, args.class_index]
    train_data = eval_train_dataset.imgs 
    train_labels = eval_train_dataset.labels[:, args.class_index] 
   
    train_ids = list(range(len(train_data)))
    np.random.shuffle(train_ids)
    train_data = train_data[train_ids]
    train_labels = train_labels[train_ids]
    
    val_data = val_dataset.imgs 
    val_labels = val_dataset.labels[:, args.class_index]

    test_data = test_dataset.imgs 
    test_labels = test_dataset.labels[:,args.class_index]
    
    eval_train_labels[eval_train_labels != args.pos_class] = -1
    eval_train_labels[eval_train_labels == args.pos_class] = 1
    train_labels[train_labels != args.pos_class] = -1
    train_labels[train_labels == args.pos_class] = 1
    val_labels[val_labels != args.pos_class] = -1
    val_labels[val_labels == args.pos_class] = 1
    test_labels[test_labels != args.pos_class] = -1
    test_labels[test_labels == args.pos_class] = 1


    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((26, 26), padding=None),
            transforms.Grayscale(3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            AddGaussianNoise(0. + (rank-8)*0.01, args.noise_std),
        ])
    eval_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
    
    
    eval_train_data = np.swapaxes(eval_train_data, 1, 3)
    train_data = np.swapaxes(train_data, 1, 3)
    val_data = np.swapaxes(val_data, 1, 3)
    test_data = np.swapaxes(test_data, 1, 3)

    eval_train_data = torch.tensor(eval_train_data)
    eval_train_labels = torch.tensor(eval_train_labels)
    train_data = torch.tensor(train_data)
    train_labels = torch.tensor(train_labels)
    val_data = torch.tensor(val_data)
    val_labels = torch.tensor(val_labels)
    test_data = torch.tensor(test_data)
    test_labels = torch.tensor(test_labels)
    pos_data = train_data[train_labels==1]
    pos_labels = train_labels[train_labels==1]
    neg_data = train_data[train_labels==-1]
    neg_labels = train_labels[train_labels==-1]

    num_pos = len(pos_labels)
    num_neg = len(neg_labels)
    pos_data = pos_data[int(rank/(size*1.0)*num_pos):int((rank+1)/(size*1.0)*num_pos)] 
    pos_labels = [1,]*(int((rank+1)/(size*1.0)*num_pos) - int(rank/(size*1.0)*num_pos)) 
    neg_data = neg_data[int(rank/(size*1.0)*num_neg):int((rank+1)/(size*1.0)*num_neg)] 
    neg_labels = [-1,]*(int((rank+1)/(size*1.0)*num_neg) - int(rank/(size*1.0)*num_neg)) 
     
     
    pos_data = dataset(pos_data, pos_labels, trans = transform)
    neg_data = dataset(neg_data, neg_labels, trans = transform) 
    u = torch.ones(len(pos_labels))
 
    eval_train_data = dataset(eval_train_data, eval_train_labels, trans=eval_transform)
    val_data = dataset(val_data, val_labels, trans=eval_transform)
    test_data = dataset(test_data, test_labels, trans=eval_transform)

    if rank == 0:
        print ('Pos:Neg: [%d : %d]'%(np.count_nonzero(train_labels == 1), np.count_nonzero(train_labels == -1)), flush=True)
    print("rank " + str(rank) + "; numpos:" + str(num_pos) + "; numneg"+str(num_neg))

    
    pos_loader = torch.utils.data.DataLoader(pos_data, batch_size=args.B1, shuffle=True, num_workers=0, drop_last=True)
    neg_loader = torch.utils.data.DataLoader(neg_data, batch_size=args.B2, shuffle=True, num_workers=0, drop_last=True)

    eval_train_loader = torch.utils.data.DataLoader(eval_train_data, batch_size=args.test_batchsize, \
                                                    shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batchsize, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batchsize, shuffle=False, num_workers=0)
     

    import torchvision.models as models
    from densenet import DenseNet121
    net = DenseNet121(pretrained=True, last_activation='sigmoid', activations="relu", num_classes=1)
    torch.manual_seed(args.random_seed + rank)
    torch.cuda.manual_seed(args.random_seed + rank)
    np.random.seed(args.random_seed + rank) 
    random.seed(args.random_seed + rank)
    net = net.cuda()
     
    configs = args.data+'_N_' + str(size) + '_K_' + str(args.I) + "_B1_" + str(args.B1) \
               + "_B2_" + str(args.B2)  + "_lr_" + str(args.lr) + "_tau_" + str(args.tau) \
               +"_OPAUC_" + str(args.OPAUC) + "_CE_" + str(args.ce) + "_seed_" + str(args.random_seed) + "_class_index_"+str(args.class_index)\
               + "_pos_class_"+str(args.pos_class) + "_noisestd_" + str(args.noise_std) + "_lamb_" + str(args.lamb)
    
    t_total = 0
    if 0 == rank:
        print("configs: " + configs)    

    end_all = False
    net.zero_grad()
    comm_count = 0
    start_eval_time = time.time() 
    grad_ma = copy.deepcopy(net.state_dict())
    for name, param in net.named_parameters():
        grad_ma[name] *= 0 
    pos_pool = list()
    neg_pool = list()
    u_pool = list()
    args.pos_size = int(args.I*size*args.tau*args.B1)
    args.neg_size = int(args.I*size*args.tau*args.B2)
    last_comm_count = -1 

    # initialize memory bank
    net.eval()
    pos_score_this = list()
    neg_score_this = list()
    u_this=list()
    try:
        tmp_pos, tmp_pos_label, tmp_pos_idx = pos_iter.next()
    except:
        pos_iter = iter(pos_loader)
        tmp_pos, tmp_pos_label, tmp_pos_idx = pos_iter.next()

    try:
        tmp_neg, tmp_neg_label, tmp_neg_idx = neg_iter.next()
    except:
        neg_iter = iter(neg_loader)
        tmp_neg, tmp_neg_label, tmp_neg_idx = neg_iter.next()
     
    tmp_pos, tmp_pos_label = tmp_pos.cuda(), tmp_pos_label.cuda()
    tmp_neg, tmp_neg_label = tmp_neg.cuda(), tmp_neg_label.cuda()
    tmp_pos = tmp_pos.cuda()
    tmp_neg = tmp_neg.cuda()
    tmp = torch.cat([tmp_pos, tmp_neg], 0)
    score = net(tmp)[:,0] 
    B1 = len(tmp_pos_idx)
    B2 = len(tmp_neg_idx)
    pos_score = score[0:B1]
    neg_score = score[B1:]
    G = 0

    pos_score_this = pos_score_this + list(pos_score.detach().clone().cpu())
    neg_score_this = neg_score_this + list(neg_score.detach().clone().cpu())
            
    with torch.no_grad():
        for i in range(B1):
            L = (1 - (pos_score[i] - neg_score)) 
            L = torch.square((L + torch.abs(L))/2)
            g = 1.0/B2*torch.sum(torch.exp(L/args.lamb))
            
            u[tmp_pos_idx[i]] = (1-args.gamma)*u[tmp_pos_idx[i]] \
                                + args.gamma*g.detach().clone().cpu()
            u_this.append(u[tmp_pos_idx[i]].detach().clone())
                 
    for s in np.arange(1, 20):
        if True == end_all:
            break
        T = args.T0 
        lr = args.lr*0.1**(s-1)  
         
        for t in range(T):
            # evaluation  
            if 0 == t_total % args.test_freq: 
               last_comm_count = comm_count
               eval_start_time = time.time()
               net.eval()

               if 0 == rank:
                   evaluate(net, configs, eval_train_loader, val_loader, test_loader)
                   net.train()
               torch.cuda.empty_cache()
    
            if t_total > args.total_iter:
                end_all = True
                break
 
            start_iteration = datetime.now()

            # communicate over all nodes
            if 0 == t_total%args.I:
                with torch.no_grad():
                    average_model(net, size) 
                    if args.OPAUC==1: 
                        grad_ma_tensor = grad_ma[name].detach().clone()
                        dist.all_reduce(grad_ma_tensor, op=dist.ReduceOp.SUM)
                        grad_ma_tensor /= size
                        grad_ma[name] = grad_ma_tensor.detach().clone()
                        
                        recv_pos = [torch.zeros(len(pos_score_this), device="cuda") for _ in range(size)] 
                        recv_neg = [torch.zeros(len(neg_score_this), device="cuda") for _ in range(size)] 
                        recv_u = [torch.zeros(len(u_this), device="cuda") for _ in range(size)] 
                        dist.all_gather(recv_pos, torch.tensor(pos_score_this, device="cuda"))
                        dist.all_gather(recv_neg, torch.tensor(neg_score_this, device="cuda"))
                        dist.all_gather(recv_u, torch.tensor(u_this, device="cuda"))
                        recv_pos = torch.cat(recv_pos)
                        recv_neg = torch.cat(recv_neg)
                        recv_u = torch.cat(recv_u)
                        pos_pool = pos_pool + list(recv_pos)
                        pos_pool = pos_pool[-args.pos_size:]
 
                        neg_pool = neg_pool + list(recv_neg)
                        neg_pool = neg_pool[-args.neg_size:]
                        u_pool = u_pool + list(recv_u)
                        u_pool = u_pool[-args.pos_size:]
                    pos_score_this = list() 
                    neg_score_this = list() 
                    u_this = list()
                    t_in_round = 0 
                    comm_count += 1
  
            try:
                tmp_pos, tmp_pos_label, tmp_pos_idx = pos_iter.next()
            except:
                pos_iter = iter(pos_loader)
                tmp_pos, tmp_pos_label, tmp_pos_idx = pos_iter.next()

            try:
                tmp_neg, tmp_neg_label, tmp_neg_idx = neg_iter.next()
            except:
                neg_iter = iter(neg_loader)
                tmp_neg, tmp_neg_label, tmp_neg_idx = neg_iter.next()
     
            tmp_pos, tmp_pos_label = tmp_pos.cuda(), tmp_pos_label.cuda()
            tmp_neg, tmp_neg_label = tmp_neg.cuda(), tmp_neg_label.cuda()
            tmp = torch.cat([tmp_pos, tmp_neg], 0)
            score = net(tmp)[:, 0]
            B1 = len(tmp_pos_idx)
            B2 = len(tmp_neg_idx)
            pos_score = score[0:B1]
            neg_score = score[B1:]
            G = 0

            pos_score_this = pos_score_this + list(pos_score.detach().clone())
            neg_score_this = neg_score_this + list(neg_score.detach().clone())
            if args.OPAUC==1 and size>1:
                # G1
                neg_old_idx = torch.randint(0, len(neg_pool), (B2,))
                neg_old = torch.tensor([neg_pool[i] for i in neg_old_idx], device="cuda")
            
                for i in range(B1):
                    L = (args.margin - (pos_score[i] - neg_old.detach().clone())) 
                    L = torch.square((L + torch.abs(L))/2)
                    g = 1.0/B2*torch.sum(torch.exp(L/args.lamb))
            
                    with torch.no_grad():
                        u[tmp_pos_idx[i]] = (1-args.gamma)*u[tmp_pos_idx[i]] \
                                            + args.gamma*g.detach().clone()
                        u_this.append(u[tmp_pos_idx[i]].detach().clone())
                    G = G + g/B1 * 1.0/u[tmp_pos_idx[i]].detach().clone()
                # G2
                pos_old_idx = torch.randint(0, len(pos_pool), (B1, ))
                pos_old = torch.tensor([pos_pool[i] for i in pos_old_idx], device="cuda")
                u_old = torch.tensor([u_pool[i] for i in pos_old_idx], device="cuda")
                for i in range(len(pos_old)):
                    L2 = (args.margin - (pos_old[i].detach().clone() - neg_score))
                    L2 = torch.square((L2+torch.abs(L2))/2)
                    g2 = 1.0/B2*torch.sum(torch.exp(L2/args.lamb))
                    G = G + g2/B1*1.0 / u_old[i].detach().clone()
            else:
                if args.ce==0:
                    for i in range(B1):
                        L = (args.margin - (pos_score[i] - neg_score)) 
                        L = torch.square((L + torch.abs(L))/2)
                        g = 1.0/B2*torch.sum(torch.exp(L/args.lamb))
            
                        with torch.no_grad():
                            u[tmp_pos_idx[i]] = (1-args.gamma)*u[tmp_pos_idx[i]] \
                                                + args.gamma*g.detach().clone()
                            u_this.append(u[tmp_pos_idx[i]].detach().clone())
                        G = G + g/B1 * 1.0/u[tmp_pos_idx[i]].detach().clone()
                elif args.ce==1:
                     batch_label = torch.cat([tmp_pos_label, tmp_neg_label*0.0], 0)
                     G = CE(score, batch_label)
                     
            G.backward(retain_graph=True)
            for name, param in net.named_parameters():
                if t_total <= 0:
                    grad_ma[name] = param.grad.data
                else:
                    grad_ma[name] = (1-args.beta)*grad_ma[name] + args.beta*param.grad.data
                param.data = param.data - lr*grad_ma[name]
             
            net.zero_grad()
            t_in_round += 1
            t_total += 1
            end_iteration = datetime.now()
            time_spent_on_training += (end_iteration - start_iteration).total_seconds()

def evaluate(net, configs, eval_train_loader, val_loader, test_loader):
    net.eval()
    # Training AUC
    score_list = list()
    label_list = list()
    for _, data in enumerate(eval_train_loader, 0):
        tmp_data, tmp_label, tmp_idx = data
        tmp_data, tmp_label = tmp_data.cuda(), tmp_label.cuda()
                
        tmp_score = net(tmp_data)[:,0].detach().clone().cpu()
        score_list.append(tmp_score)
        label_list.append(tmp_label.cpu())
        eval_train_label = torch.cat(label_list)
        eval_train_score = torch.cat(score_list)
                   
        train_auc_3 = AUC(eval_train_label, eval_train_score, 0.3)
        train_auc_3_list.append(train_auc_3)
        train_auc_5 = AUC(eval_train_label, eval_train_score, 0.5)
        train_auc_5_list.append(train_auc_5)
        train_auc_10 = AUC(eval_train_label, eval_train_score, 1.0)
        train_auc_10_list.append(train_auc_10)
    # Validation AUC
    score_list = list()
    label_list = list()
    for _, data in enumerate(val_loader, 0):
        tmp_data, tmp_label, tmp_idx = data
        tmp_data, tmp_label = tmp_data.cuda(), tmp_label.cuda()
                
        tmp_score = net(tmp_data)[:,0].detach().clone().cpu()
        score_list.append(tmp_score)
        label_list.append(tmp_label.cpu())
        val_label = torch.cat(label_list)
        val_score = torch.cat(score_list)
                   
        val_auc_3 = AUC(val_label, val_score, 0.3)
        val_auc_3_list.append(val_auc_3)
        val_auc_5 = AUC(val_label, val_score, 0.5)
        val_auc_5_list.append(val_auc_5)
        val_auc_10 = AUC(val_label, val_score, 1.0)
        val_auc_10_list.append(val_auc_10)
    # Testing AUC
    score_list = list()
    label_list = list()
    for _, data in enumerate(test_loader, 0):
        tmp_data, tmp_label, tmp_idx = data
        tmp_data, tmp_label = tmp_data.cuda(), tmp_label.cuda()
                
        tmp_score = net(tmp_data)[:,0].detach().clone().cpu()
        score_list.append(tmp_score)
        label_list.append(tmp_label.cpu())
        test_label = torch.cat(label_list)
        test_score = torch.cat(score_list)
                   
        test_auc_3 = AUC(test_label, test_score, 0.3)
        test_auc_3_list.append(test_auc_3)
        test_auc_5 = AUC(test_label, test_score, 0.5)
        test_auc_5_list.append(test_auc_5)
        test_auc_10 = AUC(test_label, test_score, 1.0)
        test_auc_10_list.append(test_auc_10)
                    
                    
        print(datetime.now().time(),"Stage: " + str(s) + "; Iter: " + str(t_total)  \
                +'; lr: %.3f '%(lr) \
                + "; Tr 3: %.4f"%(train_auc_3)  + "; Val 3: %.4f"%val_auc_3 + "; Test 3: %.4f"%test_auc_3 
                + "**** Tr 5: %.4f"%(train_auc_5) + "; Val 5: %.4f"%val_auc_5 + "; Test 5: %.4f"%test_auc_5, flush=True)
        print("****************************************************** \
                Tr 10: %.4f"%(train_auc_10) + "; Val 10: %.4f"%val_auc_10 + "; Test 10: %.4f"%test_auc_10\
                +"*************", flush=True)
            
        tmp_time = time_spent_on_training
        time_spent_list.append(tmp_time)
        iteration_list.append(t_total)
        comm_r_list.append(comm_count)
                    
        df = pd.DataFrame(data={'comm_round':comm_r_list, 'total_iteration':iteration_list, \
                            'time':time_spent_list, 'train_auc_3':train_auc_3_list, \
                            'val_auc_3':val_auc_3_list, 'test_auc_3':test_auc_3_list, \
                            'train_auc_5':train_auc_5_list, \
                            'val_auc_5':val_auc_5_list, 'test_auc_5':test_auc_5_list, \
                            'train_auc_10':train_auc_10_list, \
                            'val_auc_10':val_auc_10_list, 'test_auc_10':test_auc_10_list}) 
        df.to_csv('history/history_'+configs+'.csv') 
            
def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    rank = int(os.environ['SLURM_PROCID'])
    size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='gloo', rank=rank, world_size=size)
    print("Rank " + str(rank) + " of World Size " + str(size), flush=True)
    print("dist.get_world_size:" + str(dist.get_world_size()), flush=True)
    
    train(rank, size) 
