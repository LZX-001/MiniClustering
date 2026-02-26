# -*- coding: UTF-8 -*-
import argparse
import os

import torch
import torch.distributed as dist
import numpy as np
import random
from torch import nn

from models.Accuracy import cluster_accuracy, clustering
from models.byol import BYOL


from models.util import get_dataset, logger, get_embedding_for_test
import torch.nn.functional as F




parser = argparse.ArgumentParser('Default arguments for training of different methods')
parser.add_argument('--save_freq', type=int, default=100, help='save frequency')




parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')


parser.add_argument('--resume_epoch', type=int, default=0, help='resume epoch')
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--resume', default=False, type=bool, help='if resume training')
parser.add_argument('--dataset', type=str, default='stl10', help='dataset')
parser.add_argument('--imb_ratio', type=int, default=10, help='imbalance ratio')
parser.add_argument('--train_from_start', default=True, type=bool, help='if resume training')
parser.add_argument('--train_from_resume', default=False, type=bool, help='if resume training')
parser.add_argument('--data_folder', type=str, default='', help='path to custom dataset')
parser.add_argument('--other_cluster_num', type=int, default=40, help='mini-cluster number M')
parser.add_argument('--minimum_ratio', type=int, default=0.3, help='mini-cluster assignment threshold delta')
parser.add_argument('--alpha', type=int, default=1, help='trade-off parameter')
parser.add_argument('--beta', type=int, default=0.2, help='trade-off parameter')
parser.add_argument('--threshold', type=int, default=0.99, help='self-training threshold')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='base learning rate')
parser.add_argument('--maxiclustering', type=bool, default=False, help='maxiclustering mode')


# 分布式训练相关
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_devices', type=int, default=1, help='number of devices to use')

# 优化器相关
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--momentum_base', type=float, default=0.996, help='momentum')
parser.add_argument('--momentum_max', type=float, default=1, help='momentum')
parser.add_argument('--momentum_increase', help='momentum_increase', action='store_true')
parser.add_argument('--amp', action='store_true', help='amp')
# parser.add_argument('--encoder_name', type=str, default='bigresnet18', help='the type of encoder')
parser.add_argument('--exclude_bias_and_bn', help='exclude_bias_and_bn', action='store_true')


parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

parser.add_argument('--acc_grd_step', type=int, default=1)
parser.add_argument('--warmup_epochs', type=int, default=10, help='lr warmup epochs')
parser.add_argument('--dist', action='store_false', help='use for clustering')
parser.add_argument('--hidden_size', help='hidden_size', type=int, default=4096)
parser.add_argument('--syncbn', help='syncbn', action='store_false')
parser.add_argument('--shuffling_bn', help='shuffling_bn', action='store_false')
parser.add_argument('--temperature', help='temperature', type=float, default=0.5)
parser.add_argument('--test_resized_crop', action='store_true', help='imagenet test transform')
parser.add_argument('--resized_crop_scale', type=float, default=0.08, help='randomresizedcrop scale')
parser.add_argument('--use_gaussian_blur', action='store_true', help='use_gaussian_blur')




def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)




class Cluster_projector(nn.Module):
    def __init__(self, fea_dim, cluster_num,image_centers, other_cluster_num,other_image_centers):
        super().__init__()
        self.layers1 = nn.Linear(fea_dim, cluster_num)
        self.layers2 = nn.Linear(fea_dim,other_cluster_num)
        self.layers1.weight.data = 100 * image_centers.detach().clone().cuda()
        self.layers2.weight.data = 100 * other_image_centers.detach().clone().cuda()

        nn.init.zeros_(self.layers1.bias)
        nn.init.zeros_(self.layers2.bias)
        self.cluster_num = cluster_num
        self.other_cluster_num = other_cluster_num

    def forward(self, x):
        return self.layers1(x), self.layers2(x)
def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=1e-8)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))

def get_class_weight(predictions,other_predictions,cluster_num,other_cluster_num, minimum_ratio, maxiclustering = False):
    flat = predictions * other_cluster_num + other_predictions
    cnt_matrix = torch.bincount(
        flat,
        minlength=cluster_num * other_cluster_num
    ).reshape(cluster_num, other_cluster_num).cuda()
    cnt_matrix = cnt_matrix.float()
    if maxiclustering==False:
        weight = F.normalize(cnt_matrix, p=1, dim=0)
        weight = weight * (weight > minimum_ratio).long()

        result = torch.count_nonzero(weight, dim=1)
        result = torch.clamp(result, min=0.5)

        return 1 / (result / other_cluster_num), weight, cnt_matrix
    else:
        weight = F.normalize(cnt_matrix, p=1, dim=1)
        weight = weight * (weight > minimum_ratio).long()

        result = torch.count_nonzero(weight, dim=1)
        result = torch.clamp(result, min=0.5)

        return result, weight, cnt_matrix   # 1 / (other_cluster_num/result)
def train_cluster_projector_instance_cluster(opt, model, optimizer, train_loader, epoch, log,cluster_projector):
    n_iter = opt.num_batch * (epoch - 1) + 1

    for i, (inputs, target, idx) in enumerate(train_loader):
        inputs_1, inputs_2, inputs_3 = inputs
        inputs_1 = inputs_1.cuda(non_blocking=True)
        inputs_2 = inputs_2.cuda(non_blocking=True)
        inputs_3 = inputs_3.cuda(non_blocking=True)

        model.train()
        cluster_projector.train()
        # lr = adjust_learning_rate_self_labeling(opt, cluster_projector, optimizer, n_iter)
        with torch.no_grad():
            feature2 = model.encoder(inputs_2)
            logits2, other_logits2 = cluster_projector(feature2)

        feature1 = model.encoder(inputs_1)
        logits1,other_logits1 = cluster_projector(feature1)

        c = F.softmax(logits1 / 1, dim=1)
        confidence = c.max(dim=1).values
        prediction = c.argmax(dim=1)

        other_c = F.softmax(other_logits1 / 1, dim=1)
        other_confidence = other_c.max(dim=1).values
        other_prediction = other_c.argmax(dim=1)

        threshold = opt.threshold
        idx = torch.where(confidence > threshold)[0]
        other_idx = torch.where(other_confidence > threshold)[0]
        if len(idx) <=0 or len(other_idx) <= 0:
            print(len(idx),len(other_idx))
            print(max(confidence),max(other_confidence))

        weight, weight_matrix, cnt_matrix = get_class_weight(prediction, other_prediction, opt.cluster_num, opt.other_cluster_num, opt.minimum_ratio,opt.maxiclustering)

        criterion_weight = ConfidenceBasedCE(threshold=threshold, apply_class_balancing=True, class_weight=weight)
        criterion_no_weight = ConfidenceBasedCE(threshold=threshold,apply_class_balancing=True, class_weight=None)

        ce_loss1 = criterion_weight(logits2, logits1)

        ce_loss2 = criterion_no_weight(other_logits2, other_logits1)
	if ce_loss1 is None or ce_loss2 is None:
            print("mask-all-zero happens: {}".format([epoch,i]))
            continue
        ce_loss = ce_loss1+ce_loss2*opt.alpha #

        sim = get_sim(c)
        other_sim = get_sim((other_c))
        c_loss = F.mse_loss(sim, other_sim)

        loss = ce_loss+c_loss*opt.beta

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i == 0:
            print(weight)
            print(ce_loss,c_loss)
            log.info('Epoch {}, loss: {}'.format(epoch, loss))
        n_iter += 1
def get_sim(A):
    A_norm = F.normalize(A, dim=1)
    sim_A = torch.mm(A_norm, A_norm.t())
    return sim_A
class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            return None
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing, class_weight):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.threshold = threshold
        self.apply_class_balancing = apply_class_balancing
        self.class_weight = class_weight

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak)
        max_prob, target = torch.max(weak_anchors_prob, dim=1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        if self.class_weight is not None:
        # Class balancing weights
            weight = self.class_weight
        else:
            if self.apply_class_balancing:
                idx, counts = torch.unique(target_masked, return_counts=True)
                freq = 1 / (counts.float() / n)
                weight = torch.ones(c).cuda()
                weight[idx] = freq
            else:
                weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight=weight, reduction='mean')

        return loss
def evaluate(model,cluster_predictor,opt,test_loader):
    model.eval()
    cluster_predictor.eval()
    local_prediction = []
    other_local_prediction = []
    local_labels = []
    local_embedding = []
    cluster_num = cluster_predictor.cluster_num

    for i, (inputs, target, idx) in enumerate(test_loader):
        inputs = inputs.to('cuda')
        target = target.to('cuda')
        with torch.no_grad():
            feature = model.encoder(inputs)

            c,other_c = cluster_predictor(feature)
            c = F.softmax(c / 1, dim=1)
            other_c = F.softmax(other_c / 1, dim=1)
            confidence = c.max(dim=1).values
            other_confidence = other_c.max(dim=1).values
            prediction = c.argmax(dim=1)
            other_prediction = other_c.argmax(dim=1)
            # indice = torch.nonzero(c1<c2).squeeze()
            local_prediction.append(prediction)
            other_local_prediction.append(other_prediction)
            local_labels.append(target)
            local_embedding.append(feature)
    predictions = torch.cat(local_prediction, dim=0)
    other_predictions = torch.cat(other_local_prediction, dim=0)
    labels = torch.cat(local_labels, dim=0)
    embedding = torch.cat(local_embedding,dim=0)
    embedding = embedding.cpu().numpy()
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    acc, nmi, ari, ca = cluster_accuracy(labels, predictions, verbose=False)
    print(acc, nmi, ari, np.mean(ca))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    global opt
    opt = parser.parse_args()
    if opt.dist:
        dist.init_process_group(backend='nccl', init_method='env://' )#f'tcp://localhost:10001?rank={rank}&world_size={world_size}')
        torch.cuda.set_device(dist.get_rank())
    if opt.num_devices > 0:
        assert opt.num_devices == torch.cuda.device_count()  # total batch size
    if os.path.exists(opt.save_dir) is not True:
        os.system("mkdir -p {}".format(opt.save_dir))
    seed = opt.seed

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    rank = torch.distributed.get_rank()
    logName = "log.txt"
    log = logger(path=opt.save_dir, local_rank=rank, log_name=logName)
    log.info(str(opt))
    # if opt.wandb:
    #     wandb.init(project=opt.project_name,entity=opt.entity,name=opt.run_name,config=str(opt))
    # opt.learning_rate = opt.learning_rate * (opt.batch_size / 256)
    if opt.dataset == 'cifar10':
        opt.img_size = 32
        opt.num_cluster = 10
        opt.encoder_name = "bigresnet18"
        opt.fea_dim = 512
        opt.max_num = 5000
    elif opt.dataset == 'cifar20':
        opt.img_size = 32
        opt.num_cluster = 20
        opt.encoder_name = 'bigresnet18'
        opt.fea_dim = 512
        opt.max_num = 2500

    elif opt.dataset == 'stl10':
        opt.img_size = 96
        opt.num_cluster = 10
        opt.encoder_name = 'resnet18'
        opt.fea_dim = 256
        opt.max_num = 1300

    elif opt.dataset == 'tiny-imagenet':
        opt.img_size = 64  # 96
        opt.num_cluster = 200
        opt.encoder_name = 'resnet18'
        opt.test_resized_crop = True
        opt.fea_dim = 512
        opt.max_num = 500

    elif opt.dataset == 'imagenet-lt':
        opt.img_size = 224
        opt.num_cluster = 1000
        opt.encoder_name = 'resnet50'
        opt.test_resized_crop = True
        opt.fea_dim = 512
    else:
        log.info("unknown dataset")
    opt.cluster_num = opt.num_cluster
    opt.cluster_num = opt.num_cluster
    model = BYOL(opt)
    model.cuda()
    train_datasets = get_dataset(opt, 'mem')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        pin_memory=False)
    opt.num_batch = len(train_loader)
    test_datasets = get_dataset(opt, 'test')
    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
	sampler=train_sampler,
        shuffle=False,
        pin_memory=False)
    start_epoch = 1
    assert opt.train_from_start!=opt.train_from_resume
    if opt.train_from_start:
        if opt.checkpoint == '':
            checkpoint = torch.load(os.path.join(opt.save_dir, 'model.pt'), map_location="cuda")
        else:
            checkpoint = torch.load(os.path.join(opt.save_dir, opt.checkpoint), map_location="cuda")
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        embedding, target = get_embedding_for_test(model, test_loader)

        y_pred, image_centers = clustering(embedding, opt.num_cluster)
        acc, nmi, ari, ca = cluster_accuracy(target.cpu().numpy(), y_pred.cpu().numpy())
        print([acc, nmi, ari, np.mean(ca), ca])
        image_centers = torch.nn.functional.normalize(image_centers, dim=-1)

        other_y_pred, other_image_centers = clustering(embedding, opt.other_cluster_num)
        other_image_centers = torch.nn.functional.normalize(other_image_centers, dim=-1)

        cluster_projector = Cluster_projector(opt.fea_dim, opt.cluster_num, image_centers, opt.other_cluster_num,
                                              other_image_centers)
        cluster_projector.cuda()
        params = list(model.encoder_k.parameters()) + list(cluster_projector.parameters()) + list(
            model.projector_k.parameters())

        for i in list(model.encoder_k.parameters()):
            i.requires_grad = True
        for i in list(model.projector_k.parameters()):
            i.requires_grad = True
        optimizer = torch.optim.Adam(params=params, lr=opt.learning_rate)
    elif opt.train_from_resume:
        checkpoint = torch.load(os.path.join(opt.save_dir, opt.checkpoint), map_location="cuda")
        model.load_state_dict(checkpoint['model_state_dict'])
        cluster_projector = Cluster_projector(opt.fea_dim, opt.cluster_num, torch.randn(opt.cluster_num, opt.fea_dim), opt.other_cluster_num,
                                              torch.randn(opt.other_cluster_num, opt.fea_dim))
        cluster_projector.cuda()
        cluster_projector.load_state_dict(checkpoint['projector_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        params = list(model.encoder_k.parameters()) + list(cluster_projector.parameters()) + list(
            model.projector_k.parameters())

        for i in list(model.encoder_k.parameters()):
            i.requires_grad = True
        for i in list(model.projector_k.parameters()):
            i.requires_grad = True
        optimizer = torch.optim.Adam(params=params, lr=opt.learning_rate)
        optimizer.load_state_dict(checkpoint['optim'])

    else:
        raise ValueError('no checkpoint of the encoder or clustering heads')
    for epoch in range(start_epoch, opt.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_cluster_projector_instance_cluster(opt,model,optimizer,train_loader, epoch, log,cluster_projector)

        if epoch % 5 == 0 or epoch == 1:
            evaluate(model, cluster_projector, opt, test_loader)
        if rank == 0:
            if epoch % opt.save_freq == 0 :
                save_checkpoint({
                    'epoch': epoch,
                    'projector_state_dict': cluster_projector.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, filename=os.path.join(opt.save_dir, 'MiniClustering_{}.pt'.format(epoch)))





