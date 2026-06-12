from argparse import Namespace
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def proxy_synthesis(input_l2, proxy_l2, target, ps_rate, ps_mu):
    """
    origin code, which should be used with sampler() & redist()
    :param input_l2: [batch_size, dims] l2-normalized embedding features
    :param proxy_l2: [n_classes, dims] l2-normalized proxy parameters
    :param target: [batch_size]
    :param ps_rate: lambda for linear interpolation
    :param ps_mu: generation ratio (# of synthetics / batch_size)
    """
    input_list = [input_l2]
    proxy_list = [proxy_l2]
    target_list = [target]

    input_aug = ps_rate * input_l2 + (1.0 - ps_rate) * torch.roll(input_l2, 1, dims=0)
    proxy_aug = ps_rate * proxy_l2[target, :] + (1.0 - ps_rate) * torch.roll(proxy_l2[target, :], 1, dims=0)
    input_list.append(input_aug)
    proxy_list.append(proxy_aug)

    n_classes = proxy_l2.shape[0]
    pseudo_target = torch.arange(n_classes, n_classes + input_l2.shape[0], device=input_l2.device)
    target_list.append(pseudo_target)

    embed_size = int(input_l2.shape[0] * (1.0 + ps_mu))
    proxy_size = int(n_classes + input_l2.shape[0] * ps_mu)
    input_large = torch.cat(input_list, dim=0)[:embed_size, :]
    proxy_large = torch.cat(proxy_list, dim=0)[:proxy_size, :]
    target = torch.cat(target_list, dim=0)[:embed_size]

    input_l2 = F.normalize(input_large, p=2, dim=1)
    proxy_l2 = F.normalize(proxy_large, p=2, dim=1)

    return input_l2, proxy_l2, target


def get_centroids(labels, proxies):
    labels = labels.to(proxies.device).float()
    proxies = proxies.float()
    centroids = (labels @ proxies) / labels.sum(1, keepdim=True).clamp_min(1e-12)
    return centroids


def my_proxy_synthesis1(embeddings, proxies, labels, ps_rate, ps_mu):
    """
    add multi-hot labels support
    :param embeddings: [batch_size, dims] l2-normalized embedding features
    :param proxies: [n_classes, dims] l2-normalized proxies
    :param labels: [batch_size, n_classes] multi-hot labels
    :param ps_rate: lambda for linear interpolation
    :param ps_mu: generation ratio (# of synthetics / batch_size)
    """
    input_list = [embeddings]
    proxy_list = [proxies]

    diff_idx = torch.triu((labels @ labels.T) == 0, diagonal=1).nonzero(as_tuple=False)

    n_synthetics = int(embeddings.shape[0] * ps_mu)
    assert n_synthetics <= diff_idx.shape[0], "Not enough disjoint pairs for synthesis."
    diff_idx = diff_idx[:n_synthetics]

    input_aug = ps_rate * embeddings[diff_idx[:, 0]] + (1.0 - ps_rate) * embeddings[diff_idx[:, 1]]
    proxy_aug = ps_rate * get_centroids(labels[diff_idx[:, 0]], proxies) + \
                (1.0 - ps_rate) * get_centroids(labels[diff_idx[:, 1]], proxies)

    input_list.append(input_aug)
    proxy_list.append(proxy_aug)

    label_list = [F.pad(input=labels, pad=(0, n_synthetics, 0, 0), mode="constant", value=0)]
    pseudo_labels = torch.eye(n_synthetics, device=labels.device)
    pseudo_labels = F.pad(input=pseudo_labels, pad=(labels.shape[1], 0, 0, 0), mode="constant", value=0)
    label_list.append(pseudo_labels)

    embed_size = int(embeddings.shape[0] + n_synthetics)
    proxy_size = int(proxies.shape[0] + n_synthetics)
    input_large = torch.cat(input_list, dim=0)[:embed_size, :]
    proxy_large = torch.cat(proxy_list, dim=0)[:proxy_size, :]
    labels = torch.cat(label_list, dim=0)[:embed_size]

    embeddings = F.normalize(input_large, p=2, dim=1)
    proxies = F.normalize(proxy_large, p=2, dim=1)

    return embeddings, proxies, labels


class Norm_SoftMax(nn.Module):
    def __init__(self, args: Namespace, ps_rate=None):
        super(Norm_SoftMax, self).__init__()
        self.scale = args.scale
        self.ps_mu = args.ps_mu
        self.ps_alpha = args.ps_alpha
        self.method = args.method
        self.ps_rate = ps_rate
        self.lambda_quan = getattr(args, "lambda_quan", 0.01)

        self.proxies = nn.Parameter(torch.Tensor(args.n_classes, args.n_bits))
        nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))

    def quantization_loss(self, embeddings):
        return (embeddings.abs() - 1.0).abs().mean()

    def forward(self, embeddings, labels):
        raw_embeddings = embeddings

        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)

        if self.ps_rate is None:
            ps_rate = np.random.beta(self.ps_alpha, self.ps_alpha)
        else:
            ps_rate = self.ps_rate

        # single-label
        if len(labels.shape) == 1:
            embeddings, proxies, targets = proxy_synthesis(embeddings, proxies, labels, ps_rate, self.ps_mu)
            cos_sim = embeddings.matmul(proxies.t())
            logits = self.scale * cos_sim
            cls_loss = F.cross_entropy(logits, targets)

            quan_loss = self.quantization_loss(raw_embeddings)
            total_loss = cls_loss + self.lambda_quan * quan_loss

            return total_loss, {
                "cls_loss": cls_loss.detach(),
                "quan_loss": quan_loss.detach(),
                "total_loss": total_loss.detach(),
            }

        # multi-label
        embeddings, proxies, labels = my_proxy_synthesis1(embeddings, proxies, labels, ps_rate, self.ps_mu)
        cos_sim = embeddings.matmul(proxies.t())
        logits = self.scale * cos_sim

        log_probs = F.log_softmax(logits, dim=1)
        labels = labels.float()
        labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(1e-12)
        cls_loss = -(labels * log_probs).sum(dim=1).mean()

        quan_loss = self.quantization_loss(raw_embeddings)
        total_loss = cls_loss + self.lambda_quan * quan_loss

        return total_loss, {
            "cls_loss": cls_loss.detach(),
            "quan_loss": quan_loss.detach(),
            "total_loss": total_loss.detach(),
        }


if __name__ == "__main__":
    batch_size = 64
    n_bits = 16
    n_classes = 24

    _embeddings = torch.randn(batch_size, n_bits).cuda()
    _labels = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float().cuda()
    _labels = F.pad(_labels, (0, n_classes - 3, 0, 0), "constant", 0)
    _targets = torch.argmax(_labels, dim=1).long()

    args = Namespace(
        n_bits=n_bits,
        n_classes=n_classes,
        scale=23.0,
        ps_alpha=0.40,
        ps_mu=0.8,
        method=1,
        lambda_quan=0.01,
    )

    criterion = Norm_SoftMax(args, ps_rate=0.5).cuda()

    loss, loss_dict = criterion(_embeddings, _targets)
    print("single-label total loss:", loss.item(), loss_dict)

    loss, loss_dict = criterion(_embeddings, _labels)
    print("multi-label total loss:", loss.item(), loss_dict)