import json
import sys
import os
import time
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from loguru import logger
from timm.utils import AverageMeter
from argparse import Namespace
from _data import build_loaders, get_topk, get_class_num
from _utils import (
    build_optimizer,
    calc_learnable_params,
    calc_map_eval,
    EarlyStopping,
    init,
    print_in_md,
    save_checkpoint,
    seed_everything,
    validate_smart,
    rename_output,
)
from config import get_config
from _network import build_model
from synthetic_proxy_loss import Norm_SoftMax  # 只保留 Norm_SoftMax


def train_epoch(args, dataloader, net, criterion, optimizer, epoch):
    tic = time.time()
    stat_meters = {}
    for x in ["cls_loss", "quan_loss", "loss", "mAP"]:
        stat_meters[x] = AverageMeter()
    net.train()
    for images, labels, index in dataloader:
        images = images.to(args.device)
        labels = labels.float().to(args.device)
        embeddings = net(images)
        loss, loss_dict = criterion(embeddings, labels)
        stat_meters["loss"].update(loss.item(), images.size(0))
        stat_meters["cls_loss"].update(loss_dict["cls_loss"].item(), images.size(0))
        stat_meters["quan_loss"].update(loss_dict["quan_loss"].item(), images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            map_v = calc_map_eval(embeddings.sign(), labels)
            stat_meters["mAP"].update(map_v, images.size(0))
        torch.cuda.empty_cache()
    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}]"
        f"[bits:{args.n_bits}]"
        f"[epoch:{epoch}/{args.n_epochs - 1}]"
        f"[time:{(toc - tic):.3f}]"
        f"{sm_str}"
    )


def train_init(args):
    net = build_model(args, True)
    args_syn = Namespace(
        n_bits=args.n_bits,
        n_classes=args.n_classes,
        scale=args.scale,
        ps_alpha=args.ps_alpha,
        ps_mu=args.ps_mu,
        method=1,
        lambda_quan=getattr(args, "lambda_quan", 0.01),
    )
    # 始终使用 Norm_SoftMax，ps_mu 保持原值（默认为 0.8）
    criterion = Norm_SoftMax(args_syn).to(args.device)

    # 以下两行已被移除 ↓
    # if args.loss_type in ["normsoftmax"]:
    #     criterion.ps_mu = 0.0

    logger.info(f"number of learnable params: {calc_learnable_params(net)}")
    kwargs = {"lr": args.lr, "weight_decay": args.wd}
    if args.optimizer == "sgd":
        kwargs["momentum"] = args.momentum
    params_to_optimize = [
        {"params": net.parameters()},
        {"params": criterion.parameters()},
    ]
    optimizer = build_optimizer(args.optimizer, params_to_optimize, **kwargs)
    return net, criterion, optimizer


def train(args, train_loader, query_loader, dbase_loader):
    net, criterion, optimizer = train_init(args)
    early_stopping = EarlyStopping()
    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, criterion, optimizer, epoch)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                parallel_val=args.parallel_val,
            )
            if early_stop:
                break
    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"without improvement, will save & exit, "
            f"best mAP: {early_stopping.best_map:.3f}, "
            f"best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"reach epoch limit, will save & exit, "
            f"best mAP: {early_stopping.best_map:.3f}, "
            f"best epoch: {early_stopping.best_epoch}"
        )
    save_checkpoint(args, early_stopping.best_checkpoint)
    return early_stopping.best_epoch, early_stopping.best_map


def main():
    init()
    args = get_config()
    if "rename" in args and args.rename:
        rename_output(args)
    if not hasattr(args, "lambda_quan"):
        args.lambda_quan = 0.01
    dummy_logger_id = None
    rst = []
    for dataset in ["flickr", "nuswide" , "coco"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)
        train_loader, query_loader, dbase_loader = build_loaders(
            dataset,
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
        )
        args.n_samples = len(train_loader.dataset)
        for hash_bit in [16, 32, 64, 128]:
            print(f"processing hash-bit: {hash_bit}")
            seed_everything()
            args.n_bits = hash_bit
            args.save_dir = f"./output0.5/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pth") for x in os.listdir(args.save_dir)):
                print(f"*.pth exists in {args.save_dir}, will pass")
                continue
            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")
            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(
                    vars(args),
                    f,
                    indent=4,
                    sort_keys=True,
                    default=lambda o: o if type(o) in [bool, int, float, str] else str(type(o)),
                )
            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)
            rst.append(
                {
                    "dataset": dataset,
                    "hash_bit": hash_bit,
                    "best_epoch": best_epoch,
                    "best_map": best_map,
                }
            )
    print_in_md(rst)


if __name__ == "__main__":
    main()