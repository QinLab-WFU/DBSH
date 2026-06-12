import argparse
from os import path as osp


def get_config():
    parser = argparse.ArgumentParser(description=osp.basename(osp.dirname(__file__)))

    # -----------------------------
    # common settings
    # -----------------------------
    parser.add_argument("--backbone", type=str, default="resnet50", help="see network.py")
    parser.add_argument("--data-dir", type=str, default="/dk0/home/user011/project/DSpH/_datasets",
                        help="directory to dataset")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=200, help="input batch size")
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd/rmsprop/adam/amsgrad/adamw")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=4e-3, help="weight decay")
    parser.add_argument("--device", type=str, default="cuda:0", help="device (accelerator) to use")
    parser.add_argument("--parallel-val", type=bool, default=True, help="use a separate thread for validation")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of SGD")

    # changed at runtime
    parser.add_argument("--dataset", type=str, default="cifar", help="cifar/nuswide/flickr/coco")
    parser.add_argument("--n-classes", type=int, default=24, help="number of dataset classes")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")
    parser.add_argument("--save-dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n-bits", type=int, default=128, help="length of hashing binary")

    # -----------------------------
    # loss switching (only normsoftmax related)
    # -----------------------------
    parser.add_argument(
        "--loss-type",
        type=str,
        default="normsoftmax",
        choices=["normsoftmax", "ps_normsoftmax"],
        help="criterion type (normsoftmax without PS, ps_normsoftmax with PS)"
    )

    # loss hyper-params
    parser.add_argument("--scale", type=float, default=16.0, help="scale for logits")
    parser.add_argument("--lambda-quan", type=float, default=0.5, help="weight for quantization loss")
    # -----------------------------
    # proxy synthesis (PS)
    # -----------------------------
    parser.add_argument("--ps-alpha", type=float, default=1.0, help="Beta(alpha, alpha) for PS")
    parser.add_argument("--ps-mu", type=float, default=1.0, help="#synthetics = ps_mu * batch_size")

    args = parser.parse_args()
    return args