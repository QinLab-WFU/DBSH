import importlib
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import timm
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from _utils import calc_learnable_params


def get_attr(obj, pos):
    if isinstance(pos, tuple):
        return getattr(obj, pos[0])[pos[1]]
    return getattr(obj, pos)


def set_attr(obj, pos, val):
    if isinstance(pos, tuple):
        getattr(obj, pos[0])[pos[1]] = val
    else:
        setattr(obj, pos, val)


def build_default_model(backbone, n_bits, pretrained, need_pos=False):
    if backbone in torchvision.models.__dict__.keys():
        weights = torchvision.models.get_model_weights(backbone)["IMAGENET1K_V1"] if pretrained else None
        net = torchvision.models.__dict__[backbone](weights=weights)
        if "resnet" in backbone:
            print("!!!!!backbone===resnet!!!!!")
            last_layer_pos = "fc"
        elif backbone in ["alexnet"] or "vgg" in backbone:
            print("!!!!!backbone===alexnet or vgg!!!!!")
            last_layer_pos = ("classifier", -1)
        elif "swin_" in backbone:
            print("!!!!!backbone===swin_!!!!!")
            last_layer_pos = "head"
        elif "vit_" in backbone:
            print("!!!!!backbone===vit_!!!!!")
            last_layer_pos = ("heads", -1)
        else:
            raise NotImplementedError(f"not implemented backbone: {backbone}")

        set_attr(net, last_layer_pos, nn.Linear(get_attr(net, last_layer_pos).in_features, n_bits))
    elif backbone in timm.list_models(["vit_*_patch16_224", "swin_*_patch4_window7_224"]):
        # vit_*_patch16_224
        # swin_*_patch4_window7_224
        net = timm.create_model(backbone, pretrained=pretrained)
        net.head = nn.Linear(net.head.in_features, n_bits)
        last_layer_pos = "head"
    elif "mamba_vision_" in backbone:
        # mamba_vision_T/*_S/*_B/*_L
        module = importlib.import_module("MambaVision.test")
        mamba_vision_X = getattr(module, backbone)
        net = mamba_vision_X(pretrained=pretrained)
        net.head = nn.Linear(net.head.in_features, n_bits)
        last_layer_pos = "head"
    elif "mlla_" in backbone:
        # mlla_tiny/mlla_small/mlla_base
        module = importlib.import_module("MLLA.test")
        mlla_X = getattr(module, backbone)
        net = mlla_X(pretrained)
        net.head = nn.Linear(net.head.in_features, n_bits)
        last_layer_pos = "head"
    else:
        raise NotImplementedError(f"not support backbone: {backbone}")

    # leave for application to init
    # if pretrained:
    # last_layer = get_attr(net, last_layer_pos)

    # ASL
    # nn.init.kaiming_normal_(last_layer.weight, mode="fan_out")

    # 20240930: use the following method for initialization
    # nn.init.xavier_uniform_(last_layer.weight)
    # nn.init.constant_(last_layer.bias, 0)

    return (net, last_layer_pos) if need_pos else net


def build_model(args, pretrained=True, model_cls=None):
    allowed_items = {"resnet50", "frozen", "double", "normalize", "layernorm", "tanh"}
    current_items = set(args.backbone.split("_"))

    invalid_items = current_items - allowed_items
    if not invalid_items:
        kwargs = {item: True for item in current_items}
        if kwargs.pop("resnet50", False):
            model_cls = model_cls or ResNet50
            net = model_cls(args.n_bits, pretrained, **kwargs).to(args.device)
            return net

    raise NotImplementedError(f"not support: {args.backbone}")


class ResNet50(nn.Module):
    def __init__(self, n_bits, pretrained=True, **kwargs):
        super().__init__()

        self.frozen = kwargs.pop("frozen", False)
        self.double = kwargs.pop("double", False)
        self.normalize = kwargs.pop("normalize", False)
        self.layernorm = kwargs.pop("layernorm", False)
        self.tanh = kwargs.pop("tanh", False)

        # 定义本地预训练权重路径
        local_weights_path = "/dk0/home/user011/project/DSpH/pretrained/resnet50-0676ba61.pth"

        # 加载不带预训练权重的模型
        self.backbone = torchvision.models.resnet50(weights=None)

        # 如果pretrained为True，加载本地权重
        if pretrained:
            if os.path.exists(local_weights_path):
                print(f"Loading local pretrained weights from: {local_weights_path}")
                state_dict = torch.load(local_weights_path, map_location='cpu')
                self.backbone.load_state_dict(state_dict)
            else:
                print(f"Warning: Local weights not found at {local_weights_path}")
                print("Using ImageNet pre-trained weights from torchvision...")
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
                self.backbone = torchvision.models.resnet50(weights=weights)

        self.dim_feature = self.backbone.fc.in_features

        # 替换最后的全连接层
        self.backbone.fc = nn.Linear(self.dim_feature, n_bits)

        # 初始化新的全连接层
        nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.zeros_(self.backbone.fc.bias)

        # IDML
        # nn.init.kaiming_normal_(self.backbone.fc.weight, mode="fan_out")
        # nn.init.constant_(self.backbone.fc.bias, 0)

        if self.double:
            self.backbone.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))

        if self.frozen:
            for module in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.backbone.modules()):
                module.eval()
                module.train = lambda _: None

        if self.layernorm:
            # elementwise_affine=False means no learnable parameters
            self.layer_norm = nn.LayerNorm(n_bits, elementwise_affine=False)

    def forward(self, x: torch.Tensor):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        if self.double:
            x = self.backbone.avgpool(x) + self.backbone.maxpool2(x)
        else:
            x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        if self.layernorm:
            x = self.layer_norm(x)

        if self.normalize:
            x = F.normalize(x, dim=-1)

        if self.tanh:
            x = torch.tanh(x)

        return x


if __name__ == "__main__":
    # for x in ["alexnet", "resnet18", "resnet50", "vgg19", "swin_t", "vit_b_16"]:
    for x in [
        "mamba_vision_T",
        "mamba_vision_S",
        "mamba_vision_B",
        "mamba_vision_L",
        # "swin_t",
        # "swin_s",
        # "swin_b",
        # "vit_b_16",
        # "vit_l_16",
        "swin_tiny_patch4_window7_224",
        "swin_small_patch4_window7_224",
        "swin_base_patch4_window7_224",
        "swin_large_patch4_window7_224",
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_base_patch16_224",
        "vit_large_patch16_224",
    ]:
        net = build_default_model(x, 20, False)
        print(f"{x} 参数量：{calc_learnable_params(net) / 1e6:.1f}M")
