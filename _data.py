import configparser  #读取配置文件（.ini格式）
import os.path as osp  #操作系统路径处理
import pickle  #Python对象序列化，用于CIFAR10数据集
import platform  #获取操作系统信息
import numpy as np  #数值计算库
import torch  #Pytorch深度学习框架
from PIL import Image  #Pyton图像处理库
from torch.utils.data import DataLoader, Dataset #Pytorch数据处理工具
from torchvision import transforms as T  #图像预处理和增强

#数据集配置函数
#根据数据集名称返回类别数量
def get_class_num(name):
    r = {"cifar": 10, "flickr": 38, "nuswide": 21, "coco": 80, "imagenet": 100}[name]
    return r

#返回每个数据集评估时使用的 top-k值，None表示使用全部样本
def get_topk(name):
    r = {"cifar": None, "flickr": None, "nuswide": 5000, "coco": None, "imagenet": 1000}[name]
    return r

#从文件中读取类别概念（标签名称）列表
def get_concepts(name, root):
    with open(osp.join(root, name, "concepts.txt"), "r") as f:
        lines = f.read().splitlines()
    return np.array(lines)

#数据预处理函数，构建图像预处理流水线
def build_trans(usage, resize_size=256, crop_size=224):
    if usage == "train":
        steps = [T.RandomCrop(crop_size), T.RandomHorizontalFlip()]#训练时随机裁剪+随机水平旋转
    else:
        steps = [T.CenterCrop(crop_size)]#验证/测试时中心裁剪
    return T.Compose(
        [T.Resize(resize_size)]
        + steps
        + [
            T.ToTensor(),
            # T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),#归一化使用ImageNet数据集的标准均值和标准差
        ]
    )

#数据加载器构建函数，构建三个数据加载器（训练、查询、数据库）
def build_loaders(name, root, **kwargs):
    train_trans = build_trans("train")
    other_trans = build_trans("other")

    data = init_dataset(name, root)

    train_loader = DataLoader(ImageDataset(data.train, train_trans), shuffle=True, drop_last=True, **kwargs)
    # generator=torch.Generator(): to keep torch.get_rng_state() unchanged!
    # https://discuss.pytorch.org/t/does-a-dataloader-change-random-state-even-when-shuffle-argument-is-false/92569/4
    query_loader = DataLoader(ImageDataset(data.query, other_trans), generator=torch.Generator(), **kwargs)
    dbase_loader = DataLoader(ImageDataset(data.dbase, other_trans), generator=torch.Generator(), **kwargs)

    return train_loader, query_loader, dbase_loader

#基础数据集类，负责加载和处理数据
class BaseDataset(object):
    """
    Base class of dataset
    """
    #初始化参数：数据集名称、标签文件根目录、图像文件根目录
    def __init__(self, name, txt_root, img_root, verbose=True):

        self.img_root = img_root

        self.train_txt = osp.join(txt_root, "train.txt")
        self.query_txt = osp.join(txt_root, "query.txt")
        self.dbase_txt = osp.join(txt_root, "dbase.txt")
        #检查文件是否存在
        self.check_before_run()
        #加载数据
        self.train = self.process(self.train_txt)
        self.query = self.process(self.query_txt)
        self.dbase = self.process(self.dbase_txt)
        #转化路径
        self.set_img_abspath()  # 1.jpg -> /home/x/COCO/images/1.jpg
        #打印统计信息
        if verbose:
            print(f"=> {name.upper()} loaded")
            self.print_dataset_statistics()
    #在加载数据前检查必要的文件是否存在
    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_txt):
            raise RuntimeError("'{}' is not available".format(self.train_txt))
        if not osp.exists(self.query_txt):
            raise RuntimeError("'{}' is not available".format(self.query_txt))
        if not osp.exists(self.dbase_txt):
            raise RuntimeError("'{}' is not available".format(self.dbase_txt))
    #计算数据集的类别数和图象数
    def get_imagedata_info(self, data):
        labs = data[1]
        n_cids = (labs.sum(axis=0) > 0).sum()
        n_imgs = len(data[0])
        return n_cids, n_imgs
    #打印统计信息
    def print_dataset_statistics(self):
        n_train_cids, n_train_imgs = self.get_imagedata_info(self.train)
        n_query_cids, n_query_imgs = self.get_imagedata_info(self.query)
        n_dbase_cids, n_dbase_imgs = self.get_imagedata_info(self.dbase)

        print("Image Dataset statistics:")
        print("  -----------------------------")
        print("  subset | # images | # classes")
        print("  -----------------------------")
        print("  train  | {:8d} | {:9d}".format(n_train_imgs, n_train_cids))
        print("  query  | {:8d} | {:9d}".format(n_query_imgs, n_query_cids))
        print("  dbase  | {:8d} | {:9d}".format(n_dbase_imgs, n_dbase_cids))
        print("  -----------------------------")
    #解析标签文件
    def process(self, txt_path):
        imgs, labs = [], []  #（图像文件名 + 多标签编码）
        for x in open(txt_path, "r").readlines():
            parts = x.split()
            imgs.append(parts[0])
            labs.append(parts[1:])
        #输出图像文件名数组和对应标签矩阵
        imgs = np.array(imgs)
        labs = np.array(labs, dtype=np.float32)
        return (imgs, labs)
    #相对路径转换为绝对路径
    def set_img_abspath(self):
        for x in ["train", "query", "dbase"]:
            imgs, labs = getattr(self, x)
            # imgs = [osp.join(self.img_root, img) for img in imgs]
            imgs = np.char.add(f"{self.img_root}/", imgs)
            setattr(self, x, (imgs, labs))

class COCO(BaseDataset):
    """COCO 数据集特殊处理类，图片在 train2017 和 val2017 子文件夹中"""

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    def set_img_abspath(self):
        for x in ["train", "query", "dbase"]:
            imgs, labs = getattr(self, x)

            img_paths = []
            for img_name in imgs:
                img_id = img_name.split('.')[0]  # 去掉扩展名

                # 先尝试在 train2017 中查找
                train_path = osp.join(self.img_root, "train2017", f"{img_id}.jpg")
                val_path = osp.join(self.img_root, "val2017", f"{img_id}.jpg")

                if osp.exists(train_path):
                    img_paths.append(train_path)
                elif osp.exists(val_path):
                    img_paths.append(val_path)
                else:
                    # 如果都找不到，尝试在 images 根目录查找（某些情况下可能存在）
                    root_path = osp.join(self.img_root, img_name)
                    if osp.exists(root_path):
                        img_paths.append(root_path)
                    else:
                        # 最后尝试直接拼接（可能会失败）
                        img_paths.append(osp.join(self.img_root, img_name))
                        print(f"警告: COCO 数据集找不到图片 {img_name}")

            imgs = np.array(img_paths)
            setattr(self, x, (imgs, labs))

#特殊性: CIFAR数据集不是以图像文件存储，而是以pickle格式存储二进制数据
#数据格式: 原始数据形状为 [N, 3072]，需要重塑为 [N, 3, 32, 32]
#索引转换: 根据文件名中的索引提取对应的图像数据
class CIFAR(BaseDataset):

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    @staticmethod
    def unpickle(file):
        with open(file, "rb") as fo:
            dic = pickle.load(fo, encoding="latin1")
        return dic

    def set_img_abspath(self):
        # get all img data from data_batch_1~5 & test_batch
        data_list = [f"data_batch_{x}" for x in range(1, 5 + 1)]
        data_list.append("test_batch")
        imgs = []
        for x in data_list:
            data = self.unpickle(osp.join(self.img_root, x))
            imgs.append(data["data"])
            # labs.extend(data["labels"])
        imgs = np.vstack(imgs).reshape(-1, 3, 32, 32)  #堆叠并重塑形状
        imgs = imgs.transpose((0, 2, 3, 1))  # 从 [N, C, H, W] 转换为 [N, H, W, C]

        # change image file name to image data
        for x in ["train", "query", "dbase"]:
            _imgs, _labs = getattr(self, x)
            idxes = [int(x.replace(".png", "")) for x in _imgs] #提取索引
            setattr(self, x, (imgs[idxes], _labs)) #用实际图像数据替换文件名

#数据集工厂字典 ，映射数据集名称到对应的处理类
_ds_factory = {
    "cifar": CIFAR,
    "nuswide": BaseDataset,
    "flickr": BaseDataset,
    "coco": COCO,
    "imagenet": BaseDataset,
}

#初始化并返回数据集对象
def init_dataset(name, root, **kwargs):

    if name not in list(_ds_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(_ds_factory.keys())))

    txt_root = osp.join(root, name)
    #根据location.ini配置文件或默认规则确定图像路径
    ini_loc = osp.join(root, name, "images", "location.ini")
    if osp.exists(ini_loc):
        config = configparser.ConfigParser()
        config.read(ini_loc)
        #根据主机名判断运行环境（学校集群 vs 本地）
        if "wfu.edu.cn" in platform.node():
            img_root = config["DEFAULT"]["SLURM"]
        else:
            img_root = config["DEFAULT"][platform.system()]
    else:
        img_root = osp.join(root, name)

    return _ds_factory[name](name, txt_root, img_root, **kwargs)

#将数据包装为PyTorch Dataset
class ImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        img, lab = self.data[0][idx], self.data[1][idx]
        if isinstance(img, str):  # 如果img是文件路径
            img = Image.open(img).convert("RGB")  # 打开图像并转换为RGB
        else:  # 如果img是numpy数组（CIFAR数据集）
            img = Image.fromarray(img)  # 从数组创建图像
        if self.transform is not None:
            img = self.transform(img)  # 应用预处理
        return img, lab, idx  # 返回图像、标签、索引

    def get_all_labels(self):
        return torch.from_numpy(self.data[1])  # 返回所有标签的Tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    db_name = "imagenet"
    root = "./_datasets"

    dataset = init_dataset(db_name, root)

    trans = T.Compose(
        [
            # T.ToPILImage(),
            T.Resize([224, 224]),
            T.ToTensor(),
        ]
    )

    train_set = ImageDataset(dataset.dbase, trans)
    dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
    concepts = get_concepts(db_name, root)

    for imgs, labs, _ in dataloader:
        print(imgs.shape, labs)
        plt.imshow(imgs[0].numpy().transpose(1, 2, 0))
        titles = concepts[labs[0].nonzero().squeeze(1)]
        plt.title(titles)
        plt.show()
        break
