
import os
import torch
from torch.utils.data import Dataset, DataLoader, distributed
from torchvision import transforms
from torchvision.transforms import functional as TF
import random
import torchvision
import PIL
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class RandomRotate90or270:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.choice([90, 270])
            return TF.rotate(img, angle)
        return img

def build_base_transform(resize: int = 518):
    return [
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

def build_train_transform(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
    # use_random_erasing=False,
):
    ops = []

    if use_hflip:
        ops.append(transforms.RandomHorizontalFlip(p=0.2))

    if use_vflip:
        ops.append(transforms.RandomVerticalFlip(p=0.2))

    if use_rotate90:
        ops.append(RandomRotate90or270(p=0.2))

    if use_color_jitter:
        ops.append(transforms.RandomApply(
                [transforms.ColorJitter(0.3,0.3,0.3,0.05)],
                p=0.2
            )
        )

    if use_gray:
        ops.append(transforms.RandomGrayscale(p=0.1))

    if use_blur:
        ops.append(transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23 if resize >= 384 else 11, sigma=(0.1, 2.0))],
                p=0.2
            )
        )

    ops.extend(build_base_transform(resize))

    # if use_random_erasing:
    #     ops.append(transforms.RandomErasing(p=0.25, scale=(0.02,0.15), ratio=(0.3,3.3)))

    return transforms.Compose(ops)

def build_train_transform_new(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
    p_any=0.3,  # 全局：仅以 p_any 的概率做“一种”增强
):
    # 先收集可用的候选增强（每个都是“会改变图像”的操作）
    candidates = []
    if use_hflip:      
        candidates.append(transforms.RandomHorizontalFlip(p=1.0))
    if use_vflip:      
        candidates.append(transforms.RandomVerticalFlip(p=1.0))
    if use_rotate90:   
        candidates.append(RandomRotate90or270(p=1.0))
    if use_color_jitter:
        candidates.append(transforms.ColorJitter(0.3,0.3,0.3,0.05))
    if use_gray:       
        candidates.append(transforms.Lambda(lambda im: im.convert("L").convert("RGB")))
    if use_blur:
        candidates.append(transforms.GaussianBlur(kernel_size=23 if resize>=384 else 11, sigma=(0.1,2.0)))

    ops = []
    if candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(candidates)],
                p=p_any
            )
        )

    ops.extend(build_base_transform(resize))
    return transforms.Compose(ops)

class TrainDataset(torchvision.datasets.ImageFolder):

    def __init__(self, root: str, resize = 518, **kwargs):
        super().__init__(os.path.join(root, 'train'))
        self.resize = resize
        self.root = os.path.join(root, "train")
        self.transform = build_train_transform_new(
            self.resize,
            use_hflip=kwargs.get("use_hflip",False),
            use_vflip=kwargs.get("use_vflip",False),
            use_rotate90=kwargs.get("use_rotate90",False),
            use_color_jitter=kwargs.get("use_color_jitter",False),
            use_gray=kwargs.get("use_gray",False),
            use_blur=kwargs.get("use_blur",False),
        )
        self.samples = [(path, self.classes[target]) for (path, target) in self.samples]
        print(f"Totally {len(self.samples)} will be trained..")

    def __getitem__(self, index):
        """
        Overwrite the __getitem__ method to load images from both 'ad' and 'train' directories.
        """
        path_train, target = self.samples[index]
        image_train = self.loader(path_train).convert('RGB')
        image_train = self.transform(image_train)
                
        return image_train, target, path_train
    

class TestDataset(Dataset):

    def __init__(
        self,
        source,
        classname,
        resize=518,
        datasetname = "mvtec",
        **kwargs,
    ):
        super().__init__()
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.source = source
        self.classnames_to_use = [classname] 
        self.datasetname = datasetname

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = transforms.Compose(build_base_transform(resize))

        self.transform_mask = [
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, resize, resize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly not in ("good", "ok")),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, "test")
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]
                if self.datasetname == "mvtec":
                    if  anomaly != "good":
                        anomaly_mask_path = os.path.join(maskpath, anomaly)
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                        ]
                    else:
                        maskpaths_per_class[classname]["good"] = None
                elif self.datasetname == "visa":
                    if  anomaly != "ok":
                        anomaly_mask_path = os.path.join(maskpath, anomaly)
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                        ]
                    else:
                        maskpaths_per_class[classname]["ok"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.datasetname == "mvtec":
                        if anomaly != "good":
                            data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                        else:
                            data_tuple.append(None)
                    elif self.datasetname == "visa":
                        if anomaly != "ok":
                            data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                        else:
                            data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate



def build_dataloader(
    mode: str,
    root: str,
    batch_size: int,
    pin_mem: bool = True,
    **kwargs,
):
    """Return (dataset, dataloader, sampler).

    Parameters
    ----------
    mode : str
        "paired" | "mvtec"  —— decides which Dataset subclass to instantiate.
    root : str
        Root path for the dataset. For "paired" this should contain a
        ``train`` subdirectory; for "mvtec" it should be the directory that
        has all class folders (e.g. *mvtec/*).
    **kwargs : dict
        Extra arguments forwarded to the respective dataset constructor.
    """

    if mode == "train":
        dataset = TrainDataset(root=root, **kwargs)
        sampler = distributed.DistributedSampler(
            dataset,
        )
        drop_last = True
    elif mode == "test":
        dataset = TestDataset(source=root, **kwargs)
        sampler = None  # evaluation usually not distributed
        drop_last = False
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and mode == "test"),
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return dataset, dataloader, sampler
