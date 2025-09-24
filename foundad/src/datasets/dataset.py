
import os
import torch
from torch.utils.data import Dataset, DataLoader, distributed
from torchvision import transforms
import torchvision
import PIL
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TrainDataset(torchvision.datasets.ImageFolder):

    def __init__(self, root: str, resize = 518):
        super().__init__(os.path.join(root, 'train'))
        self.resize = resize
        self.root = os.path.join(root, "train")
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.samples = [(path, self.classes[target]) for (path, target) in self.samples]

    def __getitem__(self, index):
        """
        Overwrite the __getitem__ method to load images from both 'ad' and 'train' directories.
        """
        path_train, target = self.samples[index]
        image_train = self.loader(path_train).convert('RGB')

        if self.transform is not None:
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

        self.transform_img = [
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

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
