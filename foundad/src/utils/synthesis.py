
import random
import math

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from skimage import morphology
import cv2


def generate_target_foreground_mask(img: np.ndarray, subclass: str) -> np.ndarray:
    inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    img_tensor = inv_normalize(img)

    img_tensor = torch.clamp(img_tensor, 0, 1)

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    img_np_uint8 = (img_np * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2BGR)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if subclass in ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor']:
        target_foreground_mask = np.ones_like(img_gray)
    elif subclass == 'pill':
        _, target_foreground_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        target_foreground_mask = (target_foreground_mask > 0).astype(int)
    elif subclass in ['hazelnut', 'metal_nut', 'toothbrush']:
        _, target_foreground_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        target_foreground_mask = (target_foreground_mask > 0).astype(int)
    elif subclass in ['bottle', 'capsule', 'grid', 'screw', 'zipper']:
        _, target_background_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        target_background_mask = (target_background_mask > 0).astype(int)
        target_foreground_mask = 1 - target_background_mask
    elif subclass in ['capsules']:
        target_foreground_mask = np.ones_like(img_gray)
    elif subclass in ['pcb1', 'pcb2', 'pcb3', 'pcb4']:
        _, target_foreground_mask = cv2.threshold(img_np_uint8[:, :, 2], 100, 255,
                                                    cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
        target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(8))
        target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
    elif subclass in ['candle', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pipe_fryum']:
        _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
        target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(3))
        target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
    elif subclass in ['bracket_black', 'bracket_brown', 'connector']:
        img_seg = img_np_uint8[:, :, 1]
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = 1 - target_background_mask
    elif subclass in ['bracket_white', 'tubes']:
        img_seg = img_np_uint8[:, :, 2]
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = target_background_mask
    elif subclass in ['metal_plate']:
        img_seg = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2GRAY)
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = 1 - target_background_mask
    else:
        raise NotImplementedError("Unsupported foreground segmentation category")

    target_foreground_mask = morphology.closing(
        target_foreground_mask, morphology.square(6))
    target_foreground_mask = morphology.opening(
        target_foreground_mask, morphology.square(6))

    return target_foreground_mask

class CutPaste(object):
    def __init__(self, colorJitter=0.1):
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(
                brightness=colorJitter,
                contrast=colorJitter,
                saturation=colorJitter,
                hue=colorJitter)

    def __call__(self, imgs):
        return imgs, imgs

class CutPasteNormal(CutPaste):
    def __init__(self, area_ratio=[0.02, 0.25], aspect_ratio=0.3, **kwargs):
        super().__init__(**kwargs)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, imgs, subclass):
        batch_size, _, h, w = imgs.shape
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i]
            augmented = self.process_image(img, subclass)
            augmented_imgs[i] = augmented

        return imgs, augmented_imgs

    def process_image(self, img, subclass):
        img = img.clone()
        _, h, w = img.shape

        target_foreground_mask = generate_target_foreground_mask(img, subclass)  # [H, W]


        area = h * w
        target_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * area
        aspect_ratio = random.uniform(self.aspect_ratio, 1 / self.aspect_ratio)

        cut_w = int(round(math.sqrt(target_area * aspect_ratio)))
        cut_h = int(round(math.sqrt(target_area / aspect_ratio)))

        if cut_w <= 0 or cut_h <= 0:
            return img

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)

        patch = img[:, from_y:from_y+cut_h, from_x:from_x+cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        mask_indices = np.argwhere(target_foreground_mask == 1)
        if len(mask_indices) == 0:
            return img 

        valid_indices = []
        for y, x in mask_indices:
            if y + cut_h <= h and x + cut_w <= w:
                valid_indices.append((y, x))

        if len(valid_indices) == 0:
            return img  

        to_y, to_x = random.choice(valid_indices)

        augmented = img.clone()
        augmented[:, to_y:to_y+cut_h, to_x:to_x+cut_w] = patch

        return augmented

class CutPasteScar(CutPaste):
    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, imgs, subclass):
        batch_size, _, h, w = imgs.shape
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i]
            augmented = self.process_image(img, subclass)
            augmented_imgs[i] = augmented

        return imgs, augmented_imgs

    def process_image(self, img, subclass):
        img = img.clone()
        _, h, w = img.shape

        target_foreground_mask = generate_target_foreground_mask(img, subclass)
    
        cut_w = int(random.uniform(*self.width))
        cut_h = int(random.uniform(*self.height))

        if cut_w <= 0 or cut_h <= 0:
            return img

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)

        patch = img[:, from_y:from_y+cut_h, from_x:from_x+cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        rot_deg = random.uniform(*self.rotation)
        patch = TF.rotate(patch, angle=rot_deg, interpolation=TF.InterpolationMode.BILINEAR, expand=True)

        _, patch_h, patch_w = patch.shape

        to_x = random.randint(0, w - patch_w)
        to_y = random.randint(0, h - patch_h)

        mask_indices = np.argwhere(target_foreground_mask == 1)
        if len(mask_indices) == 0:
            return img  

        valid_indices = []
        for y, x in mask_indices:
            if y + patch_h <= h and x + patch_w <= w:
                valid_indices.append((y, x))

        if len(valid_indices) == 0:
            return img  

        to_y, to_x = random.choice(valid_indices)

        augmented = img.clone()
        mask = torch.ones_like(patch)
        augmented = self.paste_with_mask(augmented, patch, mask, to_y, to_x)

        return augmented

    def paste_with_mask(self, img, patch, mask, top, left):
        _, h, w = img.shape
        _, patch_h, patch_w = patch.shape

        if top + patch_h > h or left + patch_w > w:
            return img

        img_region = img[:, top:top+patch_h, left:left+patch_w]
        mask = mask.to(img_region.device)
        img_region = img_region * (1 - mask) + patch * mask
        img[:, top:top+patch_h, left:left+patch_w] = img_region

        return img

class CutPasteUnion(object):
    def __init__(self, **kwargs):
        self.cutpaste_normal = CutPasteNormal(**kwargs)
        self.cutpaste_scar = CutPasteScar(**kwargs)

    def __call__(self, imgs, subclasses):
        batch_size = imgs.shape[0]
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i].unsqueeze(0)  # [1, C, H, W]
            subclass = subclasses[i]
            if random.random() < 0.5:
                _, augmented = self.cutpaste_normal(img, subclass)
            else:
                _, augmented = self.cutpaste_scar(img, subclass)
            augmented_imgs[i] = augmented.squeeze(0)

        return imgs, augmented_imgs