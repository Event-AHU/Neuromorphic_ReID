import logging
import torch
try:
    import torchvision.transforms as T
except ModuleNotFoundError:
    import random
    import numpy as np
    from PIL import Image, ImageOps

    class _Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, x):
            for t in self.transforms:
                x = t(x)
            return x

    class _Resize:
        def __init__(self, size):
            self.size = size

        def __call__(self, img):
            if not isinstance(img, Image.Image):
                return img
            h, w = self.size
            return img.resize((w, h), resample=Image.BILINEAR)

    class _RandomHorizontalFlip:
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, img):
            if not isinstance(img, Image.Image):
                return img
            if random.random() < self.p:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            return img

    class _Pad:
        def __init__(self, padding, fill=0):
            self.padding = padding
            self.fill = fill

        def __call__(self, img):
            if not isinstance(img, Image.Image):
                return img
            return ImageOps.expand(img, border=self.padding, fill=self.fill)

    class _RandomCrop:
        def __init__(self, size):
            self.size = size

        def __call__(self, img):
            if not isinstance(img, Image.Image):
                return img
            th, tw = self.size
            w, h = img.size
            if w == tw and h == th:
                return img
            if w < tw or h < th:
                pad_w = max(tw - w, 0)
                pad_h = max(th - h, 0)
                img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
                w, h = img.size
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            return img.crop((j, i, j + tw, i + th))

    class _ToTensor:
        def __call__(self, img):
            if torch.is_tensor(img):
                return img
            if not isinstance(img, Image.Image):
                return img
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous() / 255.0
            return tensor

    class _Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def __call__(self, tensor):
            if not torch.is_tensor(tensor):
                return tensor
            mean = self.mean.to(dtype=tensor.dtype, device=tensor.device)
            std = self.std.to(dtype=tensor.dtype, device=tensor.device)
            return (tensor - mean) / std

    class _RandomErasing:
        def __init__(self, scale=(0.02, 0.4), value=0):
            self.scale = scale
            self.value = value

        def __call__(self, tensor):
            if not torch.is_tensor(tensor):
                return tensor
            if tensor.ndim != 3:
                return tensor
            c, h, w = tensor.shape
            area = h * w
            target_area = area * random.uniform(self.scale[0], self.scale[1])
            erase_h = int(round((target_area) ** 0.5))
            erase_w = int(round((target_area) ** 0.5))
            if erase_h <= 0 or erase_w <= 0 or erase_h >= h or erase_w >= w:
                return tensor
            top = random.randint(0, h - erase_h)
            left = random.randint(0, w - erase_w)
            if isinstance(self.value, (list, tuple)):
                for ch in range(min(c, len(self.value))):
                    tensor[ch, top:top + erase_h, left:left + erase_w] = float(self.value[ch])
            else:
                tensor[:, top:top + erase_h, left:left + erase_w] = float(self.value)
            return tensor

    class _T:
        Compose = _Compose
        Resize = _Resize
        RandomHorizontalFlip = _RandomHorizontalFlip
        Pad = _Pad
        RandomCrop = _RandomCrop
        ToTensor = _ToTensor
        Normalize = _Normalize
        RandomErasing = _RandomErasing

    T = _T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset,ImageTextMLMWithBoxDataset,ImageTextMLMWithBoxDatasetByAugmentation

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .v2iveri import T2IVeRi, T2IVeRiNew

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid, 'T2I_VeRi':T2IVeRi, 'T2I_VeRi_new': T2IVeRiNew}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        elif isinstance(v[0], tuple):
            batch_tensor_dict.update({k: v})
        elif isinstance(v[0], dict):
            batch_tensor_dict.update({k: v})
        elif isinstance(v[0], list):
            # print(v[0])
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif isinstance(v[0],str):
            batch_tensor_dict.update({k: v})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    args_dict = vars(args)
    current_task = [l.strip() for l in args.loss_names.split('+')]

    loss_weights = {}
    for t in current_task:
        if t+"_loss_weight" in args_dict.keys():
            loss_weights[t] = args_dict[t+"_loss_weight"]
        else:
            loss_weights[t] = 1.0

    logger.info('loss weights '+str(loss_weights))

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)
    
    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        if args.MLM:
            if args.augmentation:
                train_set = ImageTextMLMWithBoxDatasetByAugmentation(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
            else:
                train_set = ImageTextMLMWithBoxDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
        else:
            train_set = ImageTextDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)

            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate,
                                          pin_memory=True,
                                          persistent_workers=True if num_workers > 0 else False)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate,
                                      pin_memory=True,
                                      persistent_workers=True if num_workers > 0 else False)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.test_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    persistent_workers=True if num_workers > 0 else False)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.test_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    persistent_workers=True if num_workers > 0 else False)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        # breakpoint()
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     persistent_workers=True if num_workers > 0 else False)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     persistent_workers=True if num_workers > 0 else False)
        return test_img_loader, test_txt_loader, num_classes
