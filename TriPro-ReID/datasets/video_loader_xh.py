from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import math
import torch
from torch.utils.data import Dataset
import random


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'dense']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        S = self.seq_len  # 4
        img_paths, img_paths_e, pid, camid, trackid = self.dataset[index]
        # img_paths, img_paths_e = all_img_paths[:len(all_img_paths)//2],  all_img_paths[len(all_img_paths)//2:]
        num = len(img_paths)  # 27
        """rss 操作"""
        sample_clip = []
        frame_indices = list(range(num))
        if num < S:  # 8 = chunk的数目，每个tracklet分成8段，每段随机选一帧
            strip = list(range(num)) + [frame_indices[-1]] * (S - num)
            for s in range(S):
                pool = strip[s * 1:(s + 1) * 1]
                sample_clip.append(list(pool))
        else:
            inter_val = math.ceil(num / S)
            strip = list(range(num)) + [frame_indices[-1]] * (inter_val * S - num)
            for s in range(S):
                pool = strip[inter_val * s:inter_val * (s + 1)]
                sample_clip.append(list(pool))

        sample_clip = np.array(sample_clip)

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)

            imgseq = []
            imgseq_e = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img_path_e = img_paths_e[index]
                img = Image.open(img_path).convert('RGB')  # 3x224x112
                img_e = Image.open(img_path_e).convert('RGB')  # 3x224x112
                imgseq.append(img)
                imgseq_e.append(img_e)
                
            seq = [imgseq, imgseq_e]
            if self.transform is not None:
                seq = self.transform(seq)

            img_tensor = torch.stack(seq[0], dim=0)  # seq_len 4x3x224x112
            img_tensor_e = torch.stack(seq[1], dim=0) 
            flow_tensor = None

            return torch.stack([img_tensor, img_tensor_e]), pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = list(range(num))  # 27
            indices_list = []
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index += self.seq_len

            last_seq = frame_indices[cur_index:]

            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)

            indices_list.append(last_seq)  # <class 'list'>: [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 24, 25, 24, 25, 24, 25]]
            imgs_list = []
            imgs_list_e =[]
            for indices in indices_list:  # <class 'list'>: [0, 1, 2, 3, 4, 5, 6, 7]
                imgseq = []
                imgseq_e = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img_path_e = img_paths_e[index]
                    img = Image.open(img_path).convert('RGB')  # 3x224x112
                    img_e = Image.open(img_path_e).convert('RGB')  # 3x224x112
                    imgseq.append(img)
                    imgseq_e.append(img_e)

                seq = [imgseq, imgseq_e]
                if self.transform is not None:
                    seq = self.transform(seq)

                imgs = torch.stack(seq[0], dim=0)  # seq_len 4x3x224x112
                imgs_e = torch.stack(seq[1], dim=0) 
                
                imgs_list.append(imgs)
                imgs_list_e.append(imgs_e)

            imgs_tensor = torch.stack(imgs_list)  # torch.Size([13, 8, 3, 224, 112])
            imgs_tensor_e = torch.stack(imgs_list_e)  # torch.Size([13, 8, 3, 224, 112])
            # flow_tensor = None
            return torch.stack([imgs_tensor, imgs_tensor_e]), pid, camid, trackid, ""

        elif self.sample == 'rrs_train':
            idx = np.random.choice(sample_clip.shape[1], sample_clip.shape[0])
            number = sample_clip[np.arange(len(sample_clip)), idx]
            # imgseq = []
            img_paths = np.array(list(img_paths))  # img_paths原始为tuple，转换成数组
            img_paths_e = np.array(list(img_paths_e))  # img_paths原始为tuple，转换成数组
            # flow_paths = np.array([img_path.replace('Mars', 'Mars_optical') for img_path in img_paths])
            imgseq = [Image.open(img_path).convert('RGB') for img_path in img_paths[number]]
            imgseq_e = [Image.open(img_path_e).convert('RGB') for img_path_e in img_paths_e[number]]
            # flowseq = [Image.open(flow_path).convert('RGB') for flow_path in flow_paths[number]]

            seq = [imgseq, imgseq_e]
            # seq = [imgseq, flowseq]
            if self.transform is not None:
                seq = self.transform(seq)

            img_tensor = torch.stack(seq[0], dim=0)  # seq_len 4x3x224x112
            img_tensor_e = torch.stack(seq[1], dim=0)  # seq_len 4x3x224x112

            return torch.stack([img_tensor, img_tensor_e]), pid, camid, trackid, ""

        elif self.sample == 'rrs_test':
            number = sample_clip[:, 0]
            img_paths = np.array(list(img_paths))  # img_paths原始为tuple，转换成数组
            img_paths_e = np.array(list(img_paths_e))  # img_paths原始为tuple，转换成数组
            # flow_paths = np.array([img_path.replace('Mars', 'Mars_optical') for img_path in img_paths])
            imgseq = [Image.open(img_path).convert('RGB') for img_path in img_paths[number]]
            imgseq_e = [Image.open(img_path_e).convert('RGB') for img_path_e in img_paths_e[number]]
            # flowseq = [Image.open(flow_path).convert('RGB') for flow_path in flow_paths[number]]

            seq = [imgseq, imgseq_e]
            # seq = [imgseq, flowseq]
            if self.transform is not None:
                seq = self.transform(seq)
            img_tensor = torch.stack(seq[0], dim=0)  # seq_len 4x3x224x112
            img_tensor_e = torch.stack(seq[1], dim=0)  # seq_len 4x3x224x112
            return torch.stack([img_tensor, img_tensor_e]), pid, camid, trackid, ""
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


