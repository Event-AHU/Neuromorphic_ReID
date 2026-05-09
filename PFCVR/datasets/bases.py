import json
from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("VPT2I.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img, img_path


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, text = self.caption_pids[index], self.captions[index]

        caption = tokenize(text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption,text


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret


class ImageTextMLMWithBoxDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, darkimg_path, lightimg_path, noisyimg_path, caption, boxes_path, mask, prompt = self.dataset[index]
        img = read_image(img_path) 
        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        prompt = tokenize(prompt, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        # mask = torch.Tensor(mask)

        mlm_tokens = copy.deepcopy(caption_tokens)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(mlm_tokens.cpu().numpy())

        boxes = self.get_box(boxes_path)
        
        box_masks, new_part_text, part_label = self.get_box_with_mlm(boxes_path)
        # i2t=None
        
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'image_path': img_path,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'boxes': (box_masks, new_part_text, part_label,boxes["box_masks"],boxes["part_tokens"],boxes["i2t_list"],boxes["part_text"]),
            'mask': mask,
            "prompt": prompt
        }

        return ret

    def get_box(self,box_paths):
        
        # breakpoint()
        with open(box_paths, "r", encoding="utf8") as fp:
            f = json.load(fp)
        boxes = {
            "box_masks" : [],
            "part_text" : [],
            "part_tokens" : [],
            "label" : []
        }
        
        
        for box in f:
            # if box["score"] < 0.3:
            #     continue

            c = "Car "+ box["category"]
            if c not in boxes["part_text"]:
                boxes["part_text"].append(c)
                c = tokenize(c, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
                boxes["part_tokens"].append(c)
                boxes["box_masks"].append(box["mask"])
                boxes["label"].append(len(boxes["part_text"])-1)
            else:
                index = boxes["part_text"].index(c)
                matrix1 = boxes["box_masks"][index]
                matrix2 = box["mask"]
                boxes["box_masks"][index] = [[max(matrix1[i][j], matrix2[i][j]) for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
            # print(type(box["mask"]))
                # print("1",matrix1[8])
                # print("2",matrix2[8])
                # print("all",boxes["box_masks"][index][8])


        boxes["i2t_list"] = torch.Tensor([[0]*len(boxes["part_text"])]*len(boxes["box_masks"]))
        for j in range(len(boxes["label"])):
            boxes["i2t_list"][j][boxes["label"][j]] += 1


        # breakpoint()
        return boxes
    

    def get_box_with_mlm(self,box_paths):
        
        # breakpoint()
        with open(box_paths, "r", encoding="utf8") as fp:
            f = json.load(fp)
        box_masks = []
        part_text = []
        
        for box in f:
            c = "Car "+ box["category"]
            part_text.append(c)
            box_masks.append(box["mask"])

        new_part_text = []
        part_label = []
        mask = self.tokenizer.encoder["<|mask|>"]
        for t in part_text:
            t_tokens = tokenize(t, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
            t_mask = [0]*len(t_tokens)
            t_mask[2] = int(t_tokens[2])
            # print(t_tokens)
            t_tokens[2] = mask
            part_label.append(torch.tensor(t_mask))
            new_part_text.append(t_tokens)

            # breakpoint()
        return box_masks, new_part_text, part_label
    
    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)


class ImageTextMLMWithBoxDatasetByAugmentation(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, darkimg_path, lightimg_path, noisyimg_path, caption, boxes_path, mask, prompt = self.dataset[index]
        img = read_image(img_path) 
        darkimg = read_image(darkimg_path) 
        lightimg = read_image(lightimg_path) 
        noisyimg = read_image(noisyimg_path) 
        if self.transform is not None:
            img = self.transform(img)
            darkimg = self.transform(darkimg)
            lightimg = self.transform(lightimg)
            noisyimg = self.transform(noisyimg)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        prompt = tokenize(prompt, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        # mask = torch.Tensor(mask)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        boxes = self.get_box(boxes_path)
        
        box_masks, new_part_text, part_label = self.get_box_with_mlm(boxes_path)
        # i2t=None
        
        ret = {
            'pids': [pid,pid,pid,pid],
            'image_ids': [image_id,image_id,image_id,image_id],
            'images': torch.stack([img,darkimg,lightimg,noisyimg]),
            'caption_ids': torch.stack([caption_tokens,caption_tokens,caption_tokens,caption_tokens]),
            'mlm_ids': torch.stack([mlm_tokens,mlm_tokens,mlm_tokens,mlm_tokens]),
            'mlm_labels': torch.stack([mlm_labels,mlm_labels,mlm_labels,mlm_labels]),
            'boxes': (box_masks, new_part_text, part_label,boxes["box_masks"],boxes["part_tokens"],boxes["i2t_list"],boxes["part_text"]),
            'mask': [mask,mask,mask,mask],
            "prompt": prompt
        }

        return ret

    def get_box(self,box_paths):
        
        # breakpoint()
        with open(box_paths, "r", encoding="utf8") as fp:
            f = json.load(fp)
        boxes = {
            "box_masks" : [],
            "part_text" : [],
            "part_tokens" : [],
            "label" : []
        }
        
        
        for box in f:
            # if box["score"] < 0.3:
            #     continue

            c = "Car "+ box["category"]
            if c not in boxes["part_text"]:
                boxes["part_text"].append(c)
                c = tokenize(c, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
                boxes["part_tokens"].append(c)
                boxes["box_masks"].append(box["mask"])
                boxes["label"].append(len(boxes["part_text"])-1)
            else:
                index = boxes["part_text"].index(c)
                matrix1 = boxes["box_masks"][index]
                matrix2 = box["mask"]
                boxes["box_masks"][index] = [[max(matrix1[i][j], matrix2[i][j]) for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
            # print(type(box["mask"]))
                # print("1",matrix1[8])
                # print("2",matrix2[8])
                # print("all",boxes["box_masks"][index][8])


        boxes["i2t_list"] = torch.Tensor([[0]*len(boxes["part_text"])]*len(boxes["box_masks"]))
        for j in range(len(boxes["label"])):
            boxes["i2t_list"][j][boxes["label"][j]] += 1


        # breakpoint()
        return boxes
    

    def get_box_with_mlm(self,box_paths):
        
        # breakpoint()
        with open(box_paths, "r", encoding="utf8") as fp:
            f = json.load(fp)
        box_masks = []
        part_text = []
        
        for box in f:
            c = "Car "+ box["category"]
            part_text.append(c)
            box_masks.append(box["mask"])

        new_part_text = []
        part_label = []
        mask = self.tokenizer.encoder["<|mask|>"]
        for t in part_text:
            t_tokens = tokenize(t, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
            t_mask = [0]*len(t_tokens)
            t_mask[2] = int(t_tokens[2])
            # print(t_tokens)
            t_tokens[2] = mask
            part_label.append(torch.tensor(t_mask))
            new_part_text.append(t_tokens)

            # breakpoint()
        return box_masks, new_part_text, part_label

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)