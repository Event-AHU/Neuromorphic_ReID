import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class T2IVeRi(BaseDataset):
    dataset_dir = 'T2I_VeRi'
    anno_filename = 'reid_with_mask_prompt_and_boxes_filepath_refixID.json'

    def __init__(self, root='', verbose=True):
        super(T2IVeRi, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        # self.img_dir = op.join(self.dataset_dir, 'image/')
        self.img_dir = self.dataset_dir

        self.anno_path = op.join(self.dataset_dir, self.anno_filename)
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> RSTPReid Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        if len(test_annos) == 0 and len(val_annos) > 0:
            test_annos = val_annos
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            pid_list = sorted({int(anno['id']) for anno in annos})
            pid2label = {pid: idx for idx, pid in enumerate(pid_list)}
            for anno in annos:
                pid = int(anno['id'])
                pid_label = pid2label[pid]
                pid_container.add(pid_label)
                img_path = op.join(self.img_dir, anno['file_path'])
                # breakpoint()
                darkimg_path = op.join(self.img_dir, anno['file_path'].replace("image/","Darkimage/dark_"))
                lightimg_path = op.join(self.img_dir, anno['file_path'].replace("image/","Lightimage/light_"))
                noisyimg_path = op.join(self.img_dir, anno['file_path'].replace("image/","Noisyimage/noisy_"))
                captions = anno['captions'] # caption list
                boxes = op.join(self.img_dir,anno['boxes'].replace("\\","/"))
                mask = anno['mask']
                prompt = anno['prompt']
                for caption in captions:
                    dataset.append((pid_label, image_id, img_path,darkimg_path,lightimg_path,noisyimg_path, caption,boxes,mask,prompt))
                image_id += 1
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))


class T2IVeRiNew(T2IVeRi):
    dataset_dir = 'T2I_VeRi_new'
    anno_filename = 'reid_with_mask_prompt_and_boxes_filepath_prefixid_idsplit_70_30.json'
