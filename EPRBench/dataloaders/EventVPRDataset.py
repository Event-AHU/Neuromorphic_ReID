import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
import os
import random
import numpy as np

class EventVPRDataset(Dataset):
    def __init__(self, dataset_folder, split="train", transform=None, img_per_place=5, min_img_per_place=5, use_text=True, text_folder=None):
        self.dataset_folder = Path(dataset_folder)
        self.split = split
        self.transform = transform if transform else self._default_transform()
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.use_text = use_text  # Whether to use text description
        self.text_folder = Path(text_folder)
        
        if self.split == "train":
            self.places_data = {} # Stores {place_numerical_id: [list of img_paths]}
            self.place_ids = []   # Stores unique numerical place IDs for training
            self.text_descriptions = {}  # Store scene text descriptions {place_numerical_id: text_description}
            self._load_train_dataset()
        else: # val or test
            self._database_images_paths = []
            self._query_images_paths = []
            self.all_images_paths = [] # Combined list of database and query images for evaluation
            self._all_place_ids = [] # Corresponding place IDs for all_images_paths
            self._database_num = 0
            self._queries_num = 0
            self.text_descriptions = {}  # Store scene text descriptions {image_path: text_description}
            self._load_eval_dataset()

    def _default_transform(self):
        return T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_place_numerical_id(self, place_dir_name):
        parts = place_dir_name.split('@')
        if len(parts) >= 3:
            place_id_str = '@'.join(parts[:3])
            return hash(place_id_str) % (10**6)
        return None # Should not happen with valid input

    def _load_train_dataset(self):
        split_path = self.dataset_folder / self.split
        
        current_places_data = {}

        # For training, we assume we use both database and queries for each scene
        for sub_folder_name in ["database", "queries"]:
            current_sub_path = split_path / sub_folder_name
            if not current_sub_path.exists():
                continue

            for place_dir in sorted(current_sub_path.iterdir()):
                if place_dir.is_dir() and place_dir.name.startswith('@'):
                    place_numerical_id = self._get_place_numerical_id(place_dir.name)
                    if place_numerical_id is None:
                        continue
                    
                    if place_numerical_id not in current_places_data:
                        current_places_data[place_numerical_id] = []
                    
                    # If text description is enabled and the scene does not have a text description yet, load it from the dedicated folder
                    if self.use_text and place_numerical_id not in self.text_descriptions:
                        # Try multiple possible filename formats
                        possible_filenames = [
                            f"{place_dir.name}.txt",  # Original format
                            f"{place_dir.name.split('@')[0]}.txt",  # Use only the first part
                            f"{place_dir.name.replace('@', '_')}.txt",  # Replace @ with _
                            f"{place_numerical_id}.txt"  # Use numerical ID
                        ]
                        
                        found_file = False
                        for filename in possible_filenames:
                            txt_file_path = self.text_folder / filename
                            if txt_file_path.exists():
                                try:
                                    with open(txt_file_path, 'r', encoding='utf-8') as f:
                                        text_content = f.read().strip()
                                    self.text_descriptions[place_numerical_id] = text_content
                                    found_file = True
                                    break
                                except Exception as e:
                                    print(f"Cannot read text description file {txt_file_path}: {e}")
                        
                        # If no text file is found, use default description
                        if not found_file:
                            self.text_descriptions[place_numerical_id] = "Scene description"
                            print(f"Text description for scene {place_dir.name} not found, using default description")

                    for img_file in sorted(place_dir.iterdir()):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            current_places_data[place_numerical_id].append(str(img_file))
        
        # Filter places that do not have enough images for training
        for place_id, img_paths in current_places_data.items():
            if len(img_paths) >= self.min_img_per_place:
                self.places_data[place_id] = img_paths
                self.place_ids.append(place_id)
        
        self.place_ids.sort() # Ensure deterministic order
        # Print text description loading statistics
        if self.use_text:
            print(f"Training set: Loaded text descriptions for {len(self.text_descriptions)} scenes (Total {len(self.place_ids)} scenes)")

    def _load_eval_dataset(self):
        split_path = self.dataset_folder / self.split

        # Load database images
        database_path = split_path / "database"
        if database_path.exists():
            for place_dir in sorted(database_path.iterdir()):
                if place_dir.is_dir() and place_dir.name.startswith('@'):
                    place_numerical_id = self._get_place_numerical_id(place_dir.name)
                    if place_numerical_id is None:
                        continue
                    
                    # If text description is enabled, load it from the dedicated folder (with multiple filename fallbacks)
                    if self.use_text and str(place_dir) not in self.text_descriptions:
                        possible_filenames = [
                            f"{place_dir.name}.txt",
                            f"{place_dir.name.split('@')[0]}.txt",
                            f"{place_dir.name.replace('@', '_')}.txt",
                            f"{place_numerical_id}.txt",
                        ]
                        found_file = False
                        for filename in possible_filenames:
                            txt_file_path = self.text_folder / filename
                            if txt_file_path.exists():
                                try:
                                    with open(txt_file_path, 'r', encoding='utf-8') as f:
                                        text_content = f.read().strip()
                                    # Associate text description with scene ID (keyed by directory string)
                                    self.text_descriptions[str(place_dir)] = text_content
                                    found_file = True
                                    break
                                except Exception as e:
                                    print(f"Cannot read text description file {txt_file_path}: {e}")
                        if not found_file:
                            self.text_descriptions[str(place_dir)] = "Scene description"
                    
                    for img_file in sorted(place_dir.iterdir()):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            self._database_images_paths.append(str(img_file))
                            self._all_place_ids.append(place_numerical_id) # Add to combined list as well
                            
                            # If there is a text description for the scene, associate it with the image
                            if self.use_text and str(place_dir) in self.text_descriptions:
                                self.text_descriptions[str(img_file)] = self.text_descriptions[str(place_dir)]
                                
            self._database_num = len(self._database_images_paths)

        # Load queries images
        queries_path = split_path / "queries"
        if queries_path.exists():
            for place_dir in sorted(queries_path.iterdir()):
                if place_dir.is_dir() and place_dir.name.startswith('@'):
                    place_numerical_id = self._get_place_numerical_id(place_dir.name)
                    if place_numerical_id is None:
                        continue
                    
                    # If text description is enabled, load it from the dedicated folder (with multiple filename fallbacks)
                    if self.use_text and str(place_dir) not in self.text_descriptions:
                        possible_filenames = [
                            f"{place_dir.name}.txt",
                            f"{place_dir.name.split('@')[0]}.txt",
                            f"{place_dir.name.replace('@', '_')}.txt",
                            f"{place_numerical_id}.txt",
                        ]
                        found_file = False
                        for filename in possible_filenames:
                            txt_file_path = self.text_folder / filename
                            if txt_file_path.exists():
                                try:
                                    with open(txt_file_path, 'r', encoding='utf-8') as f:
                                        text_content = f.read().strip()
                                    self.text_descriptions[str(place_dir)] = text_content
                                    found_file = True
                                    break
                                except Exception as e:
                                    print(f"Cannot read text description file {txt_file_path}: {e}")
                        if not found_file:
                            self.text_descriptions[str(place_dir)] = "Scene description"
                    
                    for img_file in sorted(place_dir.iterdir()):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            self._query_images_paths.append(str(img_file))
                            self._all_place_ids.append(place_numerical_id) # Add to combined list as well
                            
                            # If there is a text description for the scene, associate it with the image
                            if self.use_text and str(place_dir) in self.text_descriptions:
                                self.text_descriptions[str(img_file)] = self.text_descriptions[str(place_dir)]
                                
            self._queries_num = len(self._query_images_paths)
        
        self.all_images_paths = self._database_images_paths + self._query_images_paths
        
        # Print text description loading statistics
        if self.use_text:
            print(f"Evaluation set: Loaded text descriptions for {len(self.text_descriptions)} images (Total {len(self.all_images_paths)} images)")

    def __getitem__(self, idx):
        if self.split == "train":
            place_id = self.place_ids[idx]
            img_paths = self.places_data[place_id]
            
            # Randomly sample img_per_place images from the current place
            # If there are fewer than img_per_place, just take all available
            if len(img_paths) > self.img_per_place:
                selected_img_paths = random.sample(img_paths, self.img_per_place)
            else:
                selected_img_paths = img_paths
            
            imgs = []
            for img_path in selected_img_paths:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            
            # If text description is enabled and the scene has a text description, return it
            if self.use_text and place_id in self.text_descriptions:
                text_desc = self.text_descriptions[place_id]
                # Return the same text description for each image
                text_descs = [text_desc] * len(selected_img_paths)
                return torch.stack(imgs), torch.tensor(place_id).repeat(len(selected_img_paths)), text_descs
            else:
                return torch.stack(imgs), torch.tensor(place_id).repeat(len(selected_img_paths))
        else: # val or test
            img_path = self.all_images_paths[idx]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            
            if self.use_text and img_path in self.text_descriptions:
                text_desc = self.text_descriptions[img_path]
                return img, idx, text_desc
            else:
                return img, idx

    def __len__(self):
        if self.split == "train":
            return len(self.place_ids) 
        else:
            return len(self.all_images_paths) 

    @property
    def database_num(self):
        if self.split == "train":
            raise AttributeError("database_num is not available in train split mode")
        return self._database_num

    @property
    def queries_num(self):
        if self.split == "train":
            raise AttributeError("queries_num is not available in train split mode")
        return self._queries_num
    
    def get_positives(self):
        if self.split == "train":
            raise AttributeError("get_positives is not available in train split mode")
        
        positives_per_query = []
        for i in range(self.queries_num):
            query_global_idx = self.database_num + i 
            query_place_id = self._all_place_ids[query_global_idx]
            
            positive_indices = []
            for db_idx in range(self.database_num):
                db_place_id = self._all_place_ids[db_idx]
                if query_place_id == db_place_id:
                    positive_indices.append(db_idx)
            positives_per_query.append(np.array(positive_indices)) 
        return positives_per_query

def get_EventVPR(dataset_folder, split="train", use_text=True, text_folder=None):
    return EventVPRDataset(dataset_folder, split=split, use_text=use_text, text_folder=text_folder)