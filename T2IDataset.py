import os
import io
from paddle.io import Dataset, DataLoader
import h5py
import numpy as np
import pdb
from PIL import Image
import paddle
from paddle.vision import transforms


# load the dataset
class Text2ImageDataset(Dataset):
    def __init__(self, datasetFile, transform=None, split=0):
        # initialize the dataset
        super().__init__()
        self.datasetFile = datasetFile
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.dataset = None
        self.dataset_keys = None
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.h5py2int = lambda x: int(np.array(x))

    def __len__(self):
        # get the total number of data
        f = h5py.File(self.datasetFile, 'r')
        self.dataset_keys = [str(k) for k in f[self.split].keys()]
        length = len(f[self.split])
        f.close()

        return length

    def __getitem__(self, idx):
        """
        :return：
        a dictionary of data
        sample = {
            'right_images': real_image,
            'right_embed': sentence_embeddings,
            'wrong_images': wrong_image,
            'inter_embed': interpolated_embeddings,
            'txt': sentences
        }
        """
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]

        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        right_image = bytes(np.array(example['img']))
        right_embed = np.array(example['embeddings'], dtype=float)
        wrong_image = bytes(np.array(self.find_wrong_image(example['class'])))
        inter_embed = np.array(self.find_inter_embed())

        right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
        wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64, 64))

        txt = np.array(example['txt']).astype(str)

        right_image = self.transform(right_image)
        wrong_image = self.transform(wrong_image)

        sample = {
            'right_images': right_image,
            'right_embed': paddle.to_tensor(right_embed, dtype='float32'),
            'wrong_images': wrong_image,
            'inter_embed': paddle.to_tensor(inter_embed, dtype='float32'),
            'txt': str(txt)
        }

        return sample

    def find_wrong_image(self, category):
        # randomly select the wrong images for sentence embeddings
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)

    def find_inter_embed(self):
        # get the interpolated sentence embeddings
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return example['embeddings']
