import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.X = []
        self.y = []
        if train:
            for i in range(1, 6):
                with open(os.path.join(base_folder, f'data_batch_{i}'), 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    self.X.append(batch[b'data'])
                    self.y.extend(batch[b'labels'])
            self.X = np.concatenate(self.X, axis=0)
        else:
            with open(os.path.join(base_folder, 'test_batch'), 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.X = batch[b'data']
                self.y = batch[b'labels']
        self.X = self.X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.y = np.array(self.y, dtype=np.int64)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.X[index], self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
