import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as f:
            f.read(4)
            num_images = struct.unpack(">I", f.read(4))[0]
            rows = struct.unpack(">I", f.read(4))[0]
            cols = struct.unpack(">I", f.read(4))[0]
            X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols, 1)
            X = X.astype(np.float32) / 255.0

        with gzip.open(label_filename, "rb") as f:
            f.read(8)
            y = np.frombuffer(f.read(), dtype=np.uint8)

        self.X, self.y = X, y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img, label = self.X[index], self.y[index]
        return self.apply_transforms(img), label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION