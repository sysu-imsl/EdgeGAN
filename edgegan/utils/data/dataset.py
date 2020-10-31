import os
from glob import glob
from pathlib import Path

import numpy as np

from edgegan.utils import get_image


def extension_match_recursive(root, exts):
    result = []
    for ext in exts:
        paths = [str(p) for p in Path(root).rglob(ext)]
        result.extend(paths)
    return result


class Dataset():
    def __init__(self, dataroot, name, size, batchsize, config, num_classes=None, phase='train'):
        assert phase in ['train', 'test', ]
        self.batchsize = batchsize
        self.num_classes = num_classes
        self.config = config
        self.phase = phase
        if phase == 'train':
            if num_classes is not None:
                self.data = []
                for i in range(num_classes):
                    for ext in ['*.png', '*.jpg']:
                        data_path = os.path.join(
                            dataroot, name, phase, str(i), ext)
                        self.data.extend(glob(data_path))
            else:
                data_path = os.path.join(
                    dataroot, name, phase, '*.png')
                self.data = glob(data_path)
        else:
            data_path = os.path.join(dataroot, name, phase)
            self.data = extension_match_recursive(
                data_path,
                ['*.png', '*.jpg']
            )
            self.data = sorted(self.data)

        if len(self.data) == 0:
            raise Exception("[!] No data found in '" + data_path + "'")
        if len(self.data) < self.batchsize:
            raise Exception(
                "[!] Entire dataset size is less than the configured batch_size")
        self.size = min(len(self.data), size)

    def shuffle(self):
        np.random.shuffle(self.data)

    def __len__(self):
        return self.size // self.batchsize

    def __getitem__(self, idx):
        filenames = self.data[idx * self.batchsize: (idx+1)*self.batchsize]
        batch = [
            get_image(filename,
                      input_height=self.config['input_height'],
                      input_width=self.config['input_width'],
                      resize_height=self.config['output_height'],
                      resize_width=self.config['output_width'],
                      crop=self.config['crop'],
                      grayscale=self.config['grayscale']) for filename in filenames]

        batch_images = np.array(batch).astype(np.float32)

        if self.phase == 'train':
            batch_z = np.random.normal(
                size=(self.batchsize, self.config['z_dim']))

            if self.num_classes is not None:
                def get_class(filePath):
                    end = filePath.rfind("/")
                    start = filePath.rfind("/", 0, end)
                    return int(filePath[start+1:end])
                batch_classes = [get_class(batch_file)
                                 for batch_file in filenames]
                batch_classes = np.array(
                    batch_classes).reshape((self.batchsize, 1))
                batch_z = np.concatenate((batch_z, batch_classes), axis=1)

        if self.phase == 'test':
            assert batch_images.shape[0] == len(filenames)

        return (batch_images, batch_z, filenames) if self.phase == 'train' else (batch_images, filenames)
