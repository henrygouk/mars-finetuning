from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence, to_categorical
from keras_preprocessing.image.utils import load_img, img_to_array
import numpy as np
import os
import random

class ImageDataSequence(Sequence):
    def __init__(self, directories, batch_size, target_size, frac=1.0, **kwargs):
        if not isinstance(directories, list):
            directories = [directories]

        self.batch_size = batch_size
        self.target_size = target_size
        self.classes = sorted(os.listdir(directories[0]))
        self.num_classes = len(self.classes)
        self.filenames = []
        self.labels = []
        self.transformer = ImageDataGenerator(**kwargs)

        for d in range(0, len(directories)):
            for idx, cls in zip(range(len(self.classes)), self.classes):
                filenames = os.listdir(os.path.join(directories[d], cls))

                for f in filenames:
                    self.filenames.append(os.path.join(directories[d], cls, f))
                    self.labels.append(idx)

        self.on_epoch_end()

        num_instances = int(frac * len(self.filenames))
        self.filenames = self.filenames[0:num_instances]
        self.labels = self.labels[0:num_instances]

    def __getitem__(self, batch_idx):
        X_batch = np.zeros((self.batch_size,) + self.target_size)
        y_batch = np.zeros((self.batch_size, len(self.classes)))
        filenames = []
        labels = []
        aux = []
        current_index = batch_idx * self.batch_size

        for i in range(self.batch_size):
            filenames.append(self.filenames[current_index])
            labels.append(self.labels[current_index])

            if hasattr(self, "auxiliary"):
                aux.append(self.auxiliary[current_index])

            current_index += 1

        color_mode = "rgb" if self.target_size[2] == 3 else "grayscale"
        target_size = (self.target_size[0], self.target_size[1])
        X_batch = np.array([self.transformer.random_transform(img_to_array(load_img(fn, color_mode=color_mode, target_size=target_size))) for fn in filenames])
        y_batch = to_categorical(labels, num_classes=len(self.classes))

        if hasattr(self, "auxiliary"):
            return X_batch, [y_batch, aux]
        else:
            return X_batch, y_batch


    def __len__(self):
        return int(len(self.filenames) / self.batch_size)

    def on_epoch_end(self):
        X = self.filenames
        y = self.labels

        if hasattr(self, "auxiliary"):
            aux = self.auxiliary
            insts = list(zip(X, y, aux))
            random.shuffle(insts)
            X, y, aux = zip(*insts)
            self.auxiliary = aux
        else:
            insts = list(zip(X, y))
            random.shuffle(insts)
            X, y = zip(*insts)

        self.filenames = X
        self.labels = y

