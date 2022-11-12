import os
from pathlib import Path
from typing import Union

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.util import crop
from sklearn.model_selection import train_test_split


class FaceVectorizer():
    def __init__(self, path: Union[str, Path], min_images=70, crop_size=(60, 60), new_size=32,
                 test_size: int | float = 1, random_seed=42):
        if isinstance(path, str):
            self._path = Path(path)
        else:
            self._path = path
        self._min_images = min_images
        self._crop_size = crop_size
        self._new_size = new_size
        self._test_size = test_size
        self._random_seed = random_seed

    def _standardize_image(self, img: np.ndarray):
        img = crop(img, self._crop_size)
        img = resize(img, (self._new_size, self._new_size))
        img = np.resize(img, self._new_size * self._new_size)
        return img

    def _split_images(self, images: list[np.ndarray]):

        train_images, test_images = train_test_split(images, test_size=self._test_size, random_state=self._random_seed)

        return train_images, test_images

    def _read_images(self, person):
        images = []
        # image_dir = self._path / person

        for img in person.iterdir():
            if not img.is_file():
                continue
            images.append(
                self._standardize_image(
                    imread(img, as_gray=True)))
        return images

    def _get_paths_with_n_pictures(self):
        persons = []
        # Get all Directory of all persons with n_images > 70
        for item in Path(self._path).iterdir():
            if item.is_dir():
                if len(os.listdir(item)) >= self._min_images:
                    persons.append(item)
        return persons

    def get_images(self):
        persons = self._get_paths_with_n_pictures()
        train_images, test_images, train_labels, test_labels = [], [], [], []
        for person in persons:
            images = self._read_images(person)
            train, test = self._split_images(images)

            train_images.extend(train)
            test_images.extend(test)

            train_labels.extend([person.stem] * len(train))
            test_labels.extend([person.stem] * len(test))

        return np.vstack(train_images), np.vstack(test_images), np.vstack(train_labels), np.vstack(test_labels)
