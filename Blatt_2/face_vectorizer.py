import os
from pathlib import Path
from typing import Union

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.util import crop


def _standardize_image(img: np.ndarray):
    img = crop(img, (60, 60))
    img = resize(img, (32, 32))
    img = np.resize(img, 32 * 32)
    return img


def _read_images(img_dir: Path):
    imgs = []
    for img in img_dir.iterdir():
        if not img.is_file():
            continue
        imgs.append(
            _standardize_image(
                imread(img, as_gray=True)))

    return imgs


def _split_images(images: list[np.ndarray]):
    random_sample = np.random.randint(len(images))
    random_sample = 1
    train_images = images[:-random_sample]
    test_image = images[random_sample]

    return train_images, test_image


def _get_paths_with_70_pictures():
    persons = []
    # Get all Directory of all persons with n_images > 70
    for item in Path("./data/faces_in_the_wild/lfw_funneled/").iterdir():
        if item.is_dir():
            if len(os.listdir(item)) >= 70:
                persons.append(item)
    return persons


def get_images():
    persons = _get_paths_with_70_pictures()
    train_images, test_images, train_labels, test_labels = [], [], [], []
    for person in persons:
        imgs = _read_images(person)
        train, test = _split_images(imgs)

        train_images.extend(train)
        test_images.append(test)

        train_labels.extend([person.stem] * len(train))
        test_labels.append(person.stem)

    return np.vstack(train_images), np.vstack(test_images), np.vstack(train_labels), np.vstack(test_labels)


class FaceVectorizer():
    def __init__(self, path: Union[str, Path], min_images=70, crop_size=(60, 60), new_size=32):
        if isinstance(path, str):
            self._path = Path(path)
        else:
            self._path = path
        self._min_images = min_images
        self._crop_size = crop_size
        self._new_size = new_size

    def _standardize_image(self, img: np.ndarray):
        img = crop(img, self._crop_size)
        img = resize(img, (self._new_size, self._new_size))
        img = np.resize(img, self._new_size * self._new_size)
        return img

    @staticmethod
    def _split_images(images: list[np.ndarray]):
        random_sample = 1
        train_images = images[:-random_sample]
        test_image = images[random_sample]

        return train_images, test_image

    def _read_images(self, person):
        images = []
        # image_dir = self._path / person

        for img in person.iterdir():
            if not img.is_file():
                continue
            images.append(
                _standardize_image(
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
            test_images.append(test)

            train_labels.extend([person.stem] * len(train))
            test_labels.append(person.stem)

        return np.vstack(train_images), np.vstack(test_images), np.vstack(train_labels), np.vstack(test_labels)
