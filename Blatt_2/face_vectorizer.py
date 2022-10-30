import os
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.util import crop


def _standardize_image(img: np.ndarray):
    img = crop(img, 60)
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


if __name__ == "__main__":
    a, b, c, d = get_images()
    print("Hello")
