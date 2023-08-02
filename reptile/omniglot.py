import os.path
import random
import shutil
import zipfile
from pathlib import Path
from typing import Optional, NoReturn, Tuple, Union, Literal

import torch
from PIL import Image
from six.moves import urllib
from torch.utils.data import Subset, Dataset
from torchvision import transforms

from tqdm import tqdm


class Omniglot(Dataset):
    folder = "omniglot"
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]

    def __init__(self, root: str,
                 download: bool = False
                 ) -> NoReturn:

        self.root = Path(root)
        self.target_folder = self.root / self.folder
        self.mode = 'train'
        self._cache = {}
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self._alphabets = [d for d in self.target_folder.iterdir() if d.is_dir()]
        self._characters = sum(([c for c in a.iterdir()] for a in self._alphabets), [])
        self._character_images = [
            [(image, idx) for image in character.iterdir()]
            for idx, character in enumerate(self._characters)
        ]
        self._flat_character_images = sum(self._character_images, [])

    def __len__(self) -> int:
        return len(self._characters)

    def _download(self) -> NoReturn:
        if self._check_exists():
            print("Files already downloaded and verified")
            return

        for url in self.urls:
            download_file_with_progress(url, self.target_folder)

        unzip_and_delete(self.target_folder)

    def _check_exists(self) -> bool:
        return self.target_folder.exists() and len(list(self.target_folder.iterdir())) != 0

    @staticmethod
    def transform(image, target_size=28, rotation_angle=0):
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomRotation(rotation_angle),
            transforms.ToTensor(),
        ])(image)

    def sample(self, classes, num_shots):
        data = []
        for idx, name in enumerate(classes):
            image_list = self._character_images[name]
            images = self._read_images(image_list, idx, num_shots)
            data += images
        return MiniDataset(data)

    def set_mode(self, mode: Union[Literal['train'], Literal['test']]):
        self.mode = mode

    def _read_images(self, images, idx, num_shots):
        random.shuffle(images)
        rotation = random.choice([0, 90, 180, 270]) if self.mode == 'train' else 0
        res = []
        for image in images[:num_shots]:
            if image[0] in self._cache:
                img = self.transform(self._cache[image[0]], rotation_angle=rotation)
                res.append((img, idx))
                continue
            img = Image.open(image[0], mode="r").convert("L")
            self._cache[image[0]] = img
            img = self.transform(img, rotation_angle=rotation)
            res.append((img, idx))
        return res


class OmniglotSubset(Subset):
    def __init__(self, dataset: Omniglot, mode, indices):
        super().__init__(dataset, indices)
        self.dataset: Omniglot = dataset
        self.indices = indices
        self.mode = mode

    def sample(self, classes, num_shots):
        self.dataset.set_mode(self.mode)
        return self.dataset.sample(classes, num_shots)


class MiniDataset(Dataset):
    def __init__(self, data):
        super(MiniDataset, self).__init__()
        self.data = data
        self.index = 0

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration


class TqdmUpTo(tqdm):
    last_block = 0

    def update_to(self, block_num: int = 1, block_size: int = 1, total_size: Optional[int] = None) -> None:
        if total_size is not None:
            self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_file_with_progress(url: str, target_folder: Path) -> None:
    filename = os.path.basename(url)
    target_folder.mkdir(parents=True, exist_ok=True)
    file_path = target_folder / filename
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=file_path, reporthook=t.update_to)


def unzip_and_delete(path: Path) -> None:
    zip_files = path.glob("*.zip")

    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(path)
        unzip_folder = path / zip_file.stem
        for item in unzip_folder.iterdir():
            item.rename(path / item.name)
        shutil.rmtree(unzip_folder)
        zip_file.unlink()


def split_dataset(dataset: Omniglot, train_set_size: int = 1200) -> Tuple[OmniglotSubset, OmniglotSubset]:
    indices = torch.randperm(len(dataset))
    train_set = OmniglotSubset(dataset, 'train', indices[:train_set_size])
    test_set = OmniglotSubset(dataset, 'test', indices[train_set_size:])
    return train_set, test_set
