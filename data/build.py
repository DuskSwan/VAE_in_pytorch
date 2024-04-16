# encoding: utf-8

import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, root, img_shape=(64, 64)) -> None:
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path).convert('RGB')
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)


def make_data_loader(root='data/datasets/celebA', **kwargs):
    dataset = CelebADataset(root, **kwargs)
    return DataLoader(dataset, 16, shuffle=True)

if __name__ == '__main__':
    import torchvision
    print(torchvision.__version__)
    dataloader = make_data_loader()
    img = next(iter(dataloader))
    print(img.shape)
    # Concat 4x4 images
    N, C, H, W = img.shape
    assert N == 16
    img = torch.permute(img, (1, 0, 2, 3)) # C, N, H, W
    img = torch.reshape(img, (C, 4, 4 * H, W)) # C, 4, 4H, W
    img = torch.permute(img, (0, 2, 1, 3)) # C, 4H, 4, W
    img = torch.reshape(img, (C, 4 * H, 4 * W)) # C, 4H, 4W
    img = transforms.ToPILImage()(img)
    img.save('tmp.jpg')