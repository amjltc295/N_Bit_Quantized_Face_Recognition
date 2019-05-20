from torchvision import datasets, transforms
from base import BaseDataLoader


class LFWDataLoader(BaseDataLoader):
    """
    Face data loader

    Preprocessing (alignment) is assumed done. All images should be aligned and cropped to 112 * 112.
    The datadir should be like:

        datadir/
            <id_1>/
                1.jpg
                2.jpg
                ...
            <id_2>/
                1.jpg
                ...
            ...
    """

    def __init__(
            self,
            data_dir,
            batch_size,
            shuffle=True,
            validation_split=0.0,
            num_workers=4,
            training=True,
            **kwargs):
        if training:
            self.transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
