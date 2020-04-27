from torch.utils import data
from torchvision import datasets, transforms

class MNISTDataLoader(data.DataLoader):

    def __init__(self, root, batch_size, train=True, distributed=False):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        dataset = datasets.FashionMNIST(root, train=train, transform=transform, download=True)
        
        sampler = None
        if train and distributed:
            sampler = data.DistributedSampler(dataset)

        super(MNISTDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
        )