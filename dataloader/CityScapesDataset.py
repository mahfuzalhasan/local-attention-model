



from torch.utils.data as data


class CityScapesDataset(data.Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.imgs = []
        self.labels = []

        self._set_files()