
from config import *

class RandomApply(nn.Module):
    def __init__(self, prob, f1, f2=lambda x: x):
        super().__init__()
        self.f1 = f1
        self.f2 = f2
        self.prob = prob

    def forward(self, x):
        f1 = self.f1 if random.random() < self.prob else self.f2
        return f1(x)


class HistologyDataset(Dataset):
    def __init__(self, df):

        samples = []

        self.filenames = list(df['path'])
        self.labels = []
        for idx, row in df.iterrows():
            img, img_labels, cl_new_label = row['path'], \
                                            [row['cell_crowding_level'],
                                            row['cell_polarity_level'],
                                            row['mitosis_level'],
                                            row['nucleoli_level'],
                                            row['pleomorphism_level']], \
                                            row['cl_labels']

            samples.append((img, img_labels, cl_new_label))
            self.labels.append(img_labels)

        self.samples = samples

        self.transform = transforms.Compose([
            transforms.Resize(512),
            RandomApply(0.25, transforms.RandomResizedCrop(512, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                        transforms.CenterCrop(512)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.6422, 0.4224, 0.5756], std=[0.1978, 0.2395, 0.1842])
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_path, label, cl_label = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.ANTIALIAS)
        img = self.transform(img)
        return img_path, img, label, cl_label


class ImbalancedDatasetSampler(Sampler):

    def __init__(self, dataset, attr, indices=None, num_samples=None, callback_get_label=None):

        self.attr = attr
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        self.callback_get_label = callback_get_label

        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        print('attr: ', attr, Counter(label_to_count))
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.labels[idx][self.attr]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

