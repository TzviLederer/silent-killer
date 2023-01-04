import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import Dataset
import cv2


class PoisonedSubset(Dataset):
    def __init__(self, clean_dataset, indexes, norm, eps, clip=(0., 1.)):
        self.clean_dataset = clean_dataset
        self.indexes = indexes
        self.poison = torch.zeros((len(indexes), *clean_dataset[0][0].shape), requires_grad=True)

        self.clip = clip
        self.norm = norm
        self.eps = eps

    def __getitem__(self, item):
        x, y = self.clean_dataset[self.indexes[item]]
        x = x + self.poison[item]
        if self.clip is not None:
            if isinstance(x, torch.Tensor):
                x = x.clip(*self.clip)
            elif isinstance(x, (np.ndarray, np.generic)):
                x = np.clip(x, *self.clip)
            else:
                raise NotImplementedError
        return x, y

    def __len__(self):
        return len(self.indexes)

    def normalize_poison(self):
        poison = self.poison.detach()

        if self.norm == 'l_inf':
            poison = poison.clip(-self.eps, self.eps)
        elif self.norm == 'l2':
            norm = poison.norm(2, dim=(1, 2, 3), keepdim=True)
            poison[norm.flatten() > self.eps] = poison[norm.flatten() > self.eps] / norm[
                norm.flatten() > self.eps] * self.eps

        # self.poison = Settings.Transform.transform_tensor(self.poison)
        poison.requires_grad_()
        self.poison = poison


class PoisonedDataset(Dataset):
    """
    This is the poisoned dataset containing the whole clean data with the poisoned data
    """

    def __init__(self, clean_dataset, eps, norm, indexes, enabled_indexes=None):
        """
        enabled indexes - for evaluating filtering defences
        """
        self.clean_dataset = clean_dataset
        self.indexes = indexes
        self.poison_subset = PoisonedSubset(clean_dataset=clean_dataset, indexes=self.indexes, norm=norm, eps=eps)

        self.eps = eps
        self.norm = norm

        self.enabled_indexes = enabled_indexes

    def __getitem__(self, item):
        if self.enabled_indexes is not None:
            item = self.enabled_indexes[item]

        if item in self.indexes:
            i = self.indexes.index(item)
            return self.poison_subset[i]
        return self.clean_dataset[item]

    def __len__(self):
        if self.enabled_indexes is not None:
            return len(self.enabled_indexes)
        return len(self.clean_dataset)

    def normalize_poison(self):
        self.poison_subset.normalize_poison()


class TriggeredDataset(Dataset):
    """
    This dataset contains data with the trigger.
    It filters out the target labeled samples and keep only the samples with different class.
    """

    def __init__(self, clean_dataset, source_label, target_label, trigger_fn):
        self.clean_dataset = clean_dataset
        self.target_label = target_label
        self.source_label = source_label
        self.trigger_fn = trigger_fn

        self.labels = {i: x[1] for i, x in enumerate(clean_dataset)}
        self.source_label_samples = dict(filter(lambda x: x[1] == source_label, self.labels.items()))
        self.samples_list = list(self.source_label_samples.keys())

        self.label_to_return = 'both'  # options: 'both', 'true', 'target'
        print('Trigger dataset returns both true label and target label. '
              'To change it, change the `label_to_return` variable.')

        print('Creating triggered dataset:')
        print(f'source label: {self.clean_dataset.classes[source_label]} ({source_label}), '
              f'target label: {self.clean_dataset.classes[target_label]} ({target_label})')

    def __getitem__(self, item):
        x, y = self.clean_dataset[self.samples_list[item]]
        x = self.trigger_fn(x)
        if self.label_to_return == 'both':
            return x, y, self.target_label
        elif self.label_to_return == 'true':
            return x, y
        elif self.label_to_return == 'target':
            return x, self.target_label

    def __len__(self):
        return len(self.samples_list)


class TriggerAdditive:
    def __init__(self, size, method='randn', sigma=0.1, clip_lim=(0, 1), clip_trigger=16/255, **kwargs):
        self.clip_lim = clip_lim
        self.clip_trigger = clip_trigger

        if method == 'randn':
            self.trigger = torch.randn(size=size) * sigma
        else:
            raise NotImplementedError

        self.trigger.requires_grad_()

    def inject(self, sample):
        sample = sample + self.trigger.clip(-self.clip_trigger, self.clip_trigger)
        sample = torch.clip(sample, self.clip_lim[0], self.clip_lim[1])
        return sample

    def update_trigger(self, trigger):
        self.trigger = trigger
        self.trigger.requires_grad_()


class TriggerAdaptivePatch:
    def __init__(self, method='randn', sigma=0.01, clip_lim=(0, 1), patch_size=(3, 11, 11),
                 patch_path=None, trigger_loc='rand', trigger=None, **kwargs):
        """
        :param trigger_loc: `rand` or tuple of location (y, x)
        """
        self.clip_lim = clip_lim
        self.patch_size = patch_size
        self.trigger_loc = trigger_loc
        self.patch_size = patch_size

        if method == 'randn':
            self.trigger = torch.randn(size=patch_size) * sigma + 0.5
        elif method == 'from_file' and patch_path is not None:
            im_bgr = cv2.imread(patch_path)
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im_rgb, patch_size[1:3]) / 255
            self.trigger = torch.tensor(im.transpose(2, 0, 1))
        elif method == 'from_tensor':
            if trigger is None:
                raise FileNotFoundError('if `from_tensor` is chosen, must give an argument `trigger`')
            self.trigger = trigger
        else:
            raise NotImplementedError

        self.trigger.requires_grad_()

    def update_trigger(self, trigger):
        self.trigger = trigger
        self.trigger.requires_grad_()

    def inject(self, sample):
        if (np.array(sample.shape) - np.array(self.patch_size)).min() < 0:
            raise ValueError('the patch size cannot be larger than the image')
        if self.trigger_loc == 'rand':
            _, y, x = sample.shape
            x = np.random.randint(0, x - self.patch_size[2] + 1)
            y = np.random.randint(0, y - self.patch_size[1] + 1)
        else:
            y, x = self.trigger_loc

        sample[:, y:y + self.patch_size[1], x:x + self.patch_size[2]] = self.trigger
        sample = torch.clip(sample, self.clip_lim[0], self.clip_lim[1])
        return sample
