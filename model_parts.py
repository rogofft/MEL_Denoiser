import torch
from torch.utils.data import Dataset


class MelDataset(Dataset):
    """ Dataset to store [N, 80] samples of mel-spectrogram """
    def __init__(self, data, transform=False):
        self.data = data
        self.length = len(data)
        self.transform = transform

    def __getitem__(self, idx):
        data, target = self.data[idx][0], self.data[idx][1]
        if self.transform:
            data, target = torch.from_numpy(data).view(1, -1, 80), torch.from_numpy(target).view(1, -1, 80)
        return data, target

    def __len__(self):
        return self.length


def init_weights(m):
    """ Init weights for model layer filters """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        try:
            torch.nn.init.normal_(m.weight, mean=0., std=1.)
        except BaseException:
            pass


class EarlyStopDetector:
    def __init__(self, max_steps=5, reverse=False):
        self.reverse = reverse
        self.step = 0
        self.max_steps = max_steps

        if not self.reverse:
            # bigger value is better
            self.best_score = float('-inf')
        else:
            # smaller value is better
            self.best_score = float('inf')

    def check_for_best_score(self, score):
        if ((not self.reverse) and score > self.best_score) or (self.reverse and score < self.best_score):
            self.step = 0
            self.best_score = score
            result = True
        else:
            self.step += 1
            result = False

        return result

    def check_for_stop(self):
        return True if self.step >= self.max_steps else False
