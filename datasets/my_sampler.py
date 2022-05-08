import torch
import random
from torch.utils.data import Sampler
import sys
sys.path.append("..")
from engine import global_value


class MyBalancedSampler(Sampler):
    def __init__(self, data_source, batch=16, partial=20, args=None, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.batch = batch
        self.partial = partial
        self.args = args

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source) # rsna=4809
        box_n = int(n * self.partial / 100) # 20% = 961


        batch = self.batch

        box_idx = torch.randperm(box_n).tolist() 
        point_idx = random.sample(range(box_n, n), box_n)

        point_idx = torch.tensor(point_idx).split(box_n//batch)[:batch]
        box_idx = torch.tensor(box_idx).split(box_n//batch)[:batch]

        # print(point_idx)
        # print(box_idx)

        final_list = []
        for box, point in zip(box_idx, point_idx):
            box, point = box.tolist(), point.tolist()
            cur = box + point
            random.shuffle(cur)
            final_list += cur
        print('epoch imgs:', len(final_list))
        # print(sorted(final_list))
        return iter(final_list)


    def __len__(self):
        return self.num_samples
