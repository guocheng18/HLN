import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def batch_padding(batch):
    seqs, labs = tuple(zip(*batch))
    seqs = pad_sequence(seqs, batch_first=True)
    masks = seqs.ne(0)
    labs = torch.cat(labs)
    return seqs, masks, labs


class Yoochoose(Dataset):
    """ Yoochoose dataset """
    def __init__(self, pkl_file, truncate_steps=None):
        super(Yoochoose, self).__init__()
        with open(pkl_file, "rb") as f:
            self.seqs, self.labs = pickle.load(f)
        self.ts = truncate_steps if truncate_steps is not None else 0

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return (torch.LongTensor(self.seqs[idx][-self.ts:]),
                torch.LongTensor([self.labs[idx]]))


class Lastfm(Dataset):
    """ Lastfm dataset """
    def __init__(self, pkl_file, truncate_steps=None):
        super(Lastfm, self).__init__()
        with open(pkl_file, "rb") as f:
            self.seqs, self.labs = pickle.load(f)
        self.ts = truncate_steps if truncate_steps is not None else 0

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return (torch.LongTensor(self.seqs[idx][-self.ts:]),
                torch.LongTensor([self.labs[idx]]))
