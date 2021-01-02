import os
import numpy as np
import unicodedata
import string
import re
import torch
from torch.utils import data
from torch.utils.data import DataLoader

def readText(root, mode):
    path = os.path.join(root, mode + '.txt')
    
    with open(path, 'r') as reader:
        data_ = np.loadtxt(reader, dtype=np.str).reshape(-1)

    return data_

class TextDataloader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.data = readText(self.root, self.mode)
        self.chardict = CharDict()
        self.input_tenses = np.array([0, 0, 0, 0, 3, 0, 3, 2, 2, 2])
        self.targets_tenses = np.array([3, 2, 1, 1, 1, 2, 0, 0, 3, 1])

    # Return the length of the word
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'train':
            condition = index % 4
            return self.chardict.StringToLongtensor(self.data[index]), condition
        else:
            input_condition = self.input_tenses[index%10]
            target_condition = self.targets_tenses[index%10]
            return self.chardict.StringToLongtensor(self.data[index*2%20]), self.chardict.StringToLongtensor(self.data[(index*2+1)%20]), input_condition, target_condition


class CharDict:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0
        
        for i in range(26):
            self.addWord(chr(ord('a') + i))

        tokens = ["SOS", "EOS"]
        for t in tokens:
            self.addWord(t)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def StringToLongtensor(self, s):
        s = ["SOS"] + list(s) + ["EOS"]
        return torch.LongTensor([self.word2index[char] for char in s])

    def LongtensorToString(self, l, show_token=False, check_end=True):
        s = ""
        for i in l:
            ch = self.index2word[i.item()]
            if len(ch) > 1:
                if show_token:
                    __ch = "<{}>".format(ch)
                else:
                    __ch = ""
            else:
                __ch = ch
            s += __ch
            if check_end and ch == "EOS":
                break
        return s
