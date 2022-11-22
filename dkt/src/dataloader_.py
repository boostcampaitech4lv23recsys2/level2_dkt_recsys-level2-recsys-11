import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

def get_data(args, is_train: bool, leakage: bool=False):
    if is_train:
        data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'train_data.csv'))
        offset = data.iloc[-1, 0].item()
        cate_num = data.iloc[-1, 1].item()
        cont_num = data.iloc[-1, 2].item()
        data = data.iloc[:-1]
        return data.groupby('userID').apply(
        lambda df: tuple([df[col] for col in df.columns if col != 'userID'])
        ), offset, cate_num, cont_num
    else:
        data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'test_data.csv'))
        return data.groupby('userID').apply(
        lambda df: tuple([df[col] for col in df.columns if col != 'userID'])
        )


def split_data(data:pd.Series, ratio=0.9, shuffle=True):
    if shuffle:
        data = data.sample(frac=1)

    size = int(len(data) * ratio)
    data_1 = data.iloc[:size]
    data_2 = data.iloc[size:]

    return data_1, data_2

class DKTDataset(Dataset):
    def __init__(self, data, args):
        # 현재 데이터는 userID 가 인덱스인데, 
        self.args = args
        self.data = data
        self.id2idx = {v:k for k, v in enumerate(self.data.index.unique())} 
        self.data.index = self.data.index.map(self.id2idx) 
        # userID 를 인덱싱함. 어차피 유저 아이디는 피처로 안들어가긴 함.
        # 판다스 인덱싱 접근 편의성 위해서.
        # 문제 자체가 어떤 유저가 들어왔을 때 임으로 userID는 일종의 인덱스역할임.
        
    def __getitem__(self, index):
        seq_len = len(self.data[index][0])
        cate_cols = [col.values.copy() + 1 for col in self.data[index] if col.name[-2:] == '_c' and \
            col.name != 'answerCode'] # 시계열 패딩 용 + 1. 이제 unknown 은 1 이 됨. 시계열 부족이 0
            
        cont_cols = [col.values.copy() for col in self.data[index] if col.name[-2:] != '_c' and \
            col.name != 'answerCode'] # cont 는 시계열 패딩을 어떻게..? # 마지막 최근 시계열 값으로 앞 시계열 채우겠습니다. 일단은.
        # print(cate_cols)
        answer = [col.values.copy() for col in self.data[index] if col.name == 'answerCode'][0].astype(np.float32)
        
        if seq_len >= self.args.max_seq_len:
            if cate_cols:
                for i, col in enumerate(cate_cols):
                    cate_cols[i] = col[-self.args.max_seq_len:]
            if cont_cols: # cont feature 가 없을 수도 있음.
                for i, col in enumerate(cont_cols):
                    cont_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16) # Byte Tensor 를 씀
            answer = answer[-self.args.max_seq_len:]
        else:
            if cate_cols:
                for i in range(len(cate_cols)):
                    new_col = np.zeros(self.args.max_seq_len, np.int64)
                    new_col[-seq_len:] = cate_cols[i]
                    cate_cols[i] = new_col
            if cont_cols:
                for i in range(len(cont_cols)):
                    new_col = np.zeros(self.args.max_seq_len, dtype=np.float32)
                    # new_col[:seq_len] = cont_cols[i][0] # cont 의 시계열이 부족하면 가장 앞 데이터로 채움.
                    new_col[-seq_len:] = cont_cols[i]
                    cont_cols[i] = new_col
            mask = np.zeros(self.args.max_seq_len, dtype=np.int64)
            mask[-seq_len:] = 1
            new_answer = np.zeros(self.args.max_seq_len, dtype=np.float32)
            new_answer[-seq_len:] = answer
            answer = new_answer

        cate_cols = np.array(cate_cols).T.astype(np.int64)
        cont_cols = np.array(cont_cols).T.astype(np.float32)

        # return cate_cols, cont_cols, mask, target
        return cate_cols, cont_cols, mask, answer

    def __len__(self):
        return len(self.data)


def get_loader(args, train, valid):

    train_dataloader, valid_dataloader = None, None

    if train is not None:
        train_dataset = DKTDataset(train, args)
        train_dataloader = DataLoader(
            train_dataset,
            num_workers=4,
            shuffle=True,
            batch_size=args.batch_size,
        )

    if valid is not None:
        valid_dataset = DKTDataset(valid, args)
        valid_dataloader = DataLoader(
            valid_dataset,
            num_workers=4,
            shuffle=False,
            batch_size=args.batch_size,
        )

    return train_dataloader, valid_dataloader