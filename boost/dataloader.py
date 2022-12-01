#opt/ml/data에서 불러오기
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import catboost as ctb

import numpy as np
import random



# 데이터 로드 함수(train, test) from directory
def get_data(args):
    train_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'train_data.csv')) # train + test(not -1)
    test_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'test_data.csv')) # test
    
    cate_cols = [col for col in train_data.columns if col[-2:]== '_c']

    test = test_data[test_data.answerCode == -1]   # test last sequnece
    
    #테스트의 정답 컬럼을 제거
    test = test.drop('answerCode', axis=1)
    train = train_data
    # print('cate_cols:', cate_cols)
    return cate_cols, train, test


# 데이터 스플릿 함수
def data_split(train_data, ratio):
    X = train_data.drop(['answerCode'], axis=1)
    y = train_data['answerCode']

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=ratio, # 일단 이 정도로 학습해서 추이 확인
        shuffle=True,
    )
    return X_train, X_valid, y_train, y_valid


def time_loader(args):
    train = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'train_data.csv'))
    test = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'test_data.csv'))
    valid = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'valid_data.csv'))

    cate_cols = [col for col in train.columns if col[-2:]== '_c']

    test = test[test.answerCode == -1]
    test = test.drop('answerCode', axis=1)
    
    return cate_cols, train, test, valid

def time_shuffle(train, valid):
    group = (train.groupby("userID").apply(lambda r: [r[name] for name in train.columns]))

    col = train.columns.tolist()
    col.pop(3)

    X = [[] for _ in range(len(col))]
    Y = []

    # shuffle data grouped by users
    by_user = group.values
    random.shuffle(by_user)

    # realign shuffled train data to pd.DataFrame 
    for user in by_user:
        Y.extend(user.pop(3))

        for idx,feat in enumerate(user):
            X[idx].extend(feat)  

    train_X = pd.DataFrame({name:values for name,values in zip(col, X)})
    train_Y = pd.DataFrame({'answerCode': Y})

    valid_X = valid.drop('answerCode', axis=1)
    valid_Y = valid['answerCode']

    return train_X, valid_X, train_Y, valid_Y
