import os
import ast
import psutil
import humanize
import requests
import importlib
import numpy as np
import pandas as pd
import GPUtil as GPU
from typing import Any
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")


def printm():
    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " | Proc size: " + humanize.naturalsize(process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)


def send_message(msg, name, chat_id, bot_token):
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    data = {'chat_id': str(chat_id), 'text': f'{name}: {msg}'}
    requests.post(url, data)


def create_folds(df, path, name='train_folds.csv', k=5, output=True):
    """
    Take as dataframe with 'image_id' column and makes a .csv file,
    using stratified k-folds based on bbox_count
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    df_folds = df[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = df[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    df_folds.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    path = Path('.') if path is None else path
    df_folds.to_csv(path/name, index=True)
    if output:
        print(f'[FOLDS CREATED] path: {path/name}')


def combine_csv(trn, sub, path, name='train_ext.csv', output=True):
    """
    Combines train.csv file and submission.csv file
    params:
    ------
    trn, sub, path : path object/str
    """
    path = Path('.') if path is None else path
    trn_df = pd.read_csv(trn)
    sub_df = pd.read_csv(sub)

    ext_df = None
    for id, scores in sub_df.values:
        scores = scores.split()
        scores = list(map(float, scores))

        boxes = []
        for i in range(0, len(scores), 5):
            boxes.append(str(scores[i+1: i+5]))

        tmp = pd.DataFrame(boxes, columns=['bbox'])
        tmp['image_id'] = id

        if ext_df is None:
            ext_df = tmp.copy()
        else:
            ext_df = pd.concat([ext_df, tmp], axis=0)

    ext_df['source'] = 'ankur'
    ext_df['width'] = 1024
    ext_df['height'] = 1024

    ext_df = ext_df[list(trn_df.columns)]
    ext_df = pd.concat([trn_df, ext_df], axis=0)
    ext_df.to_csv(path/name, index=False)
    if output:
        print(f'[COMBINED CSV] {trn} + {sub} -> {path/name}')


def make_pseudo_labels(trn, sub, path=None, k=5):
    path = Path('.') if path is None else path

    # combining train and submission files
    combine_csv(trn, sub, path, name='train_ext.csv', output=False)

    # reading df
    df = pd.read_csv(path/'train_ext.csv')
    bboxs = np.stack(df['bbox'].apply(lambda x: ast.literal_eval(x)))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df[column] = bboxs[:, i]
    df.drop(columns=['bbox'], inplace=True)

    # Making folds form the new csv
    create_folds(df, path, name='train_folds.csv', output=False)
    print(f'[FOLDS]   path: {path/"train_folds.csv"} \n[MARKING] path: {path/"train_ext.csv"}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--trn', type=str, default='data/train.csv', help='train.csv path')
    parser.add_argument('--sub', type=str, default='submission_best.csv', help='submission.csv path')
    opt = parser.parse_args()

    make_pseudo_labels(opt.trn, opt.sub)
