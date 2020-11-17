import os

import pandas as pd

from config import EXPERIMENTS_INDEX_FILE

INDEX_PATH = EXPERIMENTS_INDEX_FILE
NAME_COL = 'Name'

index_df = None


def save_df():
    index_df.to_csv(INDEX_PATH)


if os.path.exists(INDEX_PATH):
    index_df = pd.read_csv(INDEX_PATH, index_col=0)
else:
    index_df = pd.DataFrame([{NAME_COL: 'Null', 'Explanation': 'To make data frame not empty'}])
    save_df()


def save_record(name, record):
    record[NAME_COL] = name
    record_df = pd.DataFrame([record])

    global index_df
    index_df = index_df.append(record_df, sort=False, ignore_index=True)

    save_df()


def get_index_df():
    return index_df
