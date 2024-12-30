import os

from fire import Fire
from tqdm import tqdm

import pandas as pd
import numpy as np
import h5py

import processor
import utils

# Separation Timing Efficiency


def main(data_dir: str):
    data = utils.load_data(data_dir)
    train_weeks = [1]
    train_dataset = Dataset(weeks=train_weeks, data=data, data_dir=data_dir)


class Dataset:
    def __init__(
        self,
        weeks: list[int],
        data: dict[str, pd.DataFrame],
        data_dir: str,
    ):

        proc = processor.SeparationDataProcessor(
            data["play"],
            data["players"],
            data["player_play"],
            data_dir,
        )
        seq_features, meta_features, player_features, sep = proc.process(weeks)

        self.seq_features = seq_features
        self.meta_features = meta_features
        self.player_features = player_features
        self.sep = sep


if __name__ == "__main__":
    Fire(main)
