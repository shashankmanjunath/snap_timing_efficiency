from fire import Fire
from tqdm import tqdm

import sklearn.metrics
import pandas as pd
import numpy as np

#  import torch
import h5py

import xgb_model
import processor
import models

# Separation Timing Efficiency


def main(data_dir: str, use_wandb: bool = False):
    train_weeks = [1]
    train_dataset = Dataset(weeks=train_weeks, data_dir=data_dir)

    train_dataset = Dataset(train_weeks, data_dir)

    network = models.TransformerTimeSeries(
        embed_dim=16,
        n_heads=8,
        n_blocks=4,
        n_categorical=8,
    )
    trainer = models.TransformerTrainer(
        network,
        train_dataset,
        # TODO: Repeating train dataset
        train_dataset,
        use_wandb=use_wandb,
    )
    trainer.train()


def create_feature_arr(f: h5py.File) -> dict[str, np.ndarray]:
    n = f["seq_arr"].shape[0]
    # Removing everything except position
    target_seq_cols = [
        "gameId",
        "nflId",
        "playId",
        "frameId",
        "x",
        "y",
        "s",
        "a",
        "dis",
        "o",
        "dir",
    ]

    play_overall_arr = []
    for idx in tqdm(range(n), desc="Processing data into array..."):
        # Extracting final positions of players
        play_players_df = pd.DataFrame(
            f["play_players_arr"][idx, :, :],
            columns=xgb_model.decode(f["play_players_cols"]),
        )
        play_players_df["nflId"] = play_players_df["nflId"].astype(int)
        play_overall_df = pd.DataFrame(
            f["play_overall_arr"][idx, :, :],
            columns=xgb_model.decode(f["play_overall_cols"]),
        )
        play_overall_df["nflId"] = play_overall_df["nflId"].astype(int)
        play_overall_df = play_overall_df.merge(
            play_players_df,
            on=["nflId"],
            how="outer",
        )

        seq_cols = xgb_model.decode(f["seq_cols"])
        iter_pos_arr = []
        for x in f["seq_arr"][idx, :, :, :]:
            df = pd.DataFrame(x, columns=seq_cols)
            df = df[target_seq_cols].dropna(axis=0)
            df["nflId"] = df["nflId"].astype(int)
            df = df.sort_values("nflId")
            df = df.drop(["gameId", "nflId", "playId", "frameId"], axis=1)
            iter_pos_arr.append(df)

        play_overall_df = play_overall_df.sort_values("nflId")
        play_overall_df = play_overall_df.drop(
            [
                "gameId",
                "playId",
                "nflId",
                "inMotionAtBallSnap",
                "shiftSinceLineset",
                "motionSinceLineset",
            ],
            axis=1,
        )
        play_overall_arr.append(play_overall_df)

    meta_df = pd.DataFrame(
        f["meta_arr"][()],
        columns=xgb_model.decode(f["meta_cols"]),
    )
    meta_df = meta_df.drop(["gameId", "playId"], axis=1)
    pass
    output_dict = {
        "pos_arr": pos_arr,
        "mask_arr": f["seq_mask"][()],
        "meta_arr": np.nan_to_num(meta_df.to_numpy()),
        "play_overall_arr": np.stack(
            [np.nan_to_num(x.to_numpy()) for x in play_overall_arr]
        ),
    }
    return output_dict


class Dataset:
    def __init__(
        self,
        weeks: list[int],
        data_dir: str,
    ):

        proc = processor.SeparationDataProcessor(data_dir)
        data_dict, sep = proc.process_sequence(weeks)

        self.seq_features = torch.as_tensor(data_dict["pos_arr"])
        self.seq_mask = torch.as_tensor(data_dict["mask_arr"])
        self.play_overall_features = torch.as_tensor(data_dict["play_overall_arr"])
        self.meta_features = torch.as_tensor(data_dict["meta_arr"])
        self.sep = torch.as_tensor(sep)

        self.get_baseline_acc()

    def __len__(self):
        return self.seq_features.shape[0]

    def get_baseline_acc(self):
        mean_val = self.sep.mean()
        arr = torch.Tensor(self.sep.shape).fill_(mean_val)
        test_acc = sklearn.metrics.mean_absolute_error(
            self.sep.cpu().numpy().squeeze(),
            arr.cpu().numpy(),
        )
        print(f"Baseline Accuracy: {test_acc:.3f}")
        pass

    def __getitem__(self, idx):
        # Removing football location
        feat = self.seq_features[idx, :, :-1, :]
        mask = ~self.seq_mask[idx, :, 0, 0]
        meta_feat = self.meta_features[idx, :]
        play_overall_feat = self.play_overall_features[idx, :]
        label = self.sep[idx]
        return feat, mask, meta_feat, label


if __name__ == "__main__":
    Fire(main)
