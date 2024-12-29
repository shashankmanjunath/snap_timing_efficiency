import os

from fire import Fire
from tqdm import tqdm

import pandas as pd

import utils

# Separation Timing Efficiency


def main(data_dir: str):
    data = utils.load_data(data_dir)
    train_weeks = [1]
    train_dataset = Dataset(weeks=train_weeks, data=data, data_dir=data_dir)


def separation_features_labels(
    play_ids: pd.DataFrame,
    play_data: pd.DataFrame,
    player_data: pd.DataFrame,
    weeks_data: list[pd.DataFrame],
) -> tuple[
    list[list[pd.DataFrame]], list[pd.DataFrame], list[pd.DataFrame], list[float]
]:

    count = 0
    seq_features = []
    meta_features = []
    player_play_features = []
    label = []
    for week_data in weeks_data:
        # Sub-selecting only valid data for faster search later
        game_loc = week_data["gameId"].isin(play_ids["gameId"])
        play_loc = week_data["playId"].isin(play_ids["playId"])
        week_data = week_data[game_loc & play_loc]

        # Iterating throw plays
        n_rows = play_ids.shape[0]
        for idx, (gameId, playId, nflId) in tqdm(play_ids.iterrows(), total=n_rows):
            # Getting all data from plays
            week_play_data = utils.get_data_play(week_data, gameId, playId)

            # Getting metadata from play
            meta_play_data = play_data[
                (play_data["gameId"] == gameId) & (play_data["playId"] == playId)
            ]

            play_player_ids = week_play_data["nflId"].dropna().unique()
            play_players = player_data[player_data["nflId"].isin(play_player_ids)]

            if meta_play_data["passTippedAtLine"].item():
                continue

            pre_snap_data = week_play_data[
                week_play_data["frameType"] == "BEFORE_SNAP"
            ].sort_values(by="frameId")

            # pass arrival time data
            pass_arrival_data = utils.get_pass_arrival_time_data(week_play_data)

            if pass_arrival_data.shape[0] == 0:
                count += 1
                continue

            min_dist = utils.get_separation(pass_arrival_data, nflId)

            pre_snap_seq = utils.convert_to_sequence(pre_snap_data)
            seq_features.append(pre_snap_seq)
            meta_features.append(meta_play_data)
            player_play_features.append(play_players)
            label.append(min_dist)
            pass
        pass
    return seq_features, meta_features, player_play_features, label


class Dataset:
    def __init__(
        self,
        weeks: list[int],
        data: dict[str, pd.DataFrame],
        data_dir: str,
    ):
        self.weeks_fnames = []
        self.weeks_data = []

        for week in weeks:
            week_fname = f"tracking_week_{week}.csv"
            week_path = os.path.join(data_dir, week_fname)
            self.weeks_fnames.append(week_path)
            self.weeks_data.append(pd.read_csv(week_path))

        pass_plays = utils.get_pass_plays(data)
        play_ids = pass_plays[["gameId", "playId", "target_receiver_id"]]
        seq_features, meta_features, player_features, sep = separation_features_labels(
            play_ids,
            data["play"],
            data["players"],
            self.weeks_data,
        )

        self.seq_features = seq_features
        self.meta_features = meta_features
        self.player_features = player_features
        self.sep = sep


if __name__ == "__main__":
    Fire(main)
