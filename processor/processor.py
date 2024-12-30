import os

from tqdm import tqdm

import pandas as pd
import numpy as np
import h5py

import processor
import utils


class SeparationDataProcessor:
    def __init__(
        self,
        play_data: pd.DataFrame,
        player_data: pd.DataFrame,
        player_play_data: pd.DataFrame,
        data_dir: str,
    ):
        self.play_data = play_data
        self.player_data = player_data
        self.player_play_data = player_play_data
        self.data_dir = data_dir

        print("Ordinalizing Play and Player Data...", end=" ")
        self.player_data = processor.ordinalize_player_data(self.player_data)
        print("Finished!")

        self.cache_file_fname = os.path.join(data_dir, "cache_dataset.hdf5")

        # Setting sampling rate
        self.samp_rate = 10.0

        # Maximum time for a single play sequence recording
        self.max_seq_time = 100.0

        # Getting the max sequence length
        self.max_seq_len = int(self.samp_rate * self.max_seq_time)

    def process(
        self,
        weeks: list[int],
    ) -> tuple[
        list[list[pd.DataFrame]], list[pd.DataFrame], list[pd.DataFrame], list[float]
    ]:
        if not self.check_saved_cache():
            self.calc_features()
        seq, meta, player_play, label = self.load_data(weeks)
        return seq, meta, player_play, label

    def check_saved_cache(self) -> bool:
        if not os.path.exists(self.cache_file_fname):
            return False
        return True

    def load_data(
        self,
        weeks: list[int],
    ) -> tuple[
        list[list[pd.DataFrame]], list[pd.DataFrame], list[pd.DataFrame], list[float]
    ]:
        pass

    def calc_features(self) -> None:
        weeks_data = utils.load_weeks_data(self.data_dir)
        pass_plays = utils.get_pass_plays(self.player_play_data, self.play_data)
        play_ids = pass_plays[["gameId", "playId", "target_receiver_id"]]

        featurizer = processor.DataFeaturizer()
        feat_play_data = featurizer.featurize_play(self.play_data)

        for week_num in range(1, 9):
            seq_features = []
            meta_features = []
            player_play_features = []
            label = []

            week_data = weeks_data[week_num]
            week_data = featurizer.featurize_week(week_data)
            no_pass_arrival_count = 0
            no_line_set_count = 0

            # Sub-selecting only valid data for faster search later
            game_loc = week_data["gameId"].isin(play_ids["gameId"])
            play_loc = week_data["playId"].isin(play_ids["playId"])
            week_data = week_data[game_loc & play_loc]

            # Iterating throw plays
            n_rows = play_ids.shape[0]
            pbar = tqdm(
                play_ids.iterrows(),
                total=n_rows,
                desc=f"Processing Week {week_num}",
            )
            for idx, (gameId, playId, nflId) in pbar:
                # Getting all data from plays
                week_play_data = utils.get_data_play(week_data, gameId, playId)

                if week_play_data.shape[0] == 0:
                    continue

                if "line_set" not in week_play_data["event"].unique():
                    no_line_set_count += 1
                    continue

                # Getting metadata from play
                meta_play_data = feat_play_data[
                    (feat_play_data["gameId"] == gameId)
                    & (feat_play_data["playId"] == playId)
                ]

                play_player_ids = week_play_data["nflId"].dropna().unique()
                play_players = self.player_data[
                    self.player_data["nflId"].isin(play_player_ids)
                ]

                if meta_play_data["passTippedAtLine"].item():
                    continue

                # pass arrival time data
                pass_arrival_data = utils.get_pass_arrival_time_data(week_play_data)

                if pass_arrival_data.shape[0] == 0:
                    no_pass_arrival_count += 1
                    continue

                # Getting pre-snap data after lineset but before snap
                pre_snap_data = utils.get_pre_snap_data(week_play_data)

                # Getting receiver separation
                min_dist = utils.get_separation(pass_arrival_data, nflId)

                #  pre_snap_seq = utils.convert_to_sequence(pre_snap_data)
                seq_features.append(pre_snap_data)
                meta_features.append(meta_play_data)
                player_play_features.append(play_players)
                label.append(min_dist)

                #  if idx > 100:
                #      break

            print(f"No line set failures: {no_line_set_count}")
            print(f"No pass arrival count: {no_pass_arrival_count}")

            # Saving Data
            meta_features = pd.concat(meta_features, axis=0)
            meta_arr = processor.extract_meta_features(meta_features)
            player_play_arr = processor.extract_player_play_features(
                player_play_features,
            )
            seq_arr, seq_mask = processor.extract_seq_arr(
                seq_features,
                self.max_seq_len,
            )
            with h5py.File(self.cache_file_fname, "w") as f:
                f[f"week_{week_num}/separation_arr"] = label
                f[f"week_{week_num}/player_play_arr"] = player_play_arr
                f[f"week_{week_num}/meta_arr"] = meta_arr
                f[f"week_{week_num}/seq_arr"] = seq_arr
                f[f"week_{week_num}/seq_mask"] = seq_mask
                pass
