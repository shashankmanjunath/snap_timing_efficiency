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
        data_dir: str,
        force_proc: bool = False,
        save: bool = True,
    ):
        self.data_dir = data_dir
        self.force_proc = force_proc
        self.save = save

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.check_saved_cache():
            self.calc_features()
        seq, seq_mask, meta, label = self.load_data(weeks)
        seq_mask = seq_mask.astype(bool)
        return seq, seq_mask, meta, label

    def check_saved_cache(self) -> bool:
        if self.force_proc:
            return False

        if not os.path.exists(self.cache_file_fname):
            return False
        return True

    def load_data(
        self,
        weeks: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        seq_arr = []
        seq_mask_arr = []
        meta_arr = []
        label_arr = []
        with h5py.File(self.cache_file_fname, "r") as f:
            for week in weeks:
                week_key = f"week_{week}"
                seq_arr.append(f[week_key]["seq_arr"])
                seq_mask_arr.append(f[week_key]["seq_mask"])
                meta_arr.append(f[week_key]["meta_arr"])
                label_arr.append(f[week_key]["separation_arr"])

            seq_arr = np.concatenate(seq_arr, axis=0)
            seq_mask_arr = np.concatenate(seq_mask_arr, axis=0)
            meta_arr = np.concatenate(meta_arr, axis=0)
            label_arr = np.concatenate(label_arr, axis=0)
        return seq_arr, seq_mask_arr, meta_arr, label_arr

    def calc_features(self) -> None:
        data = utils.load_data(self.data_dir)
        play_data = data["play"]
        player_data = data["players"]
        player_play_data = data["player_play"]

        print("Ordinalizing Play and Player Data...", end=" ")
        player_data = processor.ordinalize_player_data(player_data)
        print("Finished!")

        weeks_data = utils.load_weeks_data(self.data_dir)
        pass_plays = utils.get_pass_plays(player_play_data, play_data)
        play_ids = pass_plays[["gameId", "playId", "target_receiver_id"]]

        featurizer = processor.DataFeaturizer()
        feat_play_data = featurizer.featurize_play(play_data)
        feat_player_play_data = featurizer.featurize_player_play(player_play_data)

        for week_num in range(1, 10):
            seq_features = []
            meta_features = []
            label = []

            week_data = weeks_data[week_num]
            week_data = featurizer.featurize_week(week_data)
            no_pass_arrival_count = 0
            no_line_set_count = 0
            before_snap_mismatch = 0

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

                # Getting time-series sequence data from plays
                week_play_data = utils.get_play_sequence(week_data, gameId, playId)

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
                play_players = player_data[player_data["nflId"].isin(play_player_ids)]

                if meta_play_data["passTippedAtLine"].item():
                    continue

                # pass arrival time data
                pass_arrival_data = utils.get_pass_arrival_time_data(week_play_data)

                if pass_arrival_data.shape[0] == 0:
                    no_pass_arrival_count += 1
                    continue

                # Getting pre-snap data after lineset but before snap
                pre_snap_data = utils.get_pre_snap_data(week_play_data)
                if pre_snap_data.shape[0] == 0:
                    before_snap_mismatch += 1
                    continue

                # Getting receiver separation
                min_dist = utils.get_separation(pass_arrival_data, nflId)

                # Getting specific player data from the play
                play_overall_data = utils.get_player_play_data(
                    player_play_data,
                    week_play_data,
                    gameId,
                    playId,
                )

                # Merging player_play features and sequence features to ensure
                # ordering
                play_players = play_players.rename(
                    columns={"displayName": "displayName_ord"}
                )
                pre_snap_data = pre_snap_data.merge(
                    play_players,
                    how="outer",
                    on="nflId",
                )
                pre_snap_data = pre_snap_data.merge(
                    play_overall_data,
                    how="outer",
                    on="nflId",
                )
                #  pre_snap_data = pre_snap_data.sort_values(by=["frameId", "position"])
                # FEATURES:
                #  1. features that account for quarterback arm strength
                #  2. the receiver’s separation at the time the QB targeted them
                #  3. he horizontal and vertical position of the receiver on the field at the time of
                #  the throw,
                #  4. where the receiver lined up pre-snap
                #  5. the distance to the goal line
                #  6. the amount of break in the receiver’s route during the football’s journey through the air after it was
                #  released
                #  7. the depth of the QB’s drop, the number of other routes that were being run on the play
                #  8. if the play was a play-action pass or a screen
                #  9. and the number of deep safeties.

                seq_features.append(pre_snap_data)
                meta_features.append(meta_play_data)
                label.append(min_dist)
                # TODO: Add some features about route type, depth, target
                # receiver/read, etc.?
                #  if idx > 100:
                #      break

            print(f"No line set failures: {no_line_set_count}")
            print(f"No pass arrival count: {no_pass_arrival_count}")
            print(f"Mismatched pre-snap labeling count: {before_snap_mismatch}")

            # Saving Data
            print(f"Converting week {week_num} data to array...")
            meta_features = pd.concat(meta_features, axis=0)
            meta_arr = processor.extract_meta_features(meta_features)
            seq_arr, seq_mask = processor.extract_seq_arr(
                seq_features,
                self.max_seq_len,
            )
            print("Done!")
            if self.save:
                print(f"Saving Week {week_num}...")
                with h5py.File(self.cache_file_fname, "a") as f:
                    f[f"week_{week_num}/separation_arr"] = label
                    f[f"week_{week_num}/meta_arr"] = meta_arr
                    f[f"week_{week_num}/seq_arr"] = seq_arr
                    f[f"week_{week_num}/seq_mask"] = seq_mask
                print("Data saved!")
