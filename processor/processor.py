import typing
import time
import os

from tqdm import tqdm

import pandas as pd
import numpy as np
import h5py

import processor
import xgb_model
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

    def process(self, weeks: list[int]) -> tuple[np.ndarray, np.ndarray]:
        if not self.check_saved_cache():
            self.calc_features()
        X, y = self.load_data(weeks)
        return X, y

    def check_saved_cache(self) -> bool:
        if self.force_proc:
            return False

        if not os.path.exists(self.cache_file_fname):
            return False
        return True

    def load_data(
        self,
        weeks: list[int],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        #  seq_arr = []
        #  seq_mask_arr = []
        #  meta_arr = []
        #  play_players_arr = []
        #  play_overall_arr = []
        X = []
        label_arr = []
        t1 = time.time()
        with h5py.File(self.cache_file_fname, "r") as f:
            for week in weeks:
                week_key = f"week_{week}"
                feat_arr_week = xgb_model.create_feature_arr(f[week_key])
                X.append(feat_arr_week)
                label_arr.append(f[week_key]["separation_arr"][()])

        X = pd.concat(X, axis=0)
        y = np.concatenate(label_arr, axis=0)
        t_load = time.time() - t1
        print(f"Cache load time: {t_load:.3f}")
        return X, y

    def calc_features(self) -> None:
        data = utils.load_data(self.data_dir)
        play_data = data["play"]
        player_data = data["players"]
        player_play_data = data["player_play"]

        weeks_data = utils.load_weeks_data(self.data_dir)
        pass_plays = utils.get_pass_plays(player_play_data, play_data)
        #  play_ids = pass_plays[["gameId", "playId", "target_receiver_id"]]

        featurizer = processor.DataFeaturizer()
        feat_play_data = featurizer.featurize_play(play_data)
        feat_player_play_data = featurizer.featurize_player_play(player_play_data)
        player_data = featurizer.featurize_player_data(player_data)

        for week_num in range(1, 10):
            #  if week_num > 3:
            #      continue
            play_players_features = []
            play_overall_features = []
            seq_features = []
            meta_features = []
            label = []

            week_data = weeks_data[week_num]
            week_data = featurizer.featurize_week(week_data)
            no_pass_arrival_count = 0
            no_line_set_count = 0
            before_snap_mismatch = 0

            # Sub-selecting only valid data for faster search later
            game_loc = week_data["gameId"].isin(pass_plays["gameId"])
            play_loc = week_data["playId"].isin(pass_plays["playId"])
            week_data = week_data[game_loc & play_loc]

            # Iterating throw plays
            n_rows = pass_plays.shape[0]
            pbar = tqdm(
                pass_plays.iterrows(),
                total=n_rows,
                desc=f"Processing Week {week_num}",
            )
            for idx, row in pbar:
                gameId = row["gameId"]
                playId = row["playId"]
                nflId = row["target_receiver_id"]

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
                meta_play_data = meta_play_data.copy()

                play_player_ids = week_play_data["nflId"].dropna().unique()
                play_players = player_data[player_data["nflId"].isin(play_player_ids)]
                play_players = play_players.copy()

                n = play_players.shape[0]
                play_players.loc[:, "wasTargettedReceiver"] = np.zeros(n)
                id_loc = play_players["nflId"] == nflId
                play_players.loc[id_loc, "wasTargettedReceiver"] = 1
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
                    feat_player_play_data,
                    week_play_data,
                    gameId,
                    playId,
                )

                num_routes = play_overall_data["wasRunningRoute"].sum().item()
                meta_play_data["numRoutes"] = num_routes
                pass

                # FEATURES:
                #  1. Features that account for quarterback arm strength
                #       DIFFICULT
                #  2. The receiver’s separation at the time the QB targeted them
                #       DO NOT WANT FOR THIS ANALYSIS
                #  3. The horizontal and vertical position of the receiver on the
                #     field at the time of the throw,
                #       DO NOT WANT FOR THIS ANALYSIS
                #  4. Where the receiver lined up pre-snap
                #       TODO: ADD -- DIFFICULT
                #  5. The distance to the goal line
                #       INCLUDED
                #  6. The amount of break in the receiver’s route during the
                #     football’s journey through the air after it was released
                #       ADDED
                #  7. The depth of the QB’s drop
                #       INCLUDED
                #  8. The number of other routes that were being run on the play
                #       ADDED
                #  9. If the play was a play-action pass or a screen
                #       TODO: Add if screen -- DIFFICULT
                #  10. The number of deep safeties.
                #       INCLUDED in coverage type
                #  11. Other route bypes being run
                #       ADDED

                play_players_features.append(play_players)
                play_overall_features.append(play_overall_data)
                seq_features.append(pre_snap_data)
                meta_features.append(meta_play_data)
                label.append(min_dist)
                #  if idx > 100:
                #      break

            print(f"No line set failures: {no_line_set_count}")
            print(f"No pass arrival count: {no_pass_arrival_count}")
            print(f"Mismatched pre-snap labeling count: {before_snap_mismatch}")

            # Saving Data
            print(f"Converting week {week_num} data to array...")
            t1 = time.time()
            meta_features = pd.concat(meta_features, axis=0)
            meta_arr = processor.extract_meta_features(meta_features)
            meta_cols = processor.get_full_meta_feature_cols()

            seq_arr, seq_mask = processor.extract_seq_arr(
                seq_features,
                self.max_seq_len,
            )
            seq_cols = processor.get_seq_feature_columns()

            play_players_arr = processor.extract_play_players_features(
                play_players_features,
            )
            play_players_cols = processor.get_play_players_columns()

            play_overall_arr = processor.extract_play_overall_features(
                play_overall_features,
            )
            play_overall_cols = processor.get_play_overall_columns()
            label = np.asarray(label)
            t_featurize = time.time() - t1
            print(f"Done! Runtime: {t_featurize:.3f}")

            if self.save:
                print(f"Saving Week {week_num}...")
                t2 = time.time()
                with h5py.File(self.cache_file_fname, "a") as f:
                    f[f"week_{week_num}/separation_arr"] = label
                    f[f"week_{week_num}/meta_arr"] = meta_arr.astype(np.float32)
                    f[f"week_{week_num}/meta_cols"] = meta_cols
                    f[f"week_{week_num}/seq_arr"] = seq_arr.astype(np.float32)
                    f[f"week_{week_num}/seq_mask"] = seq_mask.astype(bool)
                    f[f"week_{week_num}/seq_cols"] = seq_cols
                    f[f"week_{week_num}/play_players_arr"] = play_players_arr.astype(
                        np.float32
                    )
                    f[f"week_{week_num}/play_players_cols"] = play_players_cols
                    f[f"week_{week_num}/play_overall_arr"] = play_overall_arr.astype(
                        np.float32
                    )
                    f[f"week_{week_num}/play_overall_cols"] = play_overall_cols
                t_save = time.time() - t2
                print(f"Data saved! Saving time: {t_save:.3f}")
