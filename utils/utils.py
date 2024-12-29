import time
import os

from tqdm import tqdm

import pandas as pd
import numpy as np


def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    print("Loading data from disk....")
    t1 = time.time()
    games_fname = os.path.join(data_dir, "games.csv")
    play_fname = os.path.join(data_dir, "plays.csv")
    players_fname = os.path.join(data_dir, "players.csv")
    player_play_fname = os.path.join(data_dir, "player_play.csv")

    data = {
        "games": pd.read_csv(games_fname),
        "play": pd.read_csv(play_fname),
        "players": pd.read_csv(players_fname),
        "player_play": pd.read_csv(player_play_fname),
    }
    load_time = time.time() - t1
    print(f"Data Loaded! Load time: {load_time:.3f}")
    return data


def get_pass_plays(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    player_play = data["player_play"]
    plays = data["play"][data["play"]["passResult"].isin(["C", "I"])].copy()

    target_receiver_ids = []
    for idx, play in tqdm(plays.iterrows(), total=plays.shape[0]):
        gameId = play["gameId"]
        playId = play["playId"]

        game_loc = player_play["gameId"] == gameId
        play_loc = player_play["playId"] == playId
        play_data = player_play[game_loc & play_loc]
        rec_nfl_id = play_data[play_data["wasTargettedReceiver"] == 1]["nflId"]

        if rec_nfl_id.size == 0:
            target_receiver_ids.append(np.nan)
        else:
            target_receiver_ids.append(rec_nfl_id.item())

    plays.loc[:, "target_receiver_id"] = target_receiver_ids
    plays = plays[~plays["target_receiver_id"].isna()]
    return plays


def get_data_play(plays_df: pd.DataFrame, gameId: str, playId: str) -> pd.DataFrame:
    iter_game_loc = plays_df["gameId"] == gameId
    iter_play_loc = plays_df["playId"] == playId
    week_play_data = plays_df[iter_game_loc & iter_play_loc]
    return week_play_data


def get_pass_arrival_time_data(play_data_df: pd.DataFrame) -> pd.DataFrame:
    caught_pos = play_data_df["event"] == "pass_outcome_caught"
    incomplete_pos = play_data_df["event"] == "pass_outcome_incomplete"
    td_pos = play_data_df["event"] == "pass_outcome_touchdown"
    pass_arrival_data = play_data_df[caught_pos | incomplete_pos | td_pos]
    return pass_arrival_data


def get_separation(pass_arrival_data: pd.DataFrame, nflId: str) -> float:
    receiver_data = pass_arrival_data[pass_arrival_data["nflId"] == nflId]
    off_str = receiver_data["club"]
    football_data = pass_arrival_data[pass_arrival_data["displayName"] == "football"]
    pass_arrival_data = pass_arrival_data.drop(
        index=receiver_data.index,
        axis=1,
    )
    pass_arrival_data = pass_arrival_data.drop(
        index=football_data.index,
        axis=1,
    )
    def_data = pass_arrival_data[pass_arrival_data["club"] != off_str.item()]
    def_pos = def_data[["x", "y"]].to_numpy()
    rec_pos = receiver_data[["x", "y"]].to_numpy()
    dist = np.sqrt(((def_pos - rec_pos) ** 2).sum(-1))
    min_dist = dist.min()
    return min_dist


def convert_to_sequence(play_data_df: pd.DataFrame) -> list[pd.DataFrame]:
    frame_list = play_data_df["frameId"].unique()
    frame_list.sort()

    seq = []
    for f in frame_list:
        seq.append(play_data_df[play_data_df["frameId"] == f])
    return seq
