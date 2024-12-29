import os

import pandas as pd
import numpy as np


def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
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
    return data


def get_in_motion_at_snap_plays(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Getting plays with motion
    plays = data["player_play"][data["player_play"]["inMotionAtBallSnap"] == True]

    # Finding plays where the motion player was the targeted receiver
    motion_receiver_target = plays[plays["wasTargettedReceiver"] == True]
    return motion_receiver_target


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
