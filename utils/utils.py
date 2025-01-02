import time
import os

from tqdm import tqdm

import pandas as pd
import numpy as np


def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    print("Loading data from disk....", end=" ")
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
    print(f"Data Loaded! Load time: {load_time:.3f} seconds")
    return data


def get_pass_plays(player_play: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    plays = plays[plays["passResult"].isin(["C", "I"])].copy()

    target_receiver_ids = []
    pbar = tqdm(
        plays.iterrows(),
        total=plays.shape[0],
        desc="Loading pass plays",
    )
    for idx, play in pbar:
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


def get_player_play_data(
    player_data: pd.DataFrame,
    play_seq_data: pd.DataFrame,
    gameId: str,
    playId: str,
) -> pd.DataFrame:
    game_loc = player_data["gameId"] == gameId
    play_loc = player_data["playId"] == playId
    play_player_data = player_data[game_loc & play_loc]

    after_snap = play_seq_data[play_seq_data["frameType"] == "AFTER_SNAP"]
    pass_arrival_frame = get_pass_arrival_time_data(play_seq_data)["frameId"].unique()
    pass_arrival_frame = pass_arrival_frame.min().item()

    post_snap = after_snap[after_snap["frameId"] <= pass_arrival_frame]
    post_snap = post_snap.sort_values(by="frameId")

    event_1 = play_seq_data["event"] == "ball_snap"
    event_2 = play_seq_data["event"] == "snap_direct"
    event_3 = play_seq_data["event"] == "autoevent_ballsnap"
    snap_data = play_seq_data[event_1 | event_2 | event_3]
    ball_data = snap_data[snap_data["displayName"] == "football"]
    if ball_data.shape[0] > 1:
        ball_pos = ball_data["y"].iloc[0].item()
    else:
        ball_pos = ball_data["y"].item()

    player_ids = post_snap["nflId"].dropna().unique()
    max_dist = [
        (post_snap[post_snap["nflId"] == player_id]["y"] - ball_pos).max()
        for player_id in player_ids
    ]

    # TODO: get route depth data
    #  route_runners = route_data[route_data["wasRunningRoute"] == 1.0]
    #  runner_seq = play_seq_data[play_seq_data["nflId"].isin(route_runners["nflId"])]
    #  runner_seq = runner_seq.sort_values(by=["frameId"])
    #  start_pos = runner_seq["event"]
    dist_df = pd.DataFrame({"nflId": player_ids, "maxDist": max_dist})
    route_data = play_player_data.merge(dist_df, how="outer", on="nflId")
    return route_data


def get_pre_snap_data(play_data: pd.DataFrame) -> pd.DataFrame:
    pre_snap_data = play_data[play_data["frameType"] == "BEFORE_SNAP"]
    pre_snap_data = pre_snap_data.sort_values(by="frameId")
    lineset_data = pre_snap_data[pre_snap_data["event"] == "line_set"]

    #  lineset_frame = lineset_data["frameId"].unique()[0]
    lineset_frame = lineset_data["frameId"].unique().tolist()
    if len(lineset_frame) >= 1:
        # Sometimes there are multiple line sets recorded. In this case, we choose
        # the first frame as the official line set
        lineset_frame = lineset_frame[0]
    elif len(lineset_frame) == 0:
        # No line sets recorded, we do not want to keep the data
        return pd.DataFrame()
    pre_snap_data = pre_snap_data[pre_snap_data["frameId"] >= lineset_frame]
    return pre_snap_data


def get_play_sequence(plays_df: pd.DataFrame, gameId: str, playId: str) -> pd.DataFrame:
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
    if receiver_data.shape[0] > 1:
        receiver_data = receiver_data.iloc[0].to_frame().T
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
    dist = np.sqrt(((def_pos - rec_pos.astype(float)) ** 2).sum(-1))
    min_dist = dist.min()
    return min_dist


def convert_to_sequence(play_data_df: pd.DataFrame) -> list[pd.DataFrame]:
    frame_list = play_data_df["frameId"].unique()
    frame_list.sort()

    seq = []
    for f in frame_list:
        seq.append(play_data_df[play_data_df["frameId"] == f])
    return seq


def load_weeks_data(data_dir: str) -> dict[int, pd.DataFrame]:
    weeks_data = {}

    for week in tqdm(range(1, 10), desc="Loading Weeks data"):
        #  if week > 3:
        #      continue
        week_fname = f"tracking_week_{week}.csv"
        week_path = os.path.join(data_dir, week_fname)
        weeks_data[week] = pd.read_csv(week_path)
    return weeks_data
