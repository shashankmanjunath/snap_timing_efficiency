import dateutil.parser
import time
import os

from tqdm import tqdm

import pandas as pd
import numpy as np


def get_route_cols():
    cols = [
        "routeRan_ANGLE",
        "routeRan_CORNER",
        "routeRan_CROSS",
        "routeRan_FLAT",
        "routeRan_GO",
        "routeRan_HITCH",
        "routeRan_IN",
        "routeRan_OUT",
        "routeRan_POST",
        "routeRan_SCREEN",
        "routeRan_SLANT",
        "routeRan_WHEEL",
    ]
    return cols


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

    # Calculating max distance from line of scrimmage as a measure of route
    # depth
    snap_data = play_seq_data[play_seq_data["frameType"] == "SNAP"]
    ball_data = snap_data[snap_data["displayName"] == "football"]
    if ball_data.shape[0] > 1:
        ball_pos = ball_data["x"].iloc[0].item()
    else:
        ball_pos = ball_data["x"].item()

    player_ids = post_snap["nflId"].dropna().unique()
    max_dist = [
        (post_snap[post_snap["nflId"] == player_id]["x"] - ball_pos).max()
        for player_id in player_ids
    ]

    dist_df = pd.DataFrame({"nflId": player_ids, "maxDist": max_dist})
    route_data = play_player_data.merge(dist_df, how="outer", on="nflId")

    # Calculating lateral distance changed as a measure of break in the
    # receiver's route
    pass_event_1 = after_snap["event"] == "pass_forward"
    pass_event_2 = after_snap["event"] == "pass_shovel"
    pass_time_data = after_snap[pass_event_1 | pass_event_2]
    if pass_time_data.shape[0] != 23:
        raise RuntimeError()

    catch_time_data = after_snap[after_snap["frameId"] == pass_arrival_frame]
    pass_time_data = pass_time_data[pass_time_data["displayName"] != "football"]
    catch_time_data = catch_time_data[catch_time_data["displayName"] != "football"]
    player_break = get_player_break(pass_time_data, catch_time_data)
    route_data["break"] = player_break

    # Adding routes run by players aside from the current one
    route_cols = get_route_cols()
    other_route_cols = ["other_" + x for x in route_cols]
    route_data[other_route_cols] = np.zeros(
        (
            route_data.shape[0],
            len(other_route_cols),
        )
    )

    for row_idx, _ in route_data.iterrows():
        no_row_data = route_data.drop(row_idx)
        other_route_counts = no_row_data[route_cols].sum(axis=0).tolist()
        route_data.loc[row_idx, other_route_cols] = other_route_counts

    return route_data


def get_player_break(pass_df: pd.DataFrame, rec_df: pd.DataFrame) -> np.ndarray:
    a = pass_df["a"].to_numpy()
    s = pass_df["s"].to_numpy()

    pass_rawtime = dateutil.parser.parse(pass_df["time"].iloc[0])
    rec_rawtime = dateutil.parser.parse(rec_df["time"].iloc[0])
    t = (rec_rawtime - pass_rawtime).total_seconds()
    pos_diff = (0.5 * a * (t**2)) + (s * t)
    theta = 2 * np.pi * pass_df["dir"]
    diff_vect = np.sin(theta) + 1j * np.cos(theta)
    start_pos = pass_df["x"] + 1j * pass_df["y"]
    final_pos = rec_df["x"] + 1j * rec_df["y"]
    pred_pos = start_pos + (pos_diff * diff_vect)
    pred_diff = np.linalg.norm(
        pred_pos.to_numpy()[:, None] - final_pos.to_numpy()[:, None],
        ord=2,
        axis=-1,
    )
    return pred_diff


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
        if week > 3:
            continue
        week_fname = f"tracking_week_{week}.csv"
        week_path = os.path.join(data_dir, week_fname)
        weeks_data[week] = pd.read_csv(week_path)
    return weeks_data
