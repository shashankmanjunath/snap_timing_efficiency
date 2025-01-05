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


def get_target_feature_cols():
    arr = [
        "x",
        "y",
        "s",
        "a",
        "dis",
        "o",
        "dir",
        #  "club_ARI",
        #  "club_ATL",
        #  "club_BAL",
        #  "club_BUF",
        #  "club_CAR",
        #  "club_CHI",
        #  "club_CIN",
        #  "club_CLE",
        #  "club_DAL",
        #  "club_DEN",
        #  "club_DET",
        #  "club_GB",
        #  "club_HOU",
        #  "club_IND",
        #  "club_JAX",
        #  "club_KC",
        #  "club_LA",
        #  "club_LAC",
        #  "club_LV",
        #  "club_MIA",
        #  "club_MIN",
        #  "club_NE",
        #  "club_NO",
        #  "club_NYG",
        #  "club_NYJ",
        #  "club_PHI",
        #  "club_PIT",
        #  "club_SEA",
        #  "club_SF",
        #  "club_TB",
        #  "club_TEN",
        #  "club_WAS",
        "club_football",
        "playDirection_left",
        "playDirection_right",
        "height",
        "weight",
        #  "position_C",
        #  "position_CB",
        #  "position_DB",
        #  "position_DE",
        #  "position_DT",
        #  "position_FB",
        #  "position_FS",
        #  "position_G",
        #  "position_ILB",
        #  "position_LB",
        #  "position_MLB",
        #  "position_NT",
        #  "position_OLB",
        #  "position_QB",
        #  "position_RB",
        #  "position_SS",
        #  "position_T",
        #  "position_TE",
        #  "position_WR",
        #  "wasTargettedReceiver",
        #  "inMotionAtBallSnap",
        #  "shiftSinceLineset",
        #  "motionSinceLineset",
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
        "other_routeRan_ANGLE",
        "other_routeRan_CORNER",
        "other_routeRan_CROSS",
        "other_routeRan_FLAT",
        "other_routeRan_GO",
        "other_routeRan_HITCH",
        "other_routeRan_IN",
        "other_routeRan_OUT",
        "other_routeRan_POST",
        "other_routeRan_SCREEN",
        "other_routeRan_SLANT",
        "other_routeRan_WHEEL",
        #  "pff_defensiveCoverageAssignment_2L",
        #  "pff_defensiveCoverageAssignment_2R",
        #  "pff_defensiveCoverageAssignment_3L",
        #  "pff_defensiveCoverageAssignment_3M",
        #  "pff_defensiveCoverageAssignment_3R",
        #  "pff_defensiveCoverageAssignment_4IL",
        #  "pff_defensiveCoverageAssignment_4IR",
        #  "pff_defensiveCoverageAssignment_4OL",
        #  "pff_defensiveCoverageAssignment_4OR",
        #  "pff_defensiveCoverageAssignment_CFL",
        #  "pff_defensiveCoverageAssignment_CFR",
        #  "pff_defensiveCoverageAssignment_DF",
        #  "pff_defensiveCoverageAssignment_FL",
        #  "pff_defensiveCoverageAssignment_FR",
        #  "pff_defensiveCoverageAssignment_HCL",
        #  "pff_defensiveCoverageAssignment_HCR",
        #  "pff_defensiveCoverageAssignment_HOL",
        #  "pff_defensiveCoverageAssignment_MAN",
        #  "pff_defensiveCoverageAssignment_PRE",
        "break",
        "maxDist",
        #  "position_ord",
        "down",
        "yardsToGo",
        "absoluteYardlineNumber",
        "teamScore",
        "oppScore",
        "teamWinProb",
        "oppWinProb",
        "playClockAtSnap",
        "passLength",
        "playAction",
        "dropbackDistance",
        "timeToThrow",
        "timeInTackleBox",
        "pff_runPassOption",
        "numRoutes",
        "offenseFormation_EMPTY",
        "offenseFormation_I_FORM",
        "offenseFormation_JUMBO",
        "offenseFormation_PISTOL",
        "offenseFormation_SHOTGUN",
        "offenseFormation_SINGLEBACK",
        "offenseFormation_WILDCAT",
        "receiverAlignment_1x0",
        "receiverAlignment_1x1",
        "receiverAlignment_2x0",
        "receiverAlignment_2x1",
        "receiverAlignment_2x2",
        "receiverAlignment_3x0",
        "receiverAlignment_3x1",
        "receiverAlignment_3x2",
        "receiverAlignment_3x3",
        "receiverAlignment_4x1",
        "receiverAlignment_4x2",
        "dropbackType_DESIGNED_ROLLOUT_LEFT",
        "dropbackType_DESIGNED_ROLLOUT_RIGHT",
        "dropbackType_DESIGNED_RUN",
        "dropbackType_QB_SNEAK",
        "dropbackType_SCRAMBLE",
        "dropbackType_SCRAMBLE_ROLLOUT_LEFT",
        "dropbackType_SCRAMBLE_ROLLOUT_RIGHT",
        "dropbackType_TRADITIONAL",
        "dropbackType_UNKNOWN",
        "passLocationType_INSIDE_BOX",
        "passLocationType_OUTSIDE_LEFT",
        "passLocationType_OUTSIDE_RIGHT",
        "passLocationType_UNKNOWN",
        #  "pff_passCoverage_2-Man",
        #  "pff_passCoverage_Bracket",
        #  "pff_passCoverage_Cover 6-Left",
        #  "pff_passCoverage_Cover-0",
        #  "pff_passCoverage_Cover-1",
        #  "pff_passCoverage_Cover-1 Double",
        #  "pff_passCoverage_Cover-2",
        #  "pff_passCoverage_Cover-3",
        #  "pff_passCoverage_Cover-3 Cloud Left",
        #  "pff_passCoverage_Cover-3 Cloud Right",
        #  "pff_passCoverage_Cover-3 Double Cloud",
        #  "pff_passCoverage_Cover-3 Seam",
        #  "pff_passCoverage_Cover-6 Right",
        #  "pff_passCoverage_Goal Line",
        #  "pff_passCoverage_Miscellaneous",
        #  "pff_passCoverage_Prevent",
        #  "pff_passCoverage_Quarters",
        #  "pff_passCoverage_Red Zone",
        #  "pff_manZone_Man",
        #  "pff_manZone_Other",
        #  "pff_manZone_Zone",
        #  "possessionTeam_ARI",
        #  "possessionTeam_ATL",
        #  "possessionTeam_BAL",
        #  "possessionTeam_BUF",
        #  "possessionTeam_CAR",
        #  "possessionTeam_CHI",
        #  "possessionTeam_CIN",
        #  "possessionTeam_CLE",
        #  "possessionTeam_DAL",
        #  "possessionTeam_DEN",
        #  "possessionTeam_DET",
        #  "possessionTeam_GB",
        #  "possessionTeam_HOU",
        #  "possessionTeam_IND",
        #  "possessionTeam_JAX",
        #  "possessionTeam_KC",
        #  "possessionTeam_LA",
        #  "possessionTeam_LAC",
        #  "possessionTeam_LV",
        #  "possessionTeam_MIA",
        #  "possessionTeam_MIN",
        #  "possessionTeam_NE",
        #  "possessionTeam_NO",
        #  "possessionTeam_NYG",
        #  "possessionTeam_NYJ",
        #  "possessionTeam_PHI",
        #  "possessionTeam_PIT",
        #  "possessionTeam_SEA",
        #  "possessionTeam_SF",
        #  "possessionTeam_TB",
        #  "possessionTeam_TEN",
        #  "possessionTeam_WAS",
        #  "defensiveTeam_ARI",
        #  "defensiveTeam_ATL",
        #  "defensiveTeam_BAL",
        #  "defensiveTeam_BUF",
        #  "defensiveTeam_CAR",
        #  "defensiveTeam_CHI",
        #  "defensiveTeam_CIN",
        #  "defensiveTeam_CLE",
        #  "defensiveTeam_DAL",
        #  "defensiveTeam_DEN",
        #  "defensiveTeam_DET",
        #  "defensiveTeam_GB",
        #  "defensiveTeam_HOU",
        #  "defensiveTeam_IND",
        #  "defensiveTeam_JAX",
        #  "defensiveTeam_KC",
        #  "defensiveTeam_LA",
        #  "defensiveTeam_LAC",
        #  "defensiveTeam_LV",
        #  "defensiveTeam_MIA",
        #  "defensiveTeam_MIN",
        #  "defensiveTeam_NE",
        #  "defensiveTeam_NO",
        #  "defensiveTeam_NYG",
        #  "defensiveTeam_NYJ",
        #  "defensiveTeam_PHI",
        #  "defensiveTeam_PIT",
        #  "defensiveTeam_SEA",
        #  "defensiveTeam_SF",
        #  "defensiveTeam_TB",
        #  "defensiveTeam_TEN",
        #  "defensiveTeam_WAS",
        "pct_elapsed",
    ]
    return arr


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
    game_data: pd.DataFrame,
    play_metadata: pd.DataFrame,
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

    if pass_time_data.shape[0] == 0 or (pass_time_data.shape[0] != 23):
        return pd.DataFrame()

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

    # Getting player team score and opposing score
    game_id_data = game_data[game_data["gameId"] == route_data["gameId"].iloc[0].item()]
    route_data["teamScore"] = 0.0
    route_data["oppScore"] = 0.0
    route_data["teamWinProb"] = 0.0
    route_data["oppWinProb"] = 0.0
    route_data = route_data.copy()

    home_team_loc = route_data["teamAbbr"] == game_id_data["homeTeamAbbr"].item()
    vis_team_loc = route_data["teamAbbr"] == game_id_data["visitorTeamAbbr"].item()

    home_score = play_metadata["preSnapHomeScore"].item()
    vis_score = play_metadata["preSnapVisitorScore"].item()
    home_pwin = play_metadata["preSnapHomeTeamWinProbability"].item()
    vis_pwin = play_metadata["preSnapVisitorTeamWinProbability"].item()
    route_data.loc[home_team_loc, "teamScore"] = home_score
    route_data.loc[home_team_loc, "oppScore"] = vis_score
    route_data.loc[home_team_loc, "teamWinProb"] = home_pwin
    route_data.loc[home_team_loc, "oppWinProb"] = vis_pwin

    route_data.loc[vis_team_loc, "teamScore"] = vis_score
    route_data.loc[vis_team_loc, "oppScore"] = home_score
    route_data.loc[vis_team_loc, "teamWinProb"] = vis_pwin
    route_data.loc[vis_team_loc, "oppWinProb"] = home_pwin
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
        #  if week > 1:
        #      break
        week_fname = f"tracking_week_{week}.csv"
        week_path = os.path.join(data_dir, week_fname)
        weeks_data[week] = pd.read_csv(week_path)
    return weeks_data
