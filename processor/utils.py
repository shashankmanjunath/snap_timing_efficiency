import sklearn.preprocessing
import numpy as np
import pandas as pd


#  def get_player_feature_columns() -> list[str]:
#      arr = ["height", "weight", "collegeName", "position", "displayName_ord"]
#      return arr


def get_seq_feature_columns() -> list[str]:
    arr = [
        #  "playDirection", # TODO
        "x",
        "y",
        "s",
        "a",
        "dis",
        "o",
        "dir",
        "club_ARI",
        "club_ATL",
        "club_BAL",
        "club_BUF",
        "club_CAR",
        "club_CHI",
        "club_CIN",
        "club_CLE",
        "club_DAL",
        "club_DEN",
        "club_DET",
        "club_GB",
        "club_HOU",
        "club_IND",
        "club_JAX",
        "club_KC",
        "club_LA",
        "club_LAC",
        "club_LV",
        "club_MIA",
        "club_MIN",
        "club_NE",
        "club_NO",
        "club_NYG",
        "club_NYJ",
        "club_PHI",
        "club_PIT",
        "club_SEA",
        "club_SF",
        "club_TB",
        "club_TEN",
        "club_WAS",
        "club_football",
        "height",
        "weight",
        "position_C",
        "position_CB",
        "position_DB",
        "position_DE",
        "position_DT",
        "position_FB",
        "position_FS",
        "position_G",
        "position_ILB",
        "position_LB",
        "position_MLB",
        "position_NT",
        "position_OLB",
        "position_QB",
        "position_RB",
        "position_SS",
        "position_T",
        "position_TE",
        "position_WR",
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
        "pff_defensiveCoverageAssignment_2L",
        "pff_defensiveCoverageAssignment_2R",
        "pff_defensiveCoverageAssignment_3L",
        "pff_defensiveCoverageAssignment_3M",
        "pff_defensiveCoverageAssignment_3R",
        "pff_defensiveCoverageAssignment_4IL",
        "pff_defensiveCoverageAssignment_4IR",
        "pff_defensiveCoverageAssignment_4OL",
        "pff_defensiveCoverageAssignment_4OR",
        "pff_defensiveCoverageAssignment_CFL",
        "pff_defensiveCoverageAssignment_CFR",
        "pff_defensiveCoverageAssignment_DF",
        "pff_defensiveCoverageAssignment_FL",
        "pff_defensiveCoverageAssignment_FR",
        "pff_defensiveCoverageAssignment_HCL",
        "pff_defensiveCoverageAssignment_HCR",
        "pff_defensiveCoverageAssignment_HOL",
        "pff_defensiveCoverageAssignment_MAN",
        "pff_defensiveCoverageAssignment_PRE",
        "maxDist",
    ]
    return arr


def get_meta_feature_columns() -> list[str]:
    arr = [
        "down",
        "yardsToGo",
        "preSnapHomeScore",
        "preSnapVisitorScore",
        "absoluteYardlineNumber",
        "preSnapHomeTeamWinProbability",
        "preSnapVisitorTeamWinProbability",
        "expectedPoints",
        "playClockAtSnap",
        "passLength",
        "playAction",
        "dropbackDistance",
        "timeToThrow",
        "timeInTackleBox",
        "pff_runPassOption",
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
        "pff_passCoverage_2-Man",
        "pff_passCoverage_Bracket",
        "pff_passCoverage_Cover 6-Left",
        "pff_passCoverage_Cover-0",
        "pff_passCoverage_Cover-1",
        "pff_passCoverage_Cover-1 Double",
        "pff_passCoverage_Cover-2",
        "pff_passCoverage_Cover-3",
        "pff_passCoverage_Cover-3 Cloud Left",
        "pff_passCoverage_Cover-3 Cloud Right",
        "pff_passCoverage_Cover-3 Double Cloud",
        "pff_passCoverage_Cover-3 Seam",
        "pff_passCoverage_Cover-6 Right",
        "pff_passCoverage_Goal Line",
        "pff_passCoverage_Miscellaneous",
        "pff_passCoverage_Prevent",
        "pff_passCoverage_Quarters",
        "pff_passCoverage_Red Zone",
        "pff_manZone_Man",
        "pff_manZone_Other",
        "pff_manZone_Zone",
        "possessionTeam_ARI",
        "possessionTeam_ATL",
        "possessionTeam_BAL",
        "possessionTeam_BUF",
        "possessionTeam_CAR",
        "possessionTeam_CHI",
        "possessionTeam_CIN",
        "possessionTeam_CLE",
        "possessionTeam_DAL",
        "possessionTeam_DEN",
        "possessionTeam_DET",
        "possessionTeam_GB",
        "possessionTeam_HOU",
        "possessionTeam_IND",
        "possessionTeam_JAX",
        "possessionTeam_KC",
        "possessionTeam_LA",
        "possessionTeam_LAC",
        "possessionTeam_LV",
        "possessionTeam_MIA",
        "possessionTeam_MIN",
        "possessionTeam_NE",
        "possessionTeam_NO",
        "possessionTeam_NYG",
        "possessionTeam_NYJ",
        "possessionTeam_PHI",
        "possessionTeam_PIT",
        "possessionTeam_SEA",
        "possessionTeam_SF",
        "possessionTeam_TB",
        "possessionTeam_TEN",
        "possessionTeam_WAS",
        "defensiveTeam_ARI",
        "defensiveTeam_ATL",
        "defensiveTeam_BAL",
        "defensiveTeam_BUF",
        "defensiveTeam_CAR",
        "defensiveTeam_CHI",
        "defensiveTeam_CIN",
        "defensiveTeam_CLE",
        "defensiveTeam_DAL",
        "defensiveTeam_DEN",
        "defensiveTeam_DET",
        "defensiveTeam_GB",
        "defensiveTeam_HOU",
        "defensiveTeam_IND",
        "defensiveTeam_JAX",
        "defensiveTeam_KC",
        "defensiveTeam_LA",
        "defensiveTeam_LAC",
        "defensiveTeam_LV",
        "defensiveTeam_MIA",
        "defensiveTeam_MIN",
        "defensiveTeam_NE",
        "defensiveTeam_NO",
        "defensiveTeam_NYG",
        "defensiveTeam_NYJ",
        "defensiveTeam_PHI",
        "defensiveTeam_PIT",
        "defensiveTeam_SEA",
        "defensiveTeam_SF",
        "defensiveTeam_TB",
        "defensiveTeam_TEN",
        "defensiveTeam_WAS",
    ]
    return arr


def convert_game_to_pct(row: pd.Series) -> float:
    game_length = 60.0 * 60.0
    quarter_time = 0.25 * (row["quarter"] - 1) * game_length

    cmin, csec = row["gameClock"].split(":")
    s_to_end = 60.0 * float(cmin) + float(csec)
    s_elapsed = (15.0 * 60.0) - s_to_end
    pct_elapsed = (quarter_time + s_elapsed) / game_length
    return pct_elapsed


def extract_meta_features(meta_features: pd.DataFrame):
    t_feat = meta_features[["quarter", "gameClock"]]
    pct_elapsed = t_feat.apply(convert_game_to_pct, axis=1)

    cols = get_meta_feature_columns()
    meta_df = meta_features[cols].copy()
    meta_df["pct_elapsed"] = pct_elapsed
    meta_arr = meta_df.to_numpy()
    return meta_arr


#  def extract_player_play_features(player_play: list[pd.DataFrame]) -> np.ndarray:
#      cols = get_player_feature_columns()
#      arr = [x[cols].to_numpy() for x in player_play]
#      arr = np.stack(arr, axis=0)
#      return arr


def convert_frame_to_array(frame: pd.DataFrame) -> np.ndarray:
    oh_cols = [x for x in frame.columns if "onehot" in x]
    float_cols = [x for x in frame.columns if "onehot" not in x]
    pass


def extract_seq_arr(
    seq_features: list[pd.DataFrame],
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    cols = get_seq_feature_columns()
    pad_arrs = []
    mask_arrs = []
    for seq_feature in seq_features:
        frames = sorted(seq_feature["frameId"].unique().tolist())
        frame_data = [
            seq_feature[seq_feature["frameId"] == frame_id] for frame_id in frames
        ]
        frame_data = [x.sort_values(by=["position"]) for x in frame_data]

        #  stacked_frames = np.stack([x[cols].to_numpy() for x in frame_data], axis=0)
        stacked_frames = np.stack([x[cols] for x in frame_data], axis=0)
        mask = np.ones(stacked_frames.shape)

        pad_size = max_seq_len - stacked_frames.shape[0]
        pad_arr = np.pad(stacked_frames, ((0, pad_size), (0, 0), (0, 0)))
        mask_arr = np.pad(mask, ((0, pad_size), (0, 0), (0, 0)))

        pad_arrs.append(pad_arr)
        mask_arrs.append(mask_arr)
    pad_arrs = np.stack(pad_arrs, axis=0)
    mask_arrs = np.stack(mask_arrs, axis=0)
    return pad_arrs, mask_arrs


def player_height_in(x: str) -> float:
    feet, inches = [float(k) for k in x.split("-")]
    height = (12.0 * feet) + inches
    return height


class DataFeaturizer:
    def __init__(self):
        self.encoder = sklearn.preprocessing.OneHotEncoder()

    def featurize_week(self, week_data: pd.DataFrame) -> pd.DataFrame:
        week_data["o"] = week_data["o"] / 360.0

        cols = ["club", "playDirection"]
        df_dummies = pd.get_dummies(week_data[cols], prefix=cols)
        week_data = pd.concat((week_data, df_dummies), axis=1)
        return week_data

    def featurize_play(self, play_data: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "offenseFormation",
            "receiverAlignment",
            "dropbackType",
            "passLocationType",
            "pff_passCoverage",
            "pff_manZone",
            "possessionTeam",
            "defensiveTeam",
        ]
        df_dummies = pd.get_dummies(play_data[cols], prefix=cols)
        play_data = pd.concat((play_data, df_dummies), axis=1)
        return play_data

    def featurize_player_play(self, player_play_data: pd.DataFrame) -> pd.DataFrame:
        # Filling route NaN with 0
        rcol = "wasRunningRoute"
        player_play_data[rcol] = player_play_data[rcol].fillna(0)

        # Ordinalizing other columns
        cols = [
            "routeRan",
            "pff_defensiveCoverageAssignment",
        ]
        df_dummies = pd.get_dummies(player_play_data[cols], prefix=cols)
        player_play_data = pd.concat((player_play_data, df_dummies), axis=1)
        return player_play_data

    def featurize_player_data(self, player_data: pd.DataFrame) -> pd.DataFrame:
        player_data["height"] = player_data["height"].apply(player_height_in)

        cols = ["position"]
        df_dummies = pd.get_dummies(player_data[cols], prefix=cols)
        player_data = pd.concat((player_data, df_dummies), axis=1)
        return player_data
