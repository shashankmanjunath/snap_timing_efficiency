import sklearn.preprocessing
import numpy as np
import pandas as pd


#  def get_player_feature_columns() -> list[str]:
#      arr = ["height", "weight", "collegeName", "position", "displayName_ord"]
#      return arr


def get_seq_feature_columns() -> list[str]:
    arr = [
        "club_ord",
        "x",
        "y",
        "s",
        "a",
        "dis",
        "o",
        "dir",
        "height",
        "weight",
        "collegeName",
        "position",
        "displayName_ord",
    ]
    return arr


def get_meta_feature_columns() -> list[str]:
    arr = [
        #  "quarter",
        #  "gameClock",
        "down",
        "yardsToGo",
        "possessionTeam",
        "defensiveTeam",
        "preSnapHomeScore",
        "preSnapVisitorScore",
        "absoluteYardlineNumber",
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

        stacked_frames = np.stack([x[cols].to_numpy() for x in frame_data], axis=0)
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


def ordinalize_player_data(player_data: pd.DataFrame):
    player_data["height"] = player_data["height"].apply(player_height_in)
    encoder = sklearn.preprocessing.OrdinalEncoder()
    ordinal_cols = ["collegeName", "position", "displayName"]
    player_data[ordinal_cols] = encoder.fit_transform(player_data[ordinal_cols])
    return player_data


class DataFeaturizer:
    def __init__(self):
        self.club_func = sklearn.preprocessing.OrdinalEncoder()
        self.club_fit = False

        self.play_func = sklearn.preprocessing.OrdinalEncoder()
        self.play_fit = False

        self.route_func = sklearn.preprocessing.OrdinalEncoder()
        self.route_fit = False

    def featurize_week(self, week_data: pd.DataFrame) -> pd.DataFrame:
        week_data["o"] = week_data["o"] / 360.0
        if not self.club_fit:
            self.club_func.fit(week_data[["club"]])
            self.club_fit = True

        week_data["club_ord"] = self.club_func.transform(week_data[["club"]])
        return week_data

    def featurize_play(self, play_data: pd.DataFrame):
        cols = ["possessionTeam", "defensiveTeam"]
        if not self.play_fit:
            self.play_func.fit(play_data[cols])
            self.play_fit = True

        play_data[cols] = self.play_func.transform(play_data[cols])
        return play_data

    def featurize_player_play(self, player_play_data: pd.DataFrame):
        cols = ["routeRan", "wasRunningRoute", "pff_defensiveCoverageAssignment"]
        if not self.route_fit:
            self.route_func.fit(player_play_data[cols])
            self.route_fit = True

        ord_cols = [x + "_ord" for x in cols]
        player_play_data[ord_cols] = self.route_func.transform(player_play_data[cols])
        return player_play_data
