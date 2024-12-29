import os

from fire import Fire
import pandas as pd

# Separation Timing Efficiency


def main(data_dir: str):
    data = load_data(data_dir)
    train_weeks = [1]
    train_dataset = Dataset(weeks=train_weeks, data=data, data_dir=data_dir)


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


def get_in_motion_at_snap_plays(data: dict[str, pd.DataFrame]) -> list:
    # Getting plays with motion
    plays = data["player_play"][data["player_play"]["inMotionAtBallSnap"] == True]

    # Finding plays where the motion player was the targeted receiver
    motion_receiver_target = plays[plays["wasTargettedReceiver"] == True]
    return motion_receiver_target


class Dataset:
    def __init__(
        self,
        weeks: list[int],
        data: dict[str, pd.DataFrame],
        data_dir: str,
    ):
        self.weeks_fnames = []
        self.weeks_data = []

        for week in weeks:
            week_fname = f"tracking_week_{week}.csv"
            week_path = os.path.join(data_dir, week_fname)
            self.weeks_fnames.append(week_path)
            self.weeks_data.append(pd.read_csv(week_path))

        motion_receiver_target = get_in_motion_at_snap_plays(data)
        pass


if __name__ == "__main__":
    Fire(main)
