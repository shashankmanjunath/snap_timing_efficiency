import pickle

from fire import Fire
from tqdm import tqdm

import sklearn.linear_model
import sklearn.metrics
import pandas as pd
import numpy as np
import xgboost
import h5py

import processor
import utils


def get_position_cols() -> list[str]:
    arr = [
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
    ]
    return arr


def decode(arr: list) -> list[str]:
    return [x.decode("utf-8") for x in arr]


def get_route_types() -> list[str]:
    return ["positive", "short", "medium", "long", "all"]


def extract_route_depth(
    feat_df: pd.DataFrame,
    route_type: str,
) -> list[bool]:
    if route_type == "short":
        feat_idxs = feat_df["passLength"] <= 5.0
    elif route_type == "medium":
        feat_idxs = (feat_df["passLength"] > 5.0) & (feat_df["passLength"] <= 15.0)
    elif route_type == "long":
        feat_idxs = feat_df["passLength"] <= 5.0
    elif route_type == "all":
        feat_idxs = [True] * feat_df.shape[0]
    elif route_type == "positive":
        feat_idxs = feat_df["passLength"] >= 0.0
    else:
        raise RuntimeError(f"Route type {route_type} not recognized!")
    return feat_idxs


def create_feature_arr(f: h5py.File) -> np.ndarray:
    # Creating array with final fied positions and player data of players
    X = []
    n = f["seq_arr"].shape[0]
    for idx in tqdm(range(n), desc="Processing data into array..."):
        # Extracting final positions of players
        seq_mask = f["seq_mask"][idx, :, 0, 0].astype(int)
        idxs = np.where(seq_mask)[0]
        seq_len = idxs.max() + 1
        pos_arr = f["seq_arr"][idx, :seq_len, :, :][-1, :, :]

        # Dropping row with nan value (this is the ball)
        pos_arr = pos_arr[~np.isnan(pos_arr).any(axis=1)]
        pos_df = pd.DataFrame(
            pos_arr,
            columns=decode(f["seq_cols"]),
        )
        pos_df["nflId"] = pos_df["nflId"].astype(int)
        play_players_df = pd.DataFrame(
            f["play_players_arr"][idx, :, :],
            columns=decode(f["play_players_cols"]),
        )
        play_players_df["nflId"] = play_players_df["nflId"].astype(int)
        play_overall_df = pd.DataFrame(
            f["play_overall_arr"][idx, :, :],
            columns=decode(f["play_overall_cols"]),
        )
        play_overall_df["nflId"] = play_overall_df["nflId"].astype(int)

        pos_df = pos_df.merge(play_players_df, how="outer", on="nflId")
        pos_df = pos_df.merge(
            play_overall_df, how="outer", on=["gameId", "playId", "nflId"]
        )

        pos_cols = get_position_cols()
        pos_df["position_ord"] = np.argmax(pos_df[pos_cols].to_numpy(), axis=1)
        pos_df = pos_df.sort_values(by="position_ord")
        pos_df = pos_df[pos_df["wasTargettedReceiver"] == 1.0]
        #  pos_arr = pos_df.to_numpy()

        meta_df = pd.DataFrame(
            f["meta_arr"][idx, :].reshape(1, -1),
            columns=decode(f["meta_cols"]),
        )
        pos_df = pos_df.merge(meta_df, how="outer", on=["gameId", "playId"])
        pos_df = pos_df.drop(
            [
                "gameId",
                "playId",
                "nflId",
                "frameId",
                "position_ord",
                "inMotionAtBallSnap",
                "shiftSinceLineset",
                "motionSinceLineset",
            ],
            axis=1,
        )
        X.append(pos_df)
    X = pd.concat(X, axis=0).reset_index(drop=True)
    return X


def parameter_search(data_dir: str, route_type: str) -> None:
    if route_type not in get_route_types():
        raise RuntimeError(f"Route type {route_type} not recognized!")

    train_weeks = [1, 2, 3]
    proc = processor.SeparationDataProcessor(data_dir)
    X_train, y_train = proc.process(train_weeks)

    train_idxs = extract_route_depth(X_train, route_type)
    X_train = X_train[train_idxs]
    y_train = y_train[train_idxs]

    param_grid = {
        "max_depth": [7, 15, 25, 100],
        "learning_rate": [0.01],
        "subsample": [0.5, 0.7, 1],
        "colsample_bytree": [0.5, 0.7, 1],
    }

    xgb_model = xgboost.XGBRegressor()
    print("Starting grid search...")
    grid_search = sklearn.model_selection.GridSearchCV(
        xgb_model,
        param_grid,
        cv=3,
        #  scoring=sklearn.metrics.mean_absolute_error,
    )

    grid_search.fit(X_train, y_train)

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    return


def train_model(data_dir: str, route_type: str) -> xgboost.XGBRegressor:
    if route_type not in get_route_types():
        raise RuntimeError(f"Route type {route_type} not recognized!")

    train_weeks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_weeks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #  test_weeks = [8, 9]

    proc = processor.SeparationDataProcessor(data_dir)

    X_train, y_train = proc.process(train_weeks)
    X_test, y_test = proc.process(test_weeks)

    train_idxs = extract_route_depth(X_train, route_type)
    X_train = X_train[train_idxs]
    y_train = y_train[train_idxs]

    test_idxs = extract_route_depth(X_test, route_type)
    X_test = X_test[test_idxs]
    y_test = y_test[test_idxs]

    X_train = X_train[utils.get_target_feature_cols()]
    X_test = X_test[utils.get_target_feature_cols()]

    print(f"Train Samples: {X_train.shape[0]}")
    print(f"Test Samples: {X_test.shape[0]}")

    # Train/Test Split
    bst = xgboost.XGBRegressor(
        colsample_bytree=1,
        reg_lambda=1.0,
        learning_rate=0.01,
        max_depth=7,
        subsample=0.5,
    )
    bst.fit(X_train, y_train)
    preds_train = bst.predict(X_train)
    preds_test = bst.predict(X_test)
    train_r2 = bst.score(X_train, y_train)
    test_r2 = bst.score(X_test, y_test)

    train_mae = sklearn.metrics.mean_absolute_error(y_train, preds_train)
    test_mae = sklearn.metrics.mean_absolute_error(y_test, preds_test)

    train_baseline = np.zeros(y_train.shape) + np.mean(y_train)
    test_baseline = np.zeros(y_test.shape) + np.mean(y_test)

    baseline_train_mae = sklearn.metrics.mean_absolute_error(
        y_train,
        train_baseline,
    )
    baseline_test_mae = sklearn.metrics.mean_absolute_error(
        y_test,
        test_baseline,
    )

    print("-----")
    print(f"Baseline Train MAE: {baseline_train_mae:.3f}")
    print(f"Train MAE: {train_mae:.3f}")
    print(f"Train R2: {train_r2:.3f}")
    print("")
    print(f"Baseline Test MAE: {baseline_test_mae:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Test R2: {test_r2:.3f}")
    print("-----")

    xgb_fname = f"xgb_model_{route_type}_all_weeks.pkl"
    print(f"Saving model to {xgb_fname}...")

    with open(xgb_fname, "wb") as f:
        pickle.dump(bst, f)


def main(data_dir: str, route_type: str) -> None:
    if route_type not in get_route_types():
        raise RuntimeError(f"Route type {route_type} not recognized!")
    weeks_nums = [x for x in range(1, 10)]
    n_splits = min(len(weeks_nums), 5)
    kf = sklearn.model_selection.KFold(n_splits=n_splits)

    cv_test_mae = []
    cv_test_r2 = []

    proc = processor.SeparationDataProcessor(data_dir)

    for fold_idx, (train_weeks_idx, test_weeks_idx) in enumerate(kf.split(weeks_nums)):
        train_weeks = [weeks_nums[idx] for idx in train_weeks_idx]
        test_weeks = [weeks_nums[idx] for idx in test_weeks_idx]

        print(f"Fold {fold_idx} Loading Data....")
        X_train, y_train = proc.process(train_weeks)
        X_test, y_test = proc.process(test_weeks)

        train_idxs = extract_route_depth(X_train, route_type)
        X_train = X_train[train_idxs]
        y_train = y_train[train_idxs]

        test_idxs = extract_route_depth(X_test, route_type)
        X_test = X_test[test_idxs]
        y_test = y_test[test_idxs]

        X_train = X_train[utils.get_target_feature_cols()]
        X_test = X_test[utils.get_target_feature_cols()]
        print(f"Train Samples: {X_train.shape[0]}")
        print(f"Test Samples: {X_test.shape[0]}")

        # Train/Test Split
        xgb_params = {
            "all": {
                "colsample_bytree": 0.7,
                "learning_rate": 0.01,
                "max_depth": 15,
                "subsample": 0.5,
            },
            "short": {
                "colsample_bytree": 1,
                "learning_rate": 0.01,
                "max_depth": 15,
                "subsample": 0.5,
            },
            "medium": {},
            "long": {
                "colsample_bytree": 1,
                "learning_rate": 0.01,
                "max_depth": 15,
                "subsample": 0.5,
            },
            "positive": {},
        }
        bst = xgboost.XGBRegressor(**xgb_params[route_type])

        bst.fit(X_train, y_train)
        preds_train = bst.predict(X_train)
        preds_test = bst.predict(X_test)
        train_r2 = bst.score(X_train, y_train)
        test_r2 = bst.score(X_test, y_test)

        train_mae = sklearn.metrics.mean_absolute_error(y_train, preds_train)
        test_mae = sklearn.metrics.mean_absolute_error(y_test, preds_test)

        cv_test_mae.append(test_mae)
        cv_test_r2.append(test_r2)

        train_baseline = np.zeros(y_train.shape) + np.mean(y_train)
        test_baseline = np.zeros(y_test.shape) + np.mean(y_test)

        baseline_train_mae = sklearn.metrics.mean_absolute_error(
            y_train,
            train_baseline,
        )
        baseline_test_mae = sklearn.metrics.mean_absolute_error(
            y_test,
            test_baseline,
        )

        print("-----")
        print(f"Baseline Train MAE: {baseline_train_mae:.3f}")
        print(f"Fold {fold_idx} Train MAE: {train_mae:.3f}")
        print(f"Fold {fold_idx} Train R2: {train_r2:.3f}")
        print("")
        print(f"Baseline Test MAE: {baseline_test_mae:.3f}")
        print(f"Fold {fold_idx} Test MAE: {test_mae:.3f}")
        print(f"Fold {fold_idx} Test R2: {test_r2:.3f}")
        print("-----")

    print(f"Final Average Test MAE: {np.mean(cv_test_mae):.3f}")
    print(f"Final Average Test R2: {np.mean(cv_test_r2):.3f}")
    return


if __name__ == "__main__":
    Fire(
        {
            "cv": main,
            "train": train_model,
            "param_search": parameter_search,
        }
    )
