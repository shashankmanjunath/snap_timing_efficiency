from fire import Fire

import sklearn.metrics
import numpy as np
import xgboost

import processor


def create_feature_arr(seq_features, seq_mask, meta_features) -> np.ndarray:
    # Creating array with final fied positions and player data of players
    X = []
    for pos_feat, mask, meta_feat in zip(seq_features, seq_mask, meta_features):
        # Extracting final positions of players
        seq_mask = mask[:, 0, 0]
        pos_arr = pos_feat[seq_mask, :, :][-1, :, :]

        # Dropping row with nan value (this is the ball)
        pos_arr = pos_arr[~np.isnan(pos_arr).any(axis=1)]

        feat_arr = np.concatenate((pos_arr.reshape(-1), meta_feat), axis=-1)
        X.append(feat_arr)
    X = np.stack(X)
    return X


def parameter_search(data_dir: str) -> None:
    #  train_weeks = [1, 2, 3, 4, 5, 6, 7]
    train_weeks = [1, 2]
    proc = processor.SeparationDataProcessor(data_dir)
    seq_features_train, seq_mask_train, meta_features_train, sep_train = proc.process(
        train_weeks,
    )
    X_train = create_feature_arr(
        seq_features_train,
        seq_mask_train,
        meta_features_train,
    )
    y_train = sep_train

    param_grid = {
        "max_depth": [3, 5, 7, 10, 20, 50],
        "learning_rate": [0.01],
        "subsample": [0.5, 0.7, 1],
        "colsample_bytree": [0.5, 0.7, 1],
        "lambda": [0.1, 1.0, 10.0],
    }

    xgb_model = xgboost.XGBRegressor()
    print("Starting grid search...")
    grid_search = sklearn.model_selection.GridSearchCV(
        xgb_model,
        param_grid,
        cv=5,
        #  scoring=sklearn.metrics.mean_absolute_error,
    )

    grid_search.fit(X_train, y_train)

    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    return


def main(data_dir: str) -> None:
    weeks_nums = [x for x in range(1, 3)]
    n_splits = min(len(weeks_nums), 5)
    kf = sklearn.model_selection.KFold(n_splits=n_splits)

    for fold_idx, (train_weeks_idx, test_weeks_idx) in enumerate(kf.split(weeks_nums)):
        train_weeks = [weeks_nums[idx] for idx in train_weeks_idx]
        test_weeks = [weeks_nums[idx] for idx in test_weeks_idx]

        proc = processor.SeparationDataProcessor(data_dir)

        print(f"Fold {fold_idx} Loading Data....")
        seq_features_train, seq_mask_train, meta_features_train, sep_train = (
            proc.process(
                train_weeks,
            )
        )
        X_train = create_feature_arr(
            seq_features_train,
            seq_mask_train,
            meta_features_train,
        )
        y_train = sep_train
        seq_features_test, seq_mask_test, meta_features_test, sep_test = proc.process(
            test_weeks,
        )
        X_test = create_feature_arr(
            seq_features_test,
            seq_mask_test,
            meta_features_test,
        )
        y_test = sep_test

        # Train/Test Split
        bst = xgboost.XGBRegressor(
            learning_rate=0.01,
            max_depth=7,
            subsample=0.7,
        )
        bst.fit(X_train, y_train)
        preds_train = bst.predict(X_train)
        preds_test = bst.predict(X_test)
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

        # TODO: Print baseline MAE if we predict the average value of labels
        print("-----")
        print(f"Baseline Train Accuracy: {baseline_train_mae:.3f}")
        print(f"Fold {fold_idx} Train MAE: {train_mae:.3f}")
        print("")
        print(f"Baseline Test Accuracy: {baseline_train_mae:.3f}")
        print(f"Fold {fold_idx} Test MAE: {test_mae:.3f}")
        print("-----")
    return


if __name__ == "__main__":
    Fire(
        {
            "cv": main,
            "param_search": parameter_search,
        }
    )
