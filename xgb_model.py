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


def main(data_dir: str):
    weeks_nums = [x for x in range(1, 10)]
    kf = sklearn.model_selection.KFold(n_splits=5)

    for fold_idx, (train_weeks_idx, test_weeks_idx) in enumerate(kf.split(weeks_nums)):
        train_weeks = [weeks_nums[idx] for idx in train_weeks_idx]
        test_weeks = [weeks_nums[idx] for idx in test_weeks_idx]

        proc = processor.SeparationDataProcessor(
            data_dir,
        )

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
        bst = xgboost.XGBRegressor()
        bst.fit(X_train, y_train)
        preds_train = bst.predict(X_train)
        preds_test = bst.predict(X_test)
        train_mae = sklearn.metrics.mean_absolute_error(y_train, preds_train)
        test_mae = sklearn.metrics.mean_absolute_error(y_test, preds_test)

        print(f"Fold {fold_idx} Train MAE: {train_mae:.3f}")
        print(f"Fold {fold_idx} Test MAE: {test_mae:.3f}")
    return


if __name__ == "__main__":
    Fire(main)
