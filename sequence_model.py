from fire import Fire

#  import torch.nn as nn

import processor

# Separation Timing Efficiency


def main(data_dir: str):
    train_weeks = [1]
    train_dataset = Dataset(weeks=train_weeks, data_dir=data_dir)


class Dataset:
    def __init__(
        self,
        weeks: list[int],
        data_dir: str,
    ):

        proc = processor.SeparationDataProcessor(data_dir)
        seq_features, seq_mask, meta_features, player_features, sep = proc.process(
            weeks
        )

        self.seq_features = seq_features
        self.seq_mask = seq_mask
        self.meta_features = meta_features
        self.player_features = player_features
        self.sep = sep

    def __len__(self):
        return self.seq_features.shape[0]

    def __getitem__(self, idx):
        pass


if __name__ == "__main__":
    Fire(main)
