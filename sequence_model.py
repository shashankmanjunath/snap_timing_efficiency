from fire import Fire

import sklearn.metrics
import torch

import processor
import models

# Separation Timing Efficiency


def main(data_dir: str, use_wandb: bool = False):
    train_weeks = [1]
    train_dataset = Dataset(weeks=train_weeks, data_dir=data_dir)

    train_dataset = Dataset(train_weeks, data_dir)

    network = models.TransformerTimeSeries(
        embed_dim=16,
        n_heads=8,
        n_blocks=4,
        n_categorical=8,
    )
    trainer = models.TransformerTrainer(
        network,
        train_dataset,
        # TODO: Repeating train dataset
        train_dataset,
        use_wandb=use_wandb,
    )
    trainer.train()


class Dataset:
    def __init__(
        self,
        weeks: list[int],
        data_dir: str,
    ):

        proc = processor.SeparationDataProcessor(data_dir)
        seq_features, seq_mask, meta_features, sep = proc.process(weeks)

        self.seq_features = torch.as_tensor(seq_features)
        self.seq_mask = torch.as_tensor(seq_mask)
        self.meta_features = torch.as_tensor(meta_features)
        self.sep = torch.as_tensor(sep)

        self.get_baseline_acc()

    def __len__(self):
        return self.seq_features.shape[0]

    def get_baseline_acc(self):
        mean_val = self.sep.mean()
        arr = torch.Tensor(self.sep.shape).fill_(mean_val)
        test_acc = sklearn.metrics.mean_absolute_error(
            self.sep.cpu().numpy().squeeze(),
            arr.cpu().numpy(),
        )
        print(f"Baseline Accuracy: {test_acc:.3f}")
        pass

    def __getitem__(self, idx):
        # Removing football location
        feat = self.seq_features[idx, :, :-1, :]
        mask = ~self.seq_mask[idx, :, 0, 0]
        meta_feat = self.meta_features[idx, :]
        label = self.sep[idx]
        return feat, mask, meta_feat, label


if __name__ == "__main__":
    Fire(main)
