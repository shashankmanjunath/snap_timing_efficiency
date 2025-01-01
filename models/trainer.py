from tqdm import tqdm

import torch.nn.functional as F
import torch.nn as nn
import torch

import tensordict
import wandb


__all__ = [
    "TimeseriesDataset",
    "TransformerTrainer",
]


class TimeseriesDataset:
    def __init__(
        self,
        feature_arr: list[torch.Tensor],
        categorical_arr: list[torch.Tensor],
        label_arr: list[float],
    ):
        self.feature_arr = feature_arr
        self.categorical_arr = categorical_arr
        self.label_arr = label_arr

        if len(self.feature_arr) != len(self.categorical_arr):
            raise RuntimeError(
                "Time-series features and categorical Features are not of equal size!"
            )
        if len(self.feature_arr) != len(self.label_arr):
            raise RuntimeError("Features and labels are not of equal size!")

    def __len__(self) -> int:
        return len(self.feature_arr)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        feat = torch.as_tensor(self.feature_arr[idx])
        cat = torch.as_tensor(self.categorical_arr[idx])
        label = torch.as_tensor(self.label_arr[idx])
        return feat, cat, label


class TransformerTrainer:
    def __init__(
        self,
        network: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        use_wandb: bool = False,
    ):
        self.network = network
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.use_wandb = use_wandb
        self.num_epochs = 5000
        self.lr = 1e-3
        self.batch_size = 4

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.network = self.network.to(self.device)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        if self.use_wandb:
            self.run = wandb.init(
                project="nfl_bdb_2025",
                config={
                    "epochs": self.num_epochs,
                    "lr": self.lr,
                },
            )

    def train(self):
        #  pbar = tqdm(range(self.num_epochs))
        for epoch in range(self.num_epochs):
            pbar = tqdm(self.train_loader, desc=f"[{epoch+1}/{self.num_epochs}]")
            self.network.train()
            for feat, mask, meta_feat, label in pbar:
                feat = feat.to(self.device).float()
                mask = mask.to(self.device).bool()
                meta_feat = meta_feat.to(self.device).float()
                label = label.to(self.device).float()

                self.optim.zero_grad()

                logits = self.network(
                    feat,
                    mask,
                    meta_feat,
                )
                loss = self.criterion(logits.squeeze(), label.float())

                loss.backward()
                self.optim.step()

                if self.use_wandb:
                    self.run.log({"train_loss": loss.item()})

            if epoch % 10 == 0:
                test_loss = self.test()
                pbar.set_postfix({"test_loss": test_loss})

                if self.use_wandb:
                    self.run.log({"test_loss": test_loss})

    def test(self):
        with torch.no_grad():
            self.network.eval()
            pbar = tqdm(self.test_loader, desc="Testing...")
            total_loss = 0
            total_processed = 0

            for feat, mask, meta_feat, label in pbar:
                feat = feat.to(self.device).float()
                mask = mask.to(self.device).bool()
                meta_feat = meta_feat.to(self.device).float()
                label = label.to(self.device).float()
                batch_size = feat.shape[0]

                logits = self.network(
                    feat,
                    mask,
                    meta_feat,
                )
                test_loss = F.l1_loss(logits.squeeze(), label)
                total_loss += batch_size * test_loss.item()
                total_processed += batch_size

                cur_loss = total_loss / total_processed
                pbar.set_postfix({"loss": cur_loss})
        return cur_loss
