import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch


# データセットのカスタムクラス
class TableDataset(Dataset):
    def __init__(
        self,
        data_df: pd.DataFrame,
        label_column: str,
        feature_columns: list,
        transform=None,
    ):
        self.labels = data_df[label_column].values
        self.features = data_df[feature_columns].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


# モデルビルダーのクラス
class ModelBuilder:
    def __init__(
        self,
        input_size: int
    ):
        self.input_size = input_size

    def build(
        self,
        model_type: str = "simple"
    ):
        if model_type == "simple":
            return nn.Sequential(
                nn.Linear(self.input_size, 1),
                nn.Sigmoid()
            )
        elif model_type == "mlp":
            return nn.Sequential(
                nn.Linear(self.input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


# オプティマイザビルダーのクラス
class OptimizerBuilder:
    def __init__(
        self,
        model_parameters,
        learning_rate: float
    ):
        self.model_parameters = model_parameters
        self.learning_rate = learning_rate

    def build(
        self,
        optimizer_type: str = "AdamW"
    ):
        if optimizer_type == "AdamW":
            return optim.AdamW(self.model_parameters, lr=self.learning_rate)
        elif optimizer_type == "SGD":
            return optim.SGD(self.model_parameters, lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


# スケジューラビルダーのクラス
class SchedulerBuilder:
    def __init__(
        self,
        optimizer
    ):
        self.optimizer = optimizer

    def build(
        self,
        scheduler_type: str = "None",
        step_size: int = 1,
        gamma: float = 0.1
    ):
        if scheduler_type == "StepLR":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
        elif scheduler_type == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        elif scheduler_type == "None":
            return None
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


# 実験のランナークラス
class ExperimentRunner:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion=nn.BCELoss()
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def train(self, dataloader, max_epochs: int = 5):
        mlflow.start_run()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            for inputs, labels in dataloader:
                labels = labels.float().unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss}')
            mlflow.log_metric("loss", avg_loss, step=epoch)

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

        mlflow.pytorch.log_model(self.model, "model")
        mlflow.end_run()


# # 使い方
# # ダミーデータの作成
# data_dict = {
    # 'feature1': [1.0, 2.0, 3.0],
    # 'feature2': [4.0, 5.0, 6.0],
    # 'label': [0, 1, 0]
# }
# data_df = pd.DataFrame(data_dict)
# feature_columns = ['feature1', 'feature2']
# label_column = 'label'

# # データの前処理
# scaler = StandardScaler()
# data_df[feature_columns] = scaler.fit_transform(data_df[feature_columns])

# # データセットとデータローダーの準備
# dataset = TableDataset(data_df, label_column, feature_columns)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # モデル、オプティマイザ、スケジューラの作成
# model_builder = ModelBuilder(input_size=len(feature_columns))
# model = model_builder.build(model_type="simple")

# optimizer_builder = OptimizerBuilder(model.parameters(), learning_rate=0.001)
# optimizer = optimizer_builder.build(optimizer_type="AdamW")

# scheduler_builder = SchedulerBuilder(optimizer)
# scheduler = scheduler_builder.build(scheduler_type="None")

# # 実験の実行
# runner = ExperimentRunner(model, optimizer, scheduler)
# runner.train(dataloader, max_epochs=5)
