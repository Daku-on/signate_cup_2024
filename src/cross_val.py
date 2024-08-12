import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np


class CrossValidator:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_folds: int = 5,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        """
        クロスバリデーション用のクラス

        Args:
            X (np.ndarray): 特徴量データ
            y (np.ndarray): ラベルデータ
            k_folds (int): フォールド数
            batch_size (int): バッチサイズ
            shuffle (bool): データシャッフルの有無
            random_state (int): ランダムシード
        """
        self.X = X
        self.y = y
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.kf = KFold(
            n_splits=k_folds,
            shuffle=shuffle,
            random_state=random_state,
        )

    def get_fold_data(self, fold_idx: int):
        """
        指定されたフォールドに対応するトレーニングデータとバリデーションデータを取得

        Args:
            fold_idx (int): フォールドのインデックス

        Returns:
            train_loader (DataLoader): トレーニングデータのDataLoader
            val_loader (DataLoader): バリデーションデータのDataLoader
        """
        train_idx, val_idx = list(self.kf.split(self.X))[fold_idx]
        X_train, X_val = self.X[train_idx], self.X[val_idx]
        y_train, y_val = self.y[train_idx], self.y[val_idx]

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32), 
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader
