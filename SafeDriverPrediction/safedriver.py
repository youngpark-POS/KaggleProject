'''
Safe Driver Prediction

Date: Jun 06 2021
'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


max_epochs = 1
learning_rate = 1e-3


def unroll_categorical_features(df):
    new_df = pd.DataFrame()
    df_length = len(temp_df["id"])
    for col, item in df.iteritems():
        if col.split('_')[-1] == "cat":
            onehot_df = pd.get_dummies(
                item, columns=[f'{col}_{i}' for i in range(max(item) - min(item) + 1)]
            ).iloc[:, 1:]
            new_df = pd.concat([new_df, onehot_df], axis=1)
        else:
            new_df = pd.concat([new_df, item], axis=1)
    return new_df


temp_df = pd.read_csv("train.csv")
new_df = unroll_categorical_features(temp_df)
train_df_size = int(len(new_df["id"]) * 0.8)
test_df_size = len(new_df["id"]) - train_df_size
train_df = new_df.iloc[:train_df_size]
test_df = new_df.iloc[train_df_size:]


class SafeDriverModel(nn.Module):
    def __init__(self, in_feature):
        super(SafeDriverModel, self).__init__()
        self.linear1 = nn.Linear(in_feature, 300)
        self.linear2 = nn.Linear(300, 30)
        self.linear3 = nn.Linear(30, 1)
        self.net = nn.Sequential(
            self.linear1, nn.ReLU(), self.linear2, nn.ReLU(),
            self.linear3, nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class SafeDriverDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.x_data = train_df.iloc[:, 2:]
        self.y_data = train_df["target"]

    def __getitem__(self, idx):
        x_item = torch.Tensor(np.array(self.x_data.iloc[idx]))
        y_item = torch.zeros(1) if self.y_data[idx] == 0 else torch.ones(1)
        return x_item, y_item

    def __len__(self):
        return len(self.y_data)


if __name__ == "__main__":

    safeDriverDataset = SafeDriverDataset()
    safeDriverLoader = DataLoader(safeDriverDataset, batch_size=128, shuffle=True)

    safeDriverModel = SafeDriverModel(len(train_df.columns) - 2)
    optimizer = optim.Adam(safeDriverModel.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for epoch in range(1, max_epochs + 1):
        for i, (x, y) in enumerate(safeDriverLoader, start=1):
            prediction = safeDriverModel(x)
            cost = criterion(prediction, y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            #  if i % 1000 == 0:
            print(f'Epoch {epoch:2d} Iter {i:5d} Cost {cost:.5f}')

    with torch.no_grad():
        loss = 0
        #  answer = pd.DataFrame(columns=["id", "target"])
        for idx, row in test_df.iterrows():
            x = torch.FloatTensor(row[test_df.columns[2]:])
            prediction = safeDriverModel(x)
            #  answer = pd.concat([answer, pd.Series({"id": idx, "target": row["target"]})], axis=1)
            loss += (prediction - row["target"]) ** 2
        print(f'Average loss {(loss / test_df_size):.5f}')
        #  answer.to_csv(path_or_buf="/answer.csv", index=False)



