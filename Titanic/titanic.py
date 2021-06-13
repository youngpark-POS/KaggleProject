'''
Author: youngpark-POS
Date: Jun 06 2021 up to date

Kaggle - Titanic project
'''

import numpy as np
import pandas as pd

import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

max_epochs = 30
drop_rate = 0.2
learning_rate = 1e-4


def detect_outlier(df, features):

    outlier_indices = []
    for col in features:
        Q1 = np.quantile(df[col], 0.25)
        Q3 = np.quantile(df[col], 0.75)
        outlier_step = (Q3 - Q1) * 1.5
        outlier_index = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_index)
    multiple_outliers = [key for key, value in collections.Counter(outlier_indices).items() if value > 2]
    return multiple_outliers


# data preprocessing
df_target = pd.read_csv("test.csv")
df_all = pd.read_csv("train.csv")
for frame in (df_all, df_target):
    frame.fillna(0.0)
    frame["Sex"] = pd.Series(map(lambda x: 0 if x == "male" else 1, frame["Sex"]))
    frame["Cabin"] = pd.Series(map(lambda x: 0 if x is True else 1, frame["Cabin"].isna()))

    for columnIdx in ("SibSp", "Parch", "Fare"):
        frame[columnIdx] = pd.Series(map(lambda x: x / max(frame[columnIdx]), frame[columnIdx]))

target_outlier = detect_outlier(df_all, ["SibSp", "Parch", "Fare"])
df_all.drop(target_outlier, inplace=True)
df_all.reset_index(inplace=True)


df_train = df_all.iloc[:700]
df_test = df_all.iloc[700:]


class TitanicClassifier(nn.Module):
    def __init__(self, in_features):
        super(TitanicClassifier, self).__init__()
        self.linear1 = nn.Linear(in_features, 100)
        self.linear2 = nn.Linear(100, 25)
        self.linear3 = nn.Linear(25, 10)
        self.linear4 = nn.Linear(10, 1)
        self.net = nn.Sequential(
            self.linear1, nn.ReLU(),
            self.linear2, nn.ReLU(),
            self.linear3, nn.ReLU(),
            self.linear4, nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class TitanicDataset(Dataset):
    def __init__(self):
        super(TitanicDataset, self).__init__()
        self.x_data = df_train[["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Fare"]]
        self.y_data = df_train[["Survived"]]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x_data.loc[idx]), torch.FloatTensor(self.y_data.loc[idx])

    def __len__(self):
        return self.x_data.shape[0]


if __name__ == '__main__':

    titanic_dataset = TitanicDataset()
    titanic_loader = DataLoader(dataset=titanic_dataset,
                                batch_size=4,
                                shuffle=True)
    model = TitanicClassifier(in_features=6)

    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(1, max_epochs + 1):
        for i, data in enumerate(titanic_loader, start=1):
            x, y = data

            prediction = model(x)
            cost = F.binary_cross_entropy(prediction, y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if i % 30 == 0:
                print(f'Epoch {epoch:2d} Iter {i:3d} Cost {cost:.6f}')

    with torch.no_grad():
        correctness = 0
        answer = pd.DataFrame(columns=["PassengerId", "Survived"])
        for idx, line in df_test.iterrows():
            x = torch.from_numpy(np.array(line[["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Fare"]], dtype=np.float32))
            prediction = model(x) > torch.FloatTensor([0.5])
            correctness += prediction.item() == bool(line["Survived"])
        print(f"Accuracy {(correctness / len(df_test)):.4f}")
        #  generate answer

        for idx, line in df_target.iterrows():
            x = torch.from_numpy(np.array(line[["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Fare"]], dtype=np.float32))
            prediction = 1 if model(x) > torch.FloatTensor([0.5]) else 0
            answer = answer.append(pd.DataFrame(
                {"PassengerId": [line["PassengerId"]], "Survived": [prediction]}))
        answer.to_csv(path_or_buf="answer.csv", index=False)


