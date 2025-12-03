import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skorch import NeuralNetClassifier
import skrub

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 5 * 5)
        feat = F.relu(self.fc1(x))   # feature vector
        out = self.fc2(feat)
        return out

df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)
df = df.skb.subsample(n=300)

X = df.drop("label", axis=1).skb.mark_as_X()
y = df["label"].values.skb.mark_as_y()
X = X.values.reshape(-1, 1, 28, 28).astype("float32") / 255.0

net = NeuralNetClassifier(
    module=Net,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    optimizer__lr=1e-3,
    max_epochs=5,
    batch_size=64,
    device="cpu"
)

pred = X.skb.apply(net, y=y)

splits = pred.skb.train_test_split(test_size=0.2, random_state=42)
learner = pred.skb.make_learner()
print("\nTraining model...\n")
learner.fit(splits["train"])
score = learner.score(splits["test"])
print("\nValidation accuracy:", score)