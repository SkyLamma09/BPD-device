import torch.nn as nn
import torch.nn.functional as func
import torch
from synthetic_data import load_training_data
from pathlib import Path

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resfunction1 = nn.Conv1d(12, 16, kernel_size=1)
        self.resfunction2 = nn.Conv1d(16, 32, kernel_size=1)
        self.resfunction3 = nn.Conv1d(32, 64, kernel_size=1)
        self.resfunction4 = nn.Conv1d(64, 64, kernel_size=1)
        
        self.conv1 = nn.Conv1d(12, 16, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, dilation=4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, dilation=8)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, dilation=16)
        
        self.line = nn.Linear(64, 64)
        self.line2 = nn.Linear(64, 32)
        self.line3 = nn.Linear(32, 3)

    def forward(self, inputs):
        featmap1 = torch.relu(self.conv1(func.pad(inputs, (2, 0))) + self.resfunction1(inputs))
        featmap2 = torch.relu(self.conv2(func.pad(featmap1, (8, 0))) +  self.resfunction2(featmap1))
        featmap3 = torch.relu(self.conv3(func.pad(featmap2, (16, 0))) + self.resfunction3(featmap2))
        featmap4 = torch.relu(self.conv4(func.pad(featmap3, (32, 0))) + self.resfunction4(featmap3))

        pooled = func.adaptive_avg_pool1d(featmap4, 1)
        flattened = torch.flatten(pooled, start_dim=1)

        lined = torch.relu(self.line(flattened))
        lined2 = torch.relu(self.line2(lined))

        return self.line3(lined2)
def trainer():
    x, y = load_training_data()
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long).reshape(-1)


    model = TCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(20):
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
    model_path = Path(__file__).resolve().parent / "model" / "tcn.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
if __name__ == "__main__":
    trainer()
