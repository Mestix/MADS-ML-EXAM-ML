from torch import nn
import torch.nn.functional as F

class TinyECGCNN(nn.Module):
    def __init__(self, num_classes=5, filters=16, hidden_units=64, input_size=192):
        super().__init__()
        self.conv = nn.Conv1d(1, filters, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)

        self.flatten_dim = filters * (input_size // 2)  # 192 → 96 na pooling

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class GreatECGCNN(nn.Module):
    def __init__(self, num_classes=5, filters=16, hidden_units=64, input_size=192, dropout=0.2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(filters, filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.flatten_dim = filters * (input_size // 2)  # 192 → 96 na pooling

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
    

class NotSoGreatECGCNN(nn.Module):
    def __init__(self, num_classes=5, filters=16, hidden_units=64, gru_hidden=64, dropout=0.2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(filters, filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        # self.flatten_dim = filters * (input_size // 2)  # 192 → 96 na pooling

        # GRU verwacht input van vorm (batch, tijd, features)
        self.gru = nn.GRU(input_size=filters * 2, hidden_size=gru_hidden, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, 192)
        x = self.conv(x)  # → (batch, filters*2, 48)
        x = x.permute(0, 2, 1)  # → (batch, tijd=48, features)
        _, h_n = self.gru(x)    # h_n: laatste verborgen toestand, vorm (1, batch, gru_hidden)
        x = h_n.squeeze(0)      # → (batch, gru_hidden)
        return self.fc(x)