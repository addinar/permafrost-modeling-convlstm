import torch
import torch.nn as nn

class ConvLSTM(nn.Module):
    def __init__(self, input_features, conv_out_channels=64, lstm_hidden=64, output_features=4):
        super(ConvLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=conv_out_channels, kernel_size=7, stride=1, padding=3)
        self.batch_norm = nn.BatchNorm1d(conv_out_channels)
        self.lstm = nn.LSTM(input_size=conv_out_channels, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, output_features)
        self.h0 = nn.Parameter(torch.zeros(1, lstm_hidden))
        self.c0 = nn.Parameter(torch.zeros(1, lstm_hidden))

    def forward(self, x):
        x = x.permute(0, 2, 1)    # [batch_size, features, seq_length]
        x = self.conv1(x)          # [batch_size, conv_out_channels, seq_length] 
        x = self.batch_norm(x)     # Apply BatchNorm
        x = x.permute(0, 2, 1)     # [batch_size, seq_length//2, conv_out_channels]

        batch_size = x.size(0)
        h0 = self.h0.expand(1, batch_size, -1).contiguous()
        c0 = self.c0.expand(1, batch_size, -1).contiguous()

        _, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.fc(hn.squeeze(0))
        return out