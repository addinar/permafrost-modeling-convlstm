import torch.nn as nn

class ConvLSTM(nn.Module):
    def __init__(self, input_features, conv_out_channels=64, lstm_hidden=64, output_features=4):
        super(ConvLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=conv_out_channels, kernel_size=2, stride=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.lstm = nn.LSTM(input_size=conv_out_channels, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, output_features)

    def forward(self, x): # make sure x input has shape [batch_size, seq_length, features] before permuting
        x = x.permute(0, 2, 1) # [batch_size, features, seq_length]
        x = self.conv1(x) # [batch_size, conv_out_channels, reduced_seq_length]
        x = self.pool(x) # [batch_size, conv_out_channels, pooled_seq_length]
        x = x.permute(0, 2, 1) # [batch_size, pooled_seq_length, conv_out_channels]
        x, _ = self.lstm(x) # [batch_size, pooled_seq_length, lstm_hidden]
        x = x[:, -1, :] # [batch_size, lstm_hidden]
        x = self.fc(x) # [batch_size, output_features]
        return x