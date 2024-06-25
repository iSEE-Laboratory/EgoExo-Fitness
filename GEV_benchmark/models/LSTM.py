import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(LSTM, self).__init__()
        # defining encoder LSTM layers
        self.encoder = nn.Sequential(
            nn.Linear(800, 800),
            nn.Dropout(dropout_rate),
            nn.ReLU(True),
            nn.Linear(800, 1024),
            nn.Dropout(dropout_rate),
            nn.ReLU(True)
        )
        self.rnn = nn.LSTM(1024, 512, 2, batch_first=True)
        # self.fc_final_score = nn.Linear(256,1)

    def forward(self, x):
        state = None
        x = self.encoder(x)
        lstm_output, state = self.rnn(x, state)
        # final_score = self.fc_final_score(lstm_output[:,-1,:])
        return lstm_output[:,-1,:]
