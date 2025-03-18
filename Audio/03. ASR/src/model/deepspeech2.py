import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSpeech2(nn.Module):
    def __init__(self, n_tokens, rnn_hidden_size=512, num_rnn_layers=3):
        super(DeepSpeech2, self).__init__()

        # 2D Convolutional Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(41,11), stride=(2,2), padding=(20,5))
        self.conv2 = nn.Conv2d(32,32, kernel_size=(21,11), stride=(2,1), padding=(10,5))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(32)

        # RNN Layers (GRU)
        self.rnn = nn.GRU(input_size=32 * (161 // 4),  # Output from CNN
                          hidden_size=rnn_hidden_size,
                          num_layers=num_rnn_layers,
                          batch_first=True,
                          bidirectional=True)
        
        # Fully Connected Layer
        self.fc = nn.Linear(rnn_hidden_size * 2, n_tokens) # 2x for bidirectional GRU

    def forward(self, x):
        # Convolutional Layers
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))

        # Reshape for RNN
        batch_size, channels, freq_dim, time_dim = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_dim, -1)

        # RNN Layers
        x, _ = self.rnn(x)

        # Fully Connected Layer
        x = self.fc(x)

        return x