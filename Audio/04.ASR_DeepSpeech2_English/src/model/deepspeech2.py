import torch
from torch import nn
import torch.nn.init as init

class CNNLayer(nn.Module):
    """
    Свёрточный слой с пакетной нормализацией и функцией активации Hardtanh.

    Args:
        in_channels (int): Количество входных каналов.
        out_channels (int): Количество выходных каналов.
        kernel_size (int or tuple): Размер ядра свёртки.
        stride (int or tuple): Шаг свёртки.
        padding (int or tuple, optional): Отступ. По умолчанию 0.

    Attributes:
        cnn (nn.Conv2d): Свёрточный слой.
        batch_norm (nn.BatchNorm2d): Слой пакетной нормализации.
        activation (nn.Hardtanh): Функция активации Hardtanh.

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(CNNLayer, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(num_features = out_channels)
        self.activation = nn.Hardtanh()
        # Инициализация весов свёрточного слоя с помощью Xavier initialization
        init.xavier_normal_(self.cnn.weight, gain=1)
        # Инициализация смещения свёрточного слоя нулями
        if self.cnn.bias is not None:
            init.constant_(self.cnn.bias, 0)

    def calc_new_sequence_length(self, sequence_lengths):
        """
        Вычисляет новую длину последовательности после применения свёртки.

        Args:
            sequence_lengths (list or tensor): Длины входных последовательностей.

        Returns:
            torch.Tensor: Новые длины последовательностей.
        """
        p = self.cnn.padding[1]  # Отступ по высоте/ширине (второй элемент padding)
        k = self.cnn.kernel_size[1] # Размер ядра по высоте/ширине (второй элемент kernel_size)
        s = self.cnn.stride[1] # Шаг свёртки по высоте/ширине (второй элемент stride)
        sequence_lengths = torch.tensor(sequence_lengths)  # Преобразование в тензор для вычислений
        sequence_lengths = (sequence_lengths + (2*p) - k) // s + 1 # Стандартная формула для расчёта длины последовательности после свёртки
        return torch.clamp(sequence_lengths, min=1) # Ограничение минимальной длины последовательности до 1 и преобразование в int
    
    def forward(self, x, sequence_lengths):
        """
        Прямой проход слоя.

        Args:
            x (torch.Tensor): Входной тензор.
            sequence_length (list or tensor): Длины входных последовательностей.

        Returns:
            tuple: Кортеж, содержащий выходной тензор и новые длины последовательностей.
        """
        new_sequence_lengths = self.calc_new_sequence_length(sequence_lengths)
        x = self.cnn(x)
        x = self.batch_norm(x)
        out = self.activation(x)
        return out, new_sequence_lengths


class RNNLayer(nn.Module):
    """
    Двунаправленный LSTM слой с dropout, нормализацией слоя и функцией активации Hardtanh.

    Args:
        in_channels (int): Размерность входных данных (размер эмбеддинга).
        hidden_units (int): Количество скрытых юнитов в LSTM.

    Attributes:
        rnn (nn.LSTM): Двунаправленный LSTM слой.
        dropout (nn.Dropout): Слой dropout.
        layer_norm (nn.LayerNorm): Слой нормализации.
        activation (nn.Hardtanh): Функция активации Hardtanh.
    """
    def __init__(self, in_channels, hidden_units):
        super(RNNLayer, self).__init__()
        self.rnn = nn.LSTM(in_channels, hidden_units, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(hidden_units * 2)  # hidden_units * 2 because of bidirectional LSTM
        self.activation = nn.Hardtanh()
        self._apply_xavier_initialization()

    def _apply_xavier_initialization(self):
        """
        Инициализирует веса LSTM слоя с помощью Xavier initialization.
        """
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:  # input-hidden weights
                init.xavier_normal_(param, gain=1)
            elif 'weight_hh' in name:  # hidden-hidden weights
                init.xavier_normal_(param, gain=1)
            elif 'bias' in name:
                init.constant_(param, 0)

    def forward(self, x, sequence_lengths):
        """
        Прямой проход слоя.

        Args:
            x (torch.Tensor): Входной тензор размерности (batch_size, sequence_length, in_channels).
            sequence_lengths (torch.Tensor): Тензор с длинами последовательностей в батче.

        Returns:
            torch.Tensor: Выходной тензор размерности (batch_size, sequence_length, hidden_units * 2).
        """
        # Упаковка последовательностей для эффективной обработки LSTM, удаляя паддинг.
        # sequence_lengths.cpu() is used because pack_padded_sequence expects CPU tensors for lengths.
        input_packed = nn.utils.rnn.pack_padded_sequence(x, sequence_lengths.cpu(), enforce_sorted=False, batch_first=True)
        
        try:
            output, hidden_states = self.rnn(input_packed)
        except RuntimeError as e:
            print("Input Packed:", x)  # Print input tensor for debugging if RuntimeError occurs
            raise e

        # Восстановление паддинга после LSTM.
        x, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=x.shape[1], batch_first=True)

        x = self.dropout(x)
        x = self.layer_norm(x)
        out = self.activation(x)
        return out


class DeepSpeech2(nn.Module):
    """
    Implementation of the DeepSpeech2 architecture for speech recognition.

    This model uses a combination of convolutional and recurrent layers to process
    speech input and predict character probabilities.
    """

    def __init__(self, input_features):
        """
        Initializes the DeepSpeech2 model.

        Args:
            input_features (int): Number of input features (e.g., Mel-frequency cepstral coefficients).
        """
        super(DeepSpeech2, self).__init__()

        # Convolutional layers
        self.cnn_layers = nn.ModuleList([
            CNNLayer(1, 32, (11, 11), (2, 2)),  # First CNN layer with stride (2,2) for downsampling
            CNNLayer(32, 32, (11, 11), (1, 1), padding=(5, 0)),  # Subsequent CNN layers with stride (1,1) and padding
            CNNLayer(32, 32, (11, 11), (1, 1), padding=(5, 0)),
        ])

        # Calculate feature dimension after CNN layers
        features = (input_features - 11) // 2 + 1  # Feature dimension calculation after the first CNN layer with stride (2,2)
        self.features = features

        # Recurrent layers (Bidirectional LSTMs)
        self.rnn_layers = nn.ModuleList([
            RNNLayer(
                in_channels=64 * self.features if i == 0 else 330 * 2,  # Input channels depend on the layer (first layer takes CNN output)
                hidden_units=330  # Number of hidden units in each LSTM layer
            ) for i in range(3)  # 3 Bidirectional LSTM layers
        ])

        # Classifier layer
        self.classifier = nn.Sequential(nn.Linear(330 * 2, 28))  # Linear layer to map RNN output to vocabulary size (assumed to be 28)

    def forward(self, x, sequence_lengths):
        """
        Performs a forward pass through the DeepSpeech2 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features, time).
            sequence_lengths (torch.Tensor): Lengths of the input sequences.

        Returns:
            dict: A dictionary containing the log probabilities and their lengths.
        """

        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, features, time)

        # Apply CNN layers
        for cnn_layer in self.cnn_layers:
            x, sequence_lengths = cnn_layer(x, sequence_lengths)  # Pass sequence lengths through CNN layers for length adjustments

        # Reshape and transpose for RNN layers
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])  # (batch_size, channels * features, time)
        x = x.transpose(1, 2)  # (batch_size, time, channels * features)

        # Apply RNN layers
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x, sequence_lengths)  # Pass sequence lengths to handle variable-length sequences

        # Apply classifier
        x = self.classifier(x)  # (batch_size, time, vocab_size)

        # Compute log softmax and return
        return {"log_probs": nn.functional.log_softmax(x, dim=-1), "log_probs_length": sequence_lengths}