import torch
from torch import nn
from torch.nn.init as init

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
        self.activation = nn.Hardtahn()
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
     pass