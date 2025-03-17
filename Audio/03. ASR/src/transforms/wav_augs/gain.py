import torch_audiomentations
from torch import Tensor, nn


# class Gain(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self._aug = torch_audiomentations.Gain(*args, **kwargs)

#     def __call__(self, data: Tensor):
#         x = data.unsqueeze(1)
#         return self._aug(x).squeeze(1)

class Gain(nn.Module):
    def __init__(self, *args, **kwargs):
        # Добавляем параметр output_type='dict' в инициализацию
        kwargs['output_type'] = 'dict'
        super().__init__()
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        # Получаем результат через .samples
        return self._aug(x)['samples'].squeeze(1)
