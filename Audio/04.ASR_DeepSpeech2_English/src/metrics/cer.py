from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)

# class BeamSearchCERMetric(BaseMetric):
#     def __init__(self, text_encoder, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.text_encoder = text_encoder

#     def __call__(
#             self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
#             ):
#         cers = []
#         bs_result = []

#         for log_probs_line in log_probs:
#             bs_result.append(self.text_encoder.ctc_beam_search(log_probs_line.exp().detach().cpu().numpy(), 10)[0])
#         for predicted_text, target_text in log_probs:
#             target_text = self.text_encoder.normalize_text(target_text)
#             cers.append(calc_cer(target_text, predicted_text[0]))
#         return sum(cers) / len(cers)

class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
            self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
            ):
        cers = []
        bs_result = []

        for log_probs_line in log_probs:
            bs_result.append(self.text_encoder.ctc_beam_search(log_probs_line.exp().detach().cpu().numpy(), 10)[0])

        for predicted_text, target_text in zip(bs_result, text): # Используем zip с text
            target_text = self.text_encoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, predicted_text[0])) # predicted_text уже содержит результат beam search

        return sum(cers) / len(cers)

# class BeamSearchCERMetric(BaseMetric):
#     def __init__(self, text_encoder, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.text_encoder = text_encoder

#     def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
#         cers = []
#         bs_result = []

#         # Перебираем каждую строку в батче
#         for i, log_probs_line in enumerate(log_probs):
#             # Получаем вероятности и выполняем beam search
#             beam_search_result = self.text_encoder.ctc_beam_search(
#                 log_probs_line[:log_probs_length[i]].exp().detach().cpu().numpy(), 10
#             )[0]
#             bs_result.append(beam_search_result)

#             # Нормализуем целевой текст и вычисляем CER
#             target_text = self.text_encoder.normalize_text(text[i])
#             cers.append(calc_cer(target_text, beam_search_result[0]))

#         # Среднее значение CER
#         return sum(cers) / len(cers)
