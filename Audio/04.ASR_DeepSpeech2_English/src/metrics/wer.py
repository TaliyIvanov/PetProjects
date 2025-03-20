from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)

# class BeamSearchWERMetric(BaseMetric):
#     def __init__(self,  text_encoder, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.text_encoder = text_encoder

#     def __call__(
#             self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **lwargs
#             ):
#         wers = []
#         bs_results = []
#         for log_probs_line in log_probs:
#             bs_results.append(self.text_encoder.ctc_beam_search(log_probs_line.exp().detach().cpu().numpy(), 10)[0])
#         for predicted_text, target_text in zip(bs_results, text):
#             target_text = self.text_encoder.normalize_text(target_text)
#             wers.append(calc_wer(target_text, predicted_text[0]))
#         return sum(wers) / len(wers)

class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs): # исправлено kwargs
        wers = []
        bs_results = []
        for log_probs_line in log_probs:
             try:
                bs_results.append(self.text_encoder.ctc_beam_search(log_probs_line.exp().detach().cpu().numpy(), 10)[0])
             except IndexError:
                bs_results.append("")

        for predicted_text, target_text in zip(bs_results, text):
            try:
                target_text = self.text_encoder.normalize_text(target_text)
                wers.append(calc_wer(target_text, predicted_text[0] if predicted_text else ""))
            except Exception as e:
                print(f"Error during WER calculation: {e}")

        return sum(wers) / len(wers) if wers else 0.0

# class BeamSearchWERMetric(BaseMetric):
#     def __init__(self, text_encoder, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.text_encoder = text_encoder

#     def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
#         wers = []
#         bs_results = []

#         # Перебираем каждую строку в батче
#         for i, log_probs_line in enumerate(log_probs):
#             # Получаем вероятности и выполняем beam search
#             beam_search_result = self.text_encoder.ctc_beam_search(
#                 log_probs_line[:log_probs_length[i]].exp().detach().cpu().numpy(), 10
#             )[0]
#             bs_results.append(beam_search_result)

#             # Нормализуем целевой текст и вычисляем WER
#             target_text = self.text_encoder.normalize_text(text[i])
#             wers.append(calc_wer(target_text, beam_search_result[0]))

#         # Среднее значение WER
#         return sum(wers) / len(wers)


