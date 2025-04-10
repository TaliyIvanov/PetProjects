# library for fast quick calculation of edit distance
import editdistance
# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1
    return editdistance(target_text.split(), predicted_text.split())/len(target_text.split())


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1
    return editdistance.eval(target_text, predicted_text)/len(target_text)
