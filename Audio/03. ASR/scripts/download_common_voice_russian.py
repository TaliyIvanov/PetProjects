# для запуска скрипта использовать следующую команду в терминале
# PYTHONPATH=$(pwd) python scripts/download_common_voice_russian.py


from src.datasets.common_voice_ru import CommonVoiceDataset_RU

# Загружаем train-сплит
dataset = CommonVoiceDataset_RU(split="train")

print("Метаданные CommonVoice успешно загружены!")