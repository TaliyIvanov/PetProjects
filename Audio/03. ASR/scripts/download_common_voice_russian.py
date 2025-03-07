# для запуска скрипта использовать следующую команду в терминале
# PYTHONPATH=$(pwd) python scripts/download_common_voice_russian.py


from src.project_datasets.common_voice_ru import CommonVoiceDataset_RU

# Загружаем train-сплит
dataset = CommonVoiceDataset_RU(split="train")

print("✅ Метаданные успешно загружены!")