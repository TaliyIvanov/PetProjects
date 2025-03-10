# для запуска скрипта использовать следующую команду в терминале
# PYTHONPATH=$(pwd) python scripts/download_librispeech_russian.py


from src.project_datasets.librispeech_dataset_ru import LibrispeechDataset_RU

# Загружаем train-сплит
dataset = LibrispeechDataset_RU(part="ruls_data")

print("Метаданные Librispeech успешно загружены!")


"""
После загрузки должно быть подобное сообщение:

(ASRvenv) talium@taliumPC:/media/talium/1DA5AE943A305AF1/DataSciense/Projects/PetProjects/Audio/03. ASR$ PYTHONPATH=$(pwd) python scripts/download_librispeech_russian.py
Файл индекса или данные не найдены. Начинаем загрузку части: ruls_data
Loading part ruls_data
100% [....................................................................] 9129924586 / 9129924586Папки 'LibriSpeech' нет, данные уже разархивированы корректно.
Preparing librispeech folders: ruls_data: 0it [00:00, ?it/s]
Метаданные Librispeech успешно загружены!
"""