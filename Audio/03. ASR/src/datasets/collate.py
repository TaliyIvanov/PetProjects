import torch

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version of the tensors.
    """

    # Извлекаем данные из списка словарей
    audios, spectrograms, texts, text_encodeds, audio_paths = zip(
        *[(item["audio"], item["spectrogram"], item["text"], item["text_encoded"], item["audio_path"]) for item in dataset_items]
    )

    # Определяем длины спектрограмм
    spectrogram_lengths = torch.tensor([spectrogram.shape[-1] for spectrogram in spectrograms], dtype=torch.long)

    # Создаем тензор для хранения паддированных спектрограмм
    freq_size = spectrograms[0].shape[0]  # Размер частотного спектра (обычно 128)
    max_length = max(spectrogram_lengths)  # Максимальная длина по времени
    spectrograms_padded = torch.zeros(len(spectrograms), freq_size, max_length)

    # Паддинг спектрограмм
    for i, spectrogram in enumerate(spectrograms):
        spectrograms_padded[i, :, :spectrogram.shape[-1]] = spectrogram

    # Определяем длины текстов
    text_lengths = torch.tensor([len(text) for text in text_encodeds], dtype=torch.long)

    # Создаем тензор для хранения паддированных текстов
    max_text_length = max(text_lengths)
    text_padded = torch.zeros(len(text_encodeds), max_text_length, dtype=torch.long)

    # Паддинг текстов
    for i, text in enumerate(text_encodeds):
        text_padded[i, :len(text)] = torch.tensor(text, dtype=torch.long)

    # Формируем батч в виде словаря
    return {
        'audio': audios,                                 # Сырые аудиоданные (например, волны)
        'spectrogram': spectrograms_padded,               # Паддированные спектрограммы
        'spectrogram_length': spectrogram_lengths,        # Длины спектрограмм
        'text': texts,                                    # Исходные текстовые строки
        'text_encoded': text_padded,                       # Паддированные закодированные тексты
        'text_encoded_length': text_lengths,               # Длины закодированных текстов
        'audio_path': audio_paths,                         # Пути к аудиофайлам
    }

