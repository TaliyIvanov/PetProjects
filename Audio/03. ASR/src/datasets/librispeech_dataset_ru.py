import json
import os
import shutil
from pathlib import Path

import torchaudio
import wget
from tqdm import tqdm
import logging

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "ruls_data": "https://openslr.trmal.net/resources/96/ruls_data.tar.gz",
}

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class LibrispeechDataset_RU(BaseDataset):
#     def __init__(self, part, data_dir=None, *args, **kwargs):
#         if data_dir is None:
#             data_dir = ROOT_PATH / "data" / "librispeech_russian"
#             data_dir.mkdir(exist_ok=True, parents=True)
#         self._data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

#         # Формируем путь к части датасета
#         split_dir = self._data_dir / part
#         index = self._get_or_load_index(split_dir) # Передаем split_dir

#         super().__init__(index, *args, **kwargs)

class LibrispeechDataset_RU(BaseDataset):
    def __init__(self, split, data_dir, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "librispeech_russian"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

        # Формируем путь к части датасета
        filtered_index = []
        if split.startswith('train'):
            [item for item in self._get_or_load_index(self._data_dir / split) if 'train' in Path(item['path']).parent.name]
        elif split == 'val':
            filtered_index = [item for item in self._get_or_load_index(split) if 'dev' in Path(item['path']).parent.name]
        elif split == 'test':
            filtered_index = [item for item in self._get_or_load_index(split) if 'test' in Path(item['path']).parent.name]

        super().__init__(filtered_index, *args, **kwargs)




    # def _load_part(self, part):
    #     arch_path = self._data_dir / f"{part}.tar.gz"
    #     if not arch_path.exists():
    #         logging.info(f"Downloading {part}")
    #         wget.download(URL_LINKS[part], str(arch_path))
    #     else:
    #         logging.info(f"Archive {part} already exists, skipping download.")

    #     split_dir = self._data_dir / part
    #     if not split_dir.exists() or not any(split_dir.iterdir()): # Исправлено условие
    #         logging.info(f"Extracting {part}")
    #         shutil.unpack_archive(arch_path, self._data_dir)
    #     os.remove(arch_path)

    def _parse_manifest(self, manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)  # Теперь manifest - это список
        mapping = {}
        for item in manifest:  # Итерируемся по списку
            audio_filepath = item['audio_filepath']
            # Извлекаем имя файла из пути. Предполагается, что путь всегда начинается с "audio/"
            file_id = audio_filepath[6:].rsplit('.', 1)[0]  # or os.path.splitext(os.path.basename(audio_filepath))[0] if paths are more complex
            mapping[file_id] = item['text'] #item.get('text',"")
        return mapping

    # def _get_or_load_index(self, part):
    #     index_path = self._data_dir / f"{part}_index.json"
    #     split_dir = self._data_dir / part

    #     if index_path.exists() and split_dir.exists() and any(split_dir.iterdir()):
    #         with index_path.open() as f:
    #             return json.load(f)

    #     logging.info(f"Creating index for {part}")
    #     index = self._create_index(split_dir)
    #     with index_path.open("w") as f:
    #         json.dump(index, f, indent=2)
    #     return index

    def _get_or_load_index(self, split_dir): # Принимаем полный путь к директории split
        index_path = split_dir / "index.json" # index.json в директории split

        if index_path.exists() and any(split_dir.iterdir()): # Проверяем существование index.json и наличие файлов в директории split
            with index_path.open() as f:
                return json.load(f)

        logging.info(f"Creating index for {split_dir.name}") # Выводим имя директории split
        index = self._create_index(split_dir)
        with index_path.open("w") as f:
            json.dump(index, f, indent=2)
        return index
    

    def _create_index(self, split_dir):
        index = []
        for audio_file in split_dir.glob("*/*/*.flac"): # Предполагаем, что аудиофайлы находятся в поддиректориях split_dir
            text_file = audio_file.parent / f"{audio_file.stem}.txt"
            with text_file.open() as f:
                text = f.read().strip()

            t_info = torchaudio.info(str(audio_file))
            audio_len = t_info.num_frames / t_info.sample_rate
            index.append({
                "path": str(audio_file),
                "text": text,
                "audio_len": audio_len
            })

        return index

    def _create_index(self, split_dir):
        index = []
        if not split_dir.exists():
            logging.error(f"Directory {split_dir} not found.")
            return index

        manifest_path = split_dir / "manifest.json"
        if not manifest_path.exists():
            logging.error(f"Manifest file {manifest_path} not found.")
            return index

        try:
            text_mapping = self._parse_manifest(manifest_path)
        except Exception as e:
            logging.error(f"Error parsing manifest file: {e}")
            return index


        audio_dir = split_dir / "audio"
        if not audio_dir.exists():
            logging.error(f"Audio directory {audio_dir} not found.")
            return index


        for wav_path in tqdm(list(audio_dir.glob('**/*.wav')), desc=f"Indexing {split_dir.name}"):
            f_id = wav_path.stem
            f_text = text_mapping.get(f_id)  # .get для избежания KeyError если id нет в manifest
            if f_text is None:
                logging.warning(f"Text not found for {wav_path} in manifest. Skipping.")
                continue

            try:
                t_info = torchaudio.info(str(wav_path))
                length = t_info.num_frames / t_info.sample_rate
                index.append({
                    "path": str(wav_path.absolute().resolve()),
                    "text": f_text.lower(),
                    "audio_len": length,
                })
            except Exception as e:
                logging.error(f"Error processing file {wav_path}: {e}")

        logging.info(f"Total files indexed: {len(index)}")
        return index