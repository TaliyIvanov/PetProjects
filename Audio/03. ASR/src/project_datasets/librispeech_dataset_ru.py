# """
# librispeech_dataset_ru.py — загрузка и обработка датасета **Librispeech на русском языке**.
# - Автоматически скачивает и распаковывает данные (если их нет).
# - Создает индекс из аудиофайлов и их транскрипций.
# - Хранит обработанный индекс в JSON для ускоренного доступа.
# """

import json
import os
import shutil
from pathlib import Path

import torchaudio
import wget
from tqdm import tqdm

from src.project_datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "ruls_data": "https://openslr.trmal.net/resources/96/ruls_data.tar.gz",
}

class LibrispeechDataset_RU(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == "train_all"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "librispeech_russian"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        if part == "train_all":
            index = sum(
                [
                    self._get_or_load_index(part)
                    for part in URL_LINKS
                    if "train" in part
                ],
                [],
            )
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        wget.download(URL_LINKS[part], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        
        # Проверяем, существует ли папка LibriSpeech
        libri_dir = self._data_dir / "LibriSpeech"
        if libri_dir.exists():
            for fpath in libri_dir.iterdir():
                shutil.move(str(fpath), str(self._data_dir / fpath.name))
            shutil.rmtree(str(libri_dir))
        else:
            print("Папки 'LibriSpeech' нет, данные уже разархивированы корректно.")
        os.remove(str(arch_path))  # Удаляем скачанный архив



    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        split_dir = self._data_dir / part
        
        # Если индекс существует и данные присутствуют, возвращаем индекс
        if index_path.exists() and split_dir.exists() and any(split_dir.iterdir()):
            with index_path.open() as f:
                index = json.load(f)
            return index

        # Если данных нет, скачиваем и создаем индекс
        print(f"Файл индекса или данные не найдены. Начинаем загрузку части: {part}")
        index = self._create_index(part)
        with index_path.open("w") as f:
            json.dump(index, f, indent=2)
        return index


    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
            list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index