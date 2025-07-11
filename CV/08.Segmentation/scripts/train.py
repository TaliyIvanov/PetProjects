# Запуск скрипта, как модуля : python3 -m scripts.train
import hydra
from omegaconf import DictConfig

from src.trainer import train

@hydra.main(config_path="../configs", config_name="config", version_base=None)

def main(cfg: DictConfig) -> None:
    train(cfg)

if __name__ == "__main__":
    main()