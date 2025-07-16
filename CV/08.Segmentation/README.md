### Abstract
Task: There is a binary semantic segmentation task.
I need to train a convolutional neural network to predict a building mask using satellite images.

### Steps
- Creating a dataset using overlap slicing (i have two images 4418 ∗ 4573 pixels, i need more to train CNN);
- Using augmentations to create more data;
- Create CNN
- Train CNN
- Post-processing

### Tools
I want to study and use next tools in this project:
- docker & docker compose
- git
- fastapi
- airflow
- prometeus
- grafana

### Links
- [Object-Based Augmentation for Building Semantic Segmentation: Ventura and
Santa Rosa Case Study](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/papers/Illarionova_Object-Based_Augmentation_for_Building_Semantic_Segmentation_Ventura_and_Santa_Rosa_ICCVW_2021_paper.pdf)

### Project Structure

```
segmentation_project/
├── configs/                  # Конфигурационные файлы Hydra
│   ├── config.yaml          # Основная конфигурация (общие параметры)
│   ├── data/                # Конфигурация данных (пути, параметры разбиения)
│   │   └── dataset.yaml
│   ├── logger/              # Конфигурация мониторинга за обучением
│   │   └── cometml.yaml
│   │   └── wandb.yaml
│   ├── model/               # Конфигурация модели (тип, encoder, weights)
│   │   └── linknet.yaml
│   │   └── unet.yaml
│   ├── train/               # Конфигурация обучения (оптимизатор, lr, эпохи)
│   │   └── train.yaml
│   └── transforms/               # Конфигурация обучения (оптимизатор, lr, эпохи)
│       ├──train_transforms.yaml
│       └── val_transforms.yaml
│
├── src/                     # Исходный код
│   ├── datasets/            # Классы датасетов
│   │   ├── __init__.py
│   │   └── datasets.py      # SegmentationDataset
│   ├── logger/              # Конфигурация мониторинга за обучением
│   │   ├── __init__.py
│   │   ├── cometml.py
│   │   ├── logger.py
│   │   └── wandb.py 
│   ├── loss/                # Определения функций потерь
│   │   ├── __init__.py
│   │   └── bce.py
│   ├── metrics/             # Определения метрик
│   │   ├── __init__.py
│   │   └── iou.py
│   ├── models/              # Определения моделей
│   │   ├── __init__.py
│   │   └── linknet.py
│   ├── transforms/          # Трансформации данных
│   │   ├── __init__.py
│   │   ├── train_transform.py
│   │   └── val_transform.py
│   ├── utils/               # Вспомогательные функции
│   │   ├── __init__.py
│   │   └── utils.py         # calculate_class_weights, visualize_predictions
│   ├── __init__.py
│   ├── trainer.py             # Код обучения и валидации
│   ├── predict.py           # Код для предсказания на одном изображении (для API)
│   └── ...                  # Другие модули (например, для логирования)
├── data/                    # Данные (необходимо настроить в config/data/dataset.yaml)
│   ├── dataset/             
│   │   ├── images/
│   │   │   ├── image1.png
│   │   │   └── ...
│   │   └── masks/
│   │       ├── mask1.png
│   │       └── ...
│   └── datasources          # исходные спутниковые снимки
├── outputs/                 # Выходные данные (логи, обученные модели) - Hydra управляет
│   └── ...                  # Hydra создаст структуру внутри при запуске
├── scripts/                 # Скрипты для запуска (обучение, предсказание, API)
│   ├── __init__.py
│   ├── train.py             # Скрипт для обучения (точка входа)
│   ├── pred_image.py        # Скрипт для предсказания одного изображения # пока не разработан
│   └── api.py               # Скрипт для запуска FastAPI
├── requirements.txt          # Зависимости проекта
├── Dockerfile                # Инструкции для создания Docker-образа
├── Dockercompose.yaml        # Инструкции для создания Docker-образа
├── .dockerignore             # Файлы и папки, которые нужно исключить из образа
├── .gitignore                # Файлы и папки, которые нужно исключить из Git
└── README.md                 # Описание проекта
```

### Other

25/07/07:
[Epoch 30] Val Loss: 0.0687 | IoU: 0.7983
[Test] Loss: 0.0759 | IoU: 0.7674


### Docker
- docker compose up --build: Сначала пересобирает образ segmentation-api-dev, а затем запускает контейнер на его основе. Используйте это после изменения Dockerfile или requirements.txt.
- docker compose up: Просто запускает контейнер, используя уже существующий, ранее собранный образ. Используйте это для обычного запуска и когда вы меняли только .py файлы.
- docker compose down: Останавливает и удаляет контейнеры, сети, созданные командой up.
- после сборки контейнера убедиться, что сервер запущен: docker compose logs segmentation_service

### After build and up Docker:
http://localhost:8000/docs
http://localhost:8000/redoc