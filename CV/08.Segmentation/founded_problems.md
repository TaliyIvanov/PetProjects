talium@Talium:/media/talium/1DA5AE943A305AF1/DataSciense/Projects/PetProjects/CV/08.Segmentation$ ruff check .
warning: The top-level linter settings are deprecated in favour of their counterparts in the `lint` section. Please update the following options in `pyproject.toml`:
  - 'ignore' -> 'lint.ignore'
  - 'select' -> 'lint.select'
main.py:1:32: W292 [*] No newline at end of file
  |
1 | print('Docker container START')
  |                                ^ W292
  |
  = help: Add trailing newline

preparation/crop.py:1:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 | / import cv2
2 | | import os
  | |_________^ I001
3 |
4 |   # Load the image
  |
  = help: Organize imports

preparation/crop.py:2:8: F401 [*] `os` imported but unused
  |
1 | import cv2
2 | import os
  |        ^^ F401
3 |
4 | # Load the image
  |
  = help: Remove unused import: `os`

preparation/crop.py:43:26: W292 [*] No newline at end of file
   |
41 | # cv2.imshow('Patches', img)
42 | # cv2.waitKey(0)
43 | # cv2.destroyAllWindows()
   |                          ^ W292
   |
   = help: Add trailing newline

preparation/mean_std.py:1:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 | / import os
2 | | from PIL import Image
3 | | from torchvision import transforms
4 | | from torch.utils.data import Dataset, DataLoader
5 | | import torch
6 | | import numpy as np
  | |__________________^ I001
7 |
8 |   # class
  |
  = help: Organize imports

preparation/mean_std.py:5:8: F401 [*] `torch` imported but unused
  |
3 | from torchvision import transforms
4 | from torch.utils.data import Dataset, DataLoader
5 | import torch
  |        ^^^^^ F401
6 | import numpy as np
  |
  = help: Remove unused import: `torch`

preparation/mean_std.py:6:17: F401 [*] `numpy` imported but unused
  |
4 | from torch.utils.data import Dataset, DataLoader
5 | import torch
6 | import numpy as np
  |                 ^^ F401
7 |
8 | # class
  |
  = help: Remove unused import: `numpy`

preparation/mean_std.py:52:4: W292 [*] No newline at end of file
   |
50 | Mean: tensor([0.3527, 0.3395, 0.2912])
51 | Std: tensor([0.1384, 0.1237, 0.1199])
52 | """
   |    ^ W292
   |
   = help: Add trailing newline

preparation/merge_channels_opencv.py:2:17: F401 [*] `numpy` imported but unused
  |
1 | import cv2
2 | import numpy as np
  |                 ^^ F401
3 | from src.utils.utils import normalize
  |
  = help: Remove unused import: `numpy`

preparation/merge_channels_opencv.py:12:16: W291 [*] Trailing whitespace
   |
10 | output_path = 'data/data_sources/santa_rosa/santa_rosa_rgb_opencv.png'
11 |
12 | # read channels 
   |                ^ W291
13 | red = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
14 | green = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)
   |
   = help: Remove trailing whitespace

preparation/merge_channels_opencv.py:21:29: W291 [*] Trailing whitespace
   |
20 | # visualise if need to check
21 | # cv2.imshow("Image", image) 
   |                             ^ W291
22 | # cv2.waitKey(0)
23 | # cv2.destroyAllWindows()
   |
   = help: Remove trailing whitespace

preparation/merge_channels_opencv.py:41:16: W291 [*] Trailing whitespace
   |
39 | output_path = 'data/data_sources/ventura/ventura_rgb_opencv.png'
40 |
41 | # read channels 
   |                ^ W291
42 | red = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
43 | green = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)
   |
   = help: Remove trailing whitespace

preparation/merge_channels_opencv.py:50:29: W291 [*] Trailing whitespace
   |
49 | # visualise if need to check
50 | # cv2.imshow("Image", image) 
   |                             ^ W291
51 | # cv2.waitKey(0)
52 | # cv2.destroyAllWindows()
   |
   = help: Remove trailing whitespace

preparation/merge_channels_opencv.py:59:34: W292 [*] No newline at end of file
   |
57 | # save
58 | cv2.imwrite(output_path, image_norm)
59 | print('ventura_rgb_opencv saved')
   |                                  ^ W292
   |
   = help: Add trailing newline

preparation/merge_channels_var1.py:1:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 | / import rasterio
2 | | import numpy as np
3 | | import matplotlib.pyplot as plt
4 | | from PIL import Image
5 | | from src.utils.utils import normalize
  | |_____________________________________^ I001
6 |
7 |   # santarosa
  |
  = help: Organize imports

preparation/merge_channels_var1.py:3:29: F401 [*] `matplotlib.pyplot` imported but unused
  |
1 | import rasterio
2 | import numpy as np
3 | import matplotlib.pyplot as plt
  |                             ^^^ F401
4 | from PIL import Image
5 | from src.utils.utils import normalize
  |
  = help: Remove unused import: `matplotlib.pyplot`

preparation/merge_channels_var1.py:67:27: W292 [*] No newline at end of file
   |
65 | path_to_save = 'data/data_sources/ventura/ventura_rgb.png'
66 | Image.fromarray(image_norm).save(path_to_save)
67 | print('ventura_rgb saved')
   |                           ^ W292
   |
   = help: Add trailing newline

preparation/visualize_augmentations.py:3:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 |   # to check augmentations
2 |
3 | / import os
4 | | from glob import glob
5 | | from src.transforms.val_test_transforms import train_transforms
6 | | from src.datasets.datasets import SegmentationDataset
7 | | from src.utils.utils import visualize_segmentation
  | |__________________________________________________^ I001
8 |
9 |   # Путь к данным
  |
  = help: Organize imports

preparation/visualize_augmentations.py:15:100: E501 Line too long (105 > 99)
   |
13 | # Получаем пути к изображениям и маскам
14 | image_paths = sorted(glob(os.path.join(image_dir, '*.png')))  # или .jpg, если нужно
15 | mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))    # должно совпадать по имени с изображениями
   |                                                                                                    ^^^^^^ E501
16 |
17 | assert len(image_paths) == len(mask_paths), "Количество изображений и масок не совпадает"
   |

preparation/visualize_augmentations.py:20:100: E501 Line too long (105 > 99)
   |
19 | # Создаем датасет с трансформациями
20 | dataset = SegmentationDataset(image_paths=image_paths, mask_paths=mask_paths, transform=train_transforms)
   |                                                                                                    ^^^^^^ E501
21 |
22 | # Визуализируем 3 примера
   |

preparation/visualize_augmentations.py:23:50: W292 [*] No newline at end of file
   |
22 | # Визуализируем 3 примера
23 | visualize_segmentation(dataset, idx=0, samples=3)
   |                                                  ^ W292
   |
   = help: Add trailing newline

scripts/api.py:3:1: I001 [*] Import block is un-sorted or un-formatted
   |
 1 |   # for local start: uvicorn scripts.api:app --reload
 2 |
 3 | / import io
 4 | | from fastapi import FastAPI, File, UploadFile, Request
 5 | | from fastapi.responses import StreamingResponse, FileResponse
 6 | | from fastapi.staticfiles import StaticFiles
 7 | | from fastapi.middleware.cors import CORSMiddleware
 8 | | from fastapi.concurrency import run_in_threadpool
 9 | | from PIL import Image
10 | | import os
11 | | from prometheus_fastapi_instrumentator import Instrumentator
12 | | from src.predict import load_model, predict_mask
   | |________________________________________________^ I001
13 |
14 |   # configs
   |
   = help: Organize imports

scripts/api.py:58:100: E501 Line too long (127 > 99)
   |
57 | …
58 | …ile = File(..., description="Спутниковый снимок 512х512 в формате PNG/JPEG")):
   |                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ E501
59 | …
60 | …
   |

scripts/api.py:77:45: W292 [*] No newline at end of file
   |
76 | # create endpoint metrics
77 | Instrumentator().instrument(app).expose(app)
   |                                             ^ W292
   |
   = help: Add trailing newline

scripts/train.py:2:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 |   # Запуск скрипта, как модуля : python3 -m scripts.train
2 | / import hydra
3 | | from omegaconf import DictConfig
4 | |
5 | | from src.trainer import train
  | |_____________________________^ I001
6 |
7 |   @hydra.main(config_path="../configs", config_name="config", version_base=None)
  |
  = help: Organize imports

scripts/train.py:13:11: W292 [*] No newline at end of file
   |
12 | if __name__ == "__main__":
13 |     main()
   |           ^ W292
   |
   = help: Add trailing newline

src/datasets/datasets.py:1:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 | / import cv2
2 | | import torch
3 | | from torch.utils.data import Dataset
4 | | import numpy as np
  | |__________________^ I001
5 |
6 |   class SegmentationDataset(Dataset):
  |
  = help: Organize imports

src/datasets/datasets.py:2:8: F401 [*] `torch` imported but unused
  |
1 | import cv2
2 | import torch
  |        ^^^^^ F401
3 | from torch.utils.data import Dataset
4 | import numpy as np
  |
  = help: Remove unused import: `torch`

src/datasets/datasets.py:14:1: W293 [*] Blank line contains whitespace
   |
12 |     def __len__(self):
13 |         return len(self.image_paths)
14 |     
   | ^^^^ W293
15 |     def __getitem__(self, idx):
16 |         image_path = self.image_paths[idx]
   |
   = help: Remove whitespace from blank line

src/datasets/datasets.py:34:1: W293 [*] Blank line contains whitespace
   |
33 |         return image, mask
34 |     
   | ^^^^ W293
35 | __all__ = ['SegmentationDataset']
   |
   = help: Remove whitespace from blank line

src/datasets/datasets.py:35:34: W292 [*] No newline at end of file
   |
33 |         return image, mask
34 |     
35 | __all__ = ['SegmentationDataset']
   |                                  ^ W292
   |
   = help: Add trailing newline

src/logger/cometml.py:1:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 | from comet_ml import Experiment
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ I001
2 |
3 | class CometLogger:
  |
  = help: Organize imports

src/logger/cometml.py:19:30: W292 [*] No newline at end of file
   |
18 |     def end(self):
19 |         self.experiment.end()
   |                              ^ W292
   |
   = help: Add trailing newline

src/logger/cometml_from_HSE.py:1:1: N999 Invalid module name: 'cometml_from_HSE'
src/logger/cometml_from_HSE.py:3:17: F401 [*] `numpy` imported but unused
  |
1 | from datetime import datetime
2 |
3 | import numpy as np
  |                 ^^ F401
4 | import pandas as pd
  |
  = help: Remove unused import: `numpy`

src/logger/cometml_from_HSE.py:258:36: W292 [*] No newline at end of file
    |
257 |     def add_embedding(self, embedding_name, embedding):
258 |         raise NotImplementedError()
    |                                    ^ W292
    |
    = help: Add trailing newline

src/logger/logger_from_HSE.py:1:1: N999 Invalid module name: 'logger_from_HSE'
src/logger/logger_from_HSE.py:1:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 | / import logging
2 | | import logging.config
3 | | from pathlib import Path
4 | |
5 | | from src.utils.io_utils import ROOT_PATH, read_json
  | |___________________________________________________^ I001
  |
  = help: Organize imports

src/logger/logger_from_HSE.py:33:82: W292 [*] No newline at end of file
   |
31 |     else:
32 |         print(f"Warning: logging configuration file is not found in {log_config}.")
33 |         logging.basicConfig(level=default_level, filemode="a" if append else "w")
   |                                                                                  ^ W292
   |
   = help: Add trailing newline

src/logger/wandb_from_HSE.py:1:1: N999 Invalid module name: 'wandb_from_HSE'
src/logger/wandb_from_HSE.py:230:36: W292 [*] No newline at end of file
    |
229 |     def add_embedding(self, embedding_name, embedding):
230 |         raise NotImplementedError()
    |                                    ^ W292
    |
    = help: Add trailing newline

src/loss/bce.py:1:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 | import torch.nn as nn
  | ^^^^^^^^^^^^^^^^^^^^^ I001
2 |
3 | class BCEWithLogitsLoss(nn.Module):
  |
  = help: Organize imports

src/loss/bce.py:9:43: W292 [*] No newline at end of file
  |
8 |     def forward(self, preds, targets):
9 |         return self.loss_fn(preds,targets)
  |                                           ^ W292
  |
  = help: Add trailing newline

src/metrics/iou.py:1:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 | import torch
  | ^^^^^^^^^^^^ I001
2 |
3 | def compute_iou(preds: torch.Tensor, masks: torch.Tensor, threshold: float=0.5) -> float:
  |
  = help: Organize imports

src/metrics/iou.py:13:26: W292 [*] No newline at end of file
   |
13 | __all__ = ['compute_iou']
   |                          ^ W292
   |
   = help: Add trailing newline

src/models/linknet.py:1:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 | / import torch.nn as nn
2 | | import segmentation_models_pytorch as smp
  | |_________________________________________^ I001
3 |
4 |   class Linknet(nn.Module):
  |
  = help: Organize imports

src/models/linknet.py:15:29: W292 [*] No newline at end of file
   |
14 |     def forward(self, x):
15 |         return self.model(x)
   |                             ^ W292
   |
   = help: Add trailing newline

src/predict.py:1:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 | / import torch
2 | | import numpy as np
3 | | from PIL import Image
4 | | import albumentations as A
5 | | from albumentations.pytorch import ToTensorV2
6 | | import hydra
7 | | from omegaconf import OmegaConf
  | |_______________________________^ I001
8 |
9 |   # Сначала определю трансформации глобально
  |
  = help: Organize imports

src/predict.py:4:8: N812 Lowercase `albumentations` imported as non-lowercase `A`
  |
2 | import numpy as np
3 | from PIL import Image
4 | import albumentations as A
  |        ^^^^^^^^^^^^^^^^^^^ N812
5 | from albumentations.pytorch import ToTensorV2
6 | import hydra
  |

src/predict.py:54:1: W293 [*] Blank line contains whitespace
   |
52 |     input_tensor = PREDICT_TRANSFORMS(image=image_np)["image"]
53 |     input_tensor = input_tensor.unsqueeze(0).to(device) # add batch dimension
54 |     
   | ^^^^ W293
55 |     with torch.no_grad():
56 |         logits = model(input_tensor)
   |
   = help: Remove whitespace from blank line

src/predict.py:64:22: W292 [*] No newline at end of file
   |
62 |     mask_image = Image.fromarray(binary_mask, mode="L") # "L" for graysacale image
63 |
64 |     return mask_image
   |                      ^ W292
   |
   = help: Add trailing newline

src/trainer.py:1:1: I001 [*] Import block is un-sorted or un-formatted
   |
 1 | / from comet_ml import Experiment
 2 | | import torch
 3 | | import torch.nn as nn
 4 | | from torch.utils.data import DataLoader
 5 | | from sklearn.model_selection import train_test_split
 6 | | import hydra
 7 | | from hydra.utils import get_original_cwd
 8 | | from omegaconf import DictConfig, OmegaConf
 9 | | import os
10 | | from tqdm import tqdm
11 | | from typing import Dict
12 | | from glob import glob
13 | | from pathlib import Path
14 | |
15 | | # from project files
16 | | from src.datasets.datasets import SegmentationDataset
17 | | from src.metrics.iou import compute_iou
18 | | from src.utils.utils import calculate_class_weights, visualize_predictions
19 | | from src.transforms.train_transforms import train_transforms
20 | | from src.transforms.val_test_transforms import val_test_transforms
21 | | from src.models.linknet import Linknet
22 | | # from src.models.unet import Unet
23 | | from src.loss.bce import BCEWithLogitsLoss
   | |__________________________________________^ I001
24 |
25 |   def train(cfg: DictConfig) -> None:
   |
   = help: Organize imports

src/trainer.py:1:22: F401 [*] `comet_ml.Experiment` imported but unused
  |
1 | from comet_ml import Experiment
  |                      ^^^^^^^^^^ F401
2 | import torch
3 | import torch.nn as nn
  |
  = help: Remove unused import: `comet_ml.Experiment`

src/trainer.py:3:20: F401 [*] `torch.nn` imported but unused
  |
1 | from comet_ml import Experiment
2 | import torch
3 | import torch.nn as nn
  |                    ^^ F401
4 | from torch.utils.data import DataLoader
5 | from sklearn.model_selection import train_test_split
  |
  = help: Remove unused import: `torch.nn`

src/trainer.py:11:1: UP035 `typing.Dict` is deprecated, use `dict` instead
   |
 9 | import os
10 | from tqdm import tqdm
11 | from typing import Dict
   | ^^^^^^^^^^^^^^^^^^^^^^^ UP035
12 | from glob import glob
13 | from pathlib import Path
   |

src/trainer.py:13:21: F401 [*] `pathlib.Path` imported but unused
   |
11 | from typing import Dict
12 | from glob import glob
13 | from pathlib import Path
   |                     ^^^^ F401
14 |
15 | # from project files
   |
   = help: Remove unused import: `pathlib.Path`

src/trainer.py:18:29: F401 [*] `src.utils.utils.calculate_class_weights` imported but unused
   |
16 | from src.datasets.datasets import SegmentationDataset
17 | from src.metrics.iou import compute_iou
18 | from src.utils.utils import calculate_class_weights, visualize_predictions
   |                             ^^^^^^^^^^^^^^^^^^^^^^^ F401
19 | from src.transforms.train_transforms import train_transforms
20 | from src.transforms.val_test_transforms import val_test_transforms
   |
   = help: Remove unused import

src/trainer.py:18:54: F401 [*] `src.utils.utils.visualize_predictions` imported but unused
   |
16 | from src.datasets.datasets import SegmentationDataset
17 | from src.metrics.iou import compute_iou
18 | from src.utils.utils import calculate_class_weights, visualize_predictions
   |                                                      ^^^^^^^^^^^^^^^^^^^^^ F401
19 | from src.transforms.train_transforms import train_transforms
20 | from src.transforms.val_test_transforms import val_test_transforms
   |
   = help: Remove unused import

src/trainer.py:19:45: F401 [*] `src.transforms.train_transforms.train_transforms` imported but unused
   |
17 | from src.metrics.iou import compute_iou
18 | from src.utils.utils import calculate_class_weights, visualize_predictions
19 | from src.transforms.train_transforms import train_transforms
   |                                             ^^^^^^^^^^^^^^^^ F401
20 | from src.transforms.val_test_transforms import val_test_transforms
21 | from src.models.linknet import Linknet
   |
   = help: Remove unused import: `src.transforms.train_transforms.train_transforms`

src/trainer.py:20:48: F401 [*] `src.transforms.val_test_transforms.val_test_transforms` imported but unused
   |
18 | from src.utils.utils import calculate_class_weights, visualize_predictions
19 | from src.transforms.train_transforms import train_transforms
20 | from src.transforms.val_test_transforms import val_test_transforms
   |                                                ^^^^^^^^^^^^^^^^^^^ F401
21 | from src.models.linknet import Linknet
22 | # from src.models.unet import Unet
   |
   = help: Remove unused import: `src.transforms.val_test_transforms.val_test_transforms`

src/trainer.py:21:32: F401 [*] `src.models.linknet.Linknet` imported but unused
   |
19 | from src.transforms.train_transforms import train_transforms
20 | from src.transforms.val_test_transforms import val_test_transforms
21 | from src.models.linknet import Linknet
   |                                ^^^^^^^ F401
22 | # from src.models.unet import Unet
23 | from src.loss.bce import BCEWithLogitsLoss
   |
   = help: Remove unused import: `src.models.linknet.Linknet`

src/trainer.py:23:26: F401 [*] `src.loss.bce.BCEWithLogitsLoss` imported but unused
   |
21 | from src.models.linknet import Linknet
22 | # from src.models.unet import Unet
23 | from src.loss.bce import BCEWithLogitsLoss
   |                          ^^^^^^^^^^^^^^^^^ F401
24 |
25 | def train(cfg: DictConfig) -> None:
   |
   = help: Remove unused import: `src.loss.bce.BCEWithLogitsLoss`

src/trainer.py:36:1: W293 [*] Blank line contains whitespace
   |
35 |     # 2.Data
36 |     
   | ^^^^ W293
37 |     # data paths
38 |     original_cwd = get_original_cwd()
   |
   = help: Remove whitespace from blank line

src/trainer.py:51:1: W293 [*] Blank line contains whitespace
   |
49 |                                                                   test_size=cfg.data.test_size,
50 |                                                                   random_state=cfg.data.random_state)
51 |     
   | ^^^^ W293
52 |     # datasets
53 |     train_dataset = SegmentationDataset(train_imgs,
   |
   = help: Remove whitespace from blank line

src/trainer.py:68:1: W293 [*] Blank line contains whitespace
   |
66 |                               shuffle=True,
67 |                               num_workers=cfg.data.num_workers)
68 |     
   | ^^^^ W293
69 |     val_loader = DataLoader(val_dataset,
70 |                             batch_size=cfg.data.batch_size,
   |
   = help: Remove whitespace from blank line

src/trainer.py:73:1: W293 [*] Blank line contains whitespace
   |
71 |                             shuffle=False,
72 |                             num_workers=cfg.data.num_workers)
73 |     
   | ^^^^ W293
74 |     test_loader = DataLoader(test_dataset,
75 |                              batch_size=cfg.data.batch_size,
   |
   = help: Remove whitespace from blank line

src/trainer.py:103:1: W293 [*] Blank line contains whitespace
    |
101 |             preds = model(images)
102 |             loss = loss_fn(preds, masks)
103 |             
    | ^^^^^^^^^^^^ W293
104 |             # backward
105 |             optimizer.zero_grad()
    |
    = help: Remove whitespace from blank line

src/trainer.py:118:77: UP006 [*] Use `dict` instead of `Dict` for type annotation
    |
117 |     # val
118 |     def validate_epoch(model, dataloader, loss_fn, device, logger, step) -> Dict:
    |                                                                             ^^^^ UP006
119 |         model.eval()
120 |         val_loss = 0.0
    |
    = help: Replace with `dict`

src/trainer.py:124:100: E501 Line too long (102 > 99)
    |
123 |         with torch.no_grad():
124 |             for images, masks, in tqdm(dataloader, desc=f"[Epoch {step + 1}/{cfg.train.epochs}] Val"):
    |                                                                                                    ^^^ E501
125 |                 images = images.to(device)
126 |                 masks = masks.to(device)
    |

src/trainer.py:169:1: W293 [*] Blank line contains whitespace
    |
167 |         return {"test_loss": avg_test_loss, "test_iou": avg_test_iou}
168 |
169 |     
    | ^^^^ W293
170 |     # 5. Train loop
171 |     model_filename = "best_model_linknet.pth"
    |
    = help: Remove whitespace from blank line

src/trainer.py:174:9: F841 [*] Local variable `train_loss` is assigned to but never used
    |
172 |     best_iou = 0.0
173 |     for epoch in range(cfg.train.epochs):
174 |         train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, logger, epoch)
    |         ^^^^^^^^^^ F841
175 |         metrics = validate_epoch(model, val_loader, loss_fn, device, logger, epoch)
176 |         scheduler.step(metrics["val_loss"])
    |
    = help: Remove assignment to unused variable `train_loss`

src/trainer.py:181:100: E501 Line too long (100 > 99)
    |
180 |             torch.save(model.state_dict(), model_filename)
181 |             print(f"model saved to {os.getcwd()}/{model_filename} (IoU improved to {best_iou:.4f})")
    |                                                                                                    ^ E501
182 |     
183 |     best_model_state = torch.load(model_filename)
    |

src/trainer.py:182:1: W293 [*] Blank line contains whitespace
    |
180 |             torch.save(model.state_dict(), model_filename)
181 |             print(f"model saved to {os.getcwd()}/{model_filename} (IoU improved to {best_iou:.4f})")
182 |     
    | ^^^^ W293
183 |     best_model_state = torch.load(model_filename)
184 |     model.load_state_dict(best_model_state)
    |
    = help: Remove whitespace from blank line

src/trainer.py:191:17: W292 [*] No newline at end of file
    |
190 |     logger.log_metrics(tests_results, step=cfg.train.epochs)
191 |     logger.end()
    |                 ^ W292
    |
    = help: Add trailing newline

src/trainer_base.py:2:1: I001 [*] Import block is un-sorted or un-formatted
   |
 1 |   # imports
 2 | / import torch
 3 | | import torch.nn as nn
 4 | | from torch.utils.data import DataLoader
 5 | | from torch.optim.lr_scheduler import ReduceLROnPlateau
 6 | |
 7 | | from tqdm import tqdm
 8 | | import segmentation_models_pytorch as smp
 9 | | from glob import glob
10 | | import os
11 | | from sklearn.model_selection import train_test_split
12 | | import matplotlib.pyplot as plt
13 | |
14 | | # from project files
15 | | from src.datasets.datasets import SegmentationDataset
16 | | from src.metrics import compute_iou
17 | | from src.utils.utils import calculate_class_weights, visualize_predictions
18 | | from src.transforms.val_test_transforms import val_test_transform
19 | | from src.transforms.train_transforms import train_transform
   | |___________________________________________________________^ I001
20 |
21 |   # configs
   |
   = help: Organize imports

src/trainer_base.py:44:1: W293 [*] Blank line contains whitespace
   |
42 |     classes=num_classes
43 | )
44 |  
   | ^ W293
45 | model.to(device)
46 | # print(model)
   |
   = help: Remove whitespace from blank line

src/trainer_base.py:123:1: W293 [*] Blank line contains whitespace
    |
122 |     print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
123 |     
    | ^^^^ W293
124 |
125 |     # validation
    |
    = help: Remove whitespace from blank line

src/trainer_base.py:168:1: E402 Module level import not at top of file
    |
167 | # graphics
168 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ E402
169 |
170 | epochs_range = range(1, EPOCHS + 1)
    |

src/trainer_base.py:168:29: F811 [*] Redefinition of unused `plt` from line 12
    |
167 | # graphics
168 | import matplotlib.pyplot as plt
    |                             ^^^ F811
169 |
170 | epochs_range = range(1, EPOCHS + 1)
    |
    = help: Remove definition: `plt`

src/trainer_base.py:228:65: W292 [*] No newline at end of file
    |
227 | # visualize model preds
228 | visualize_predictions(model, test_loader, device, num_samples=5)
    |                                                                 ^ W292
    |
    = help: Add trailing newline

src/transforms/train_transforms.py:2:8: N812 Lowercase `albumentations` imported as non-lowercase `A`
  |
1 | # imports
2 | import albumentations as A
  |        ^^^^^^^^^^^^^^^^^^^ N812
3 | from albumentations.pytorch import ToTensorV2
  |

src/transforms/train_transforms.py:24:1: W293 [*] Blank line contains whitespace
   |
22 |         std: list = (0.1384, 0.1237, 0.1199)
23 |         ):
24 |         
   | ^^^^^^^^ W293
25 |         result = A.Compose([
26 |             A.HorizontalFlip(p=horizontal_flip_p),
   |
   = help: Remove whitespace from blank line

src/transforms/train_transforms.py:37:25: W291 [*] Trailing whitespace
   |
35 |             A.Normalize(mean=mean,
36 |                         std=std),
37 |             ToTensorV2()   
   |                         ^^^ W291
38 |             ])
   |
   = help: Remove trailing whitespace

src/transforms/train_transforms.py:39:1: W293 [*] Blank line contains whitespace
   |
37 |             ToTensorV2()   
38 |             ])
39 |         
   | ^^^^^^^^ W293
40 |         return result
   |
   = help: Remove whitespace from blank line

src/transforms/train_transforms.py:55:19: W291 [*] Trailing whitespace
   |
53 | #     A.Normalize(mean=(0.3527, 0.3395, 0.2912),
54 | #                 std=(0.1384, 0.1237, 0.1199)),
55 | #     ToTensorV2()   
   |                   ^^^ W291
56 | # ]) 
   |
   = help: Remove trailing whitespace

src/transforms/train_transforms.py:56:5: W291 [*] Trailing whitespace
   |
54 | #                 std=(0.1384, 0.1237, 0.1199)),
55 | #     ToTensorV2()   
56 | # ]) 
   |     ^ W291
   |
   = help: Remove trailing whitespace

src/transforms/train_transforms.py:56:6: W292 [*] No newline at end of file
   |
54 | #                 std=(0.1384, 0.1237, 0.1199)),
55 | #     ToTensorV2()   
56 | # ]) 
   |      ^ W292
   |
   = help: Add trailing newline

src/transforms/val_test_transforms.py:2:8: N812 Lowercase `albumentations` imported as non-lowercase `A`
  |
1 | # imports
2 | import albumentations as A
  |        ^^^^^^^^^^^^^^^^^^^ N812
3 | from albumentations.pytorch import ToTensorV2
  |

src/transforms/val_test_transforms.py:15:1: W293 [*] Blank line contains whitespace
   |
13 |         std: list = (0.1384, 0.1237, 0.1199)
14 |         ):
15 |     
   | ^^^^ W293
16 |     result = A.Compose([
17 |         A.Normalize(mean=mean, std=std),
   |
   = help: Remove whitespace from blank line

src/transforms/val_test_transforms.py:22:1: W293 [*] Blank line contains whitespace
   |
21 |     return result
22 |     
   | ^^^^ W293
   |
   = help: Remove whitespace from blank line

src/transforms/val_test_transforms.py:32:49: W292 [*] No newline at end of file
   |
30 | # ])
31 |
32 | # __all__ = ['train_transform', 'val_transform']
   |                                                 ^ W292
   |
   = help: Add trailing newline

src/utils/utils.py:2:1: I001 [*] Import block is un-sorted or un-formatted
  |
1 |   # imports
2 | / import numpy as np
3 | | import torch
4 | | from torch.utils.data import DataLoader
5 | | from tqdm import tqdm
6 | | import matplotlib.pyplot as plt
7 | | import albumentations as A
8 | | import cv2
  | |__________^ I001
  |
  = help: Organize imports

src/utils/utils.py:7:8: N812 Lowercase `albumentations` imported as non-lowercase `A`
  |
5 | from tqdm import tqdm
6 | import matplotlib.pyplot as plt
7 | import albumentations as A
  |        ^^^^^^^^^^^^^^^^^^^ N812
8 | import cv2
  |

src/utils/utils.py:48:100: E501 Line too long (113 > 99)
   |
46 |     non_house_pixels = total_pixels - house_pixels
47 |     if house_pixels == 0:
48 |          return torch.tensor([1.0]).to(device) # если домов нет, ставим вес 1.0.  Это может быть артефакт данных.
   |                                                                                                    ^^^^^^^^^^^^^^ E501
49 |     pos_weight = torch.tensor([non_house_pixels / house_pixels]).to(device)
50 |     return pos_weight
   |

src/utils/utils.py:120:20: UP038 [*] Use `X | Y` in `isinstance` call instead of `(X, Y)`
    |
118 |         vis_transform_list = [
119 |             t for t in dataset.transform
120 |             if not isinstance(t, (A.Normalize, A.ToTensorV2))
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP038
121 |         ]
122 |         vis_transform = A.Compose(vis_transform_list)
    |
    = help: Convert to `X | Y`

src/utils/utils.py:174:24: W291 [*] Trailing whitespace
    |
174 | __all__ = ['normalize', 
    |                        ^ W291
175 |            'calculate_class_weights', 
176 |            'visualize_predictions', 
    |
    = help: Remove trailing whitespace

src/utils/utils.py:175:38: W291 [*] Trailing whitespace
    |
174 | __all__ = ['normalize', 
175 |            'calculate_class_weights', 
    |                                      ^ W291
176 |            'visualize_predictions', 
177 |            'visualize_segmentation']
    |
    = help: Remove trailing whitespace

src/utils/utils.py:176:36: W291 [*] Trailing whitespace
    |
174 | __all__ = ['normalize', 
175 |            'calculate_class_weights', 
176 |            'visualize_predictions', 
    |                                    ^ W291
177 |            'visualize_segmentation']
    |
    = help: Remove trailing whitespace

src/utils/utils.py:177:37: W292 [*] No newline at end of file
    |
175 |            'calculate_class_weights', 
176 |            'visualize_predictions', 
177 |            'visualize_segmentation']
    |                                     ^ W292
    |
    = help: Add trailing newline

Found 99 errors.
[*] 84 fixable with the --fix option.