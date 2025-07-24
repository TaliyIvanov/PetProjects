# python3 -m scripts.export_onnx

import hydra
from omegaconf import DictConfig
import torch
import os
import onnx
import onnxruntime as ort
import numpy as np

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def export_to_onnx(cfg: DictConfig) -> None:
    """
    Load model and export it in ONNX.
    """
    print("Start to export model")

    # 1. Create the model
    print(f"Create the model copy: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)

    # 2. Load weights
    weights_path = cfg.onnx_export.weights_path
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file didn't found: {weights_path}")
    
    print(f"Load weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # must have step!
    model.eval()

    # 3. Dummy input
    batch_size = cfg.onnx_export.batch_size
    height = cfg.onnx_export.height
    width = cfg.onnx_export.width

    channels = model.in_channels

    dummy_input =  torch.randn(batch_size, channels, height, width, requires_grad=False)
    print(f"Size of input tansor to trass..: {dummy_input.shape}")

    # 4. Export to ONNX
    output_dir = "onnx_models"
    os.makedirs(output_dir, exist_ok=True)

    model_name = cfg.model._target_.split(".")[-1].lower()
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    print(f"Export model to: {onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=cfg.onnx_export.opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
        )
    
    print(f"Export finished")

    # 5. Check
    print(f"Check the model")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print("ONNX model structure is valid.")

    # (Опционально, но КРАЙНЕ РЕКОМЕНДУЕТСЯ)
    # Проверяем, что выходные данные модели ONNX совпадают с PyTorch
    print("Comparing ONNX Runtime and PyTorch results...")

    # Получаем выход PyTorch
    model.eval()
    with torch.no_grad():
        torch_out = model(dummy_input)

    # Получаем выход ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: dummy_input.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # Сравниваем результаты
    # Используем np.allclose для сравнения чисел с плавающей точкой
    np.testing.assert_allclose(torch_out.cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)

    print(f"ONNX model has been tested with ONNX Runtime, and the result is consistent with PyTorch.")
    print(f"ONNX model saved and checked successfully: {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()