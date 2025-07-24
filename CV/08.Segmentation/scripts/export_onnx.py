import hydra
from omegaconf import DictConfig
import torch
import os

@hydra.main(config_name="../configs", config_name="config", version_base=None)
def export_to_onnx(cfg: DictConfig) --> None:
    """
    Load model and export it in ONNX.
    """
    print("Start to export model")

    # 1. Create the model
    print(f"Create the model copy: {cfg.model._target_}")
    model = hydra.instantiate(cfg.model)

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
    width = cfg.onnx_export.height

    channels = model.model.encoder.in_channels

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
        opset_version=cfg.onnx_eport.opset_version,
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
    print(f"Checl the model")
    onnx_model = torch.onnx.load(onnx_path)
    torch.onnx.check_model(onnx_model)
    print(f"ONNX model saved and finished the check correct: {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()