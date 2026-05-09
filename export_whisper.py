import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_id = "openai/whisper-tiny"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

model.eval()

dummy_input = torch.randn(1, 80, 3000)  # mel spectrogram shape

torch.onnx.export(
    model.model.encoder,
    dummy_input,
    "whisper_encoder.onnx",
    input_names=["mel"],
    output_names=["encoder_hidden_states"],
    dynamic_axes={
        "mel": {2: "time"}
    },
    opset_version=17
)

print("Encoder exported")