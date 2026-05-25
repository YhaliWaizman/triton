from pyannote.audio import Inference, Model
import wave
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ['HF_HUB_OFFLINE'] = '1'

AUDIO_FILE = "./saved_audio/audio_20260525_173425.wav"
MODEL_PATH = "./models/segmentation"

AUDIO_DIR = 'saved_audio'
TEST_AUDIO_PATH = [AUDIO_FILE]  # list(map(lambda x: f"{AUDIO_DIR}/{x}", os.listdir(AUDIO_DIR)))
print(TEST_AUDIO_PATH)
WINDOW_SIZE = int(16000 * 2)

def visualize_segmentation(
    audio: np.ndarray,
    sample_rate: int,
    scores: np.ndarray,
    frame_start: float,
    frame_step: float,
    label_names: list[str] | None = None,
    selected_indices: tuple[int, ...] = (1, 2, 3),
) -> None:
    """Plot waveform, all label scores, and selected output indices."""
    audio_t = np.arange(audio.shape[0], dtype=np.float32) / float(sample_rate)
    score_t = frame_start + np.arange(scores.shape[0], dtype=np.float32) * frame_step

    if label_names is None or len(label_names) != scores.shape[1]:
        label_names = [f"label_{i}" for i in range(scores.shape[1])]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)

    # 1) Waveform
    axes[0].plot(audio_t, audio, linewidth=0.6, color="black")
    axes[0].set_title("Audio waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.2)

    # 2) Scores for all model outputs (expected 7 for pyannote/segmentation)
    for i in range(scores.shape[1]):
        axes[1].plot(score_t, scores[:, i], linewidth=1.0, label=label_names[i])
    axes[1].set_title(f"Segmentation scores (all outputs: {scores.shape[1]})")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)

    # 3) Focus on indices 1, 2, 3
    valid_indices = [i for i in selected_indices if 0 <= i < scores.shape[1]]
    for i in valid_indices:
        axes[2].plot(score_t, scores[:, i], linewidth=1.4, label=f"idx {i}: {label_names[i]}")
    axes[2].set_title(f"Selected outputs {tuple(valid_indices)}")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Score")
    axes[2].grid(alpha=0.2)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    plt.show()

def read_audio(audio_path: str):
    with wave.open(audio_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()

        print(f"Audio file info: sample_rate={sample_rate}, channels={channels}, sample_width={sample_width}, n_frames={n_frames}")
        wav_data = wav_file.readframes(n_frames)
        print(f"Read {len(wav_data)} bytes of audio data from {audio_path}")
    
    return { 'data': wav_data, 'sr': sample_rate, 'ch': channels }

batch = list(map(read_audio, TEST_AUDIO_PATH))
X = batch[0]
channels = X['ch']

model = Model.from_pretrained(MODEL_PATH)
print(model)

inference = Inference(model, window="sliding")
embedding = inference(AUDIO_FILE)
print(embedding.data)

audio_type = np.int16

# audio is a 1D array of 16-bit signed integers.
audio = np.frombuffer(X['data'], dtype=audio_type)

# to normalize to [-1.0, 1.0) range, divide by the max absolute value of the min and max of the dtype. For int16, the range is -32768 to 32767, so we divide by 32768.
# For 16-bit signed integers, the max value is (2 ^ (16 - 1)) - 1 = 32767, and the min value is -(2 ^ (16 - 1)) = -32768. To normalize, we divide by 32768, which is the absolute value of the min.
audio = audio.astype(np.float32) / 32768.0

# If the audio has multiple channels, reshape and average to mono
if channels > 1:
	audio = audio.reshape(-1, channels).mean(axis=1)

x = torch.tensor(audio).unsqueeze(0)
y = model.forward(x)
print(y)
print(y.shape)


# model_classes = None
# if getattr(model, "specifications", None) is not None and getattr(model.specifications, "classes", None):
#     model_classes = list(model.specifications.classes)
# print(model_classes)

visualize_segmentation(
	audio,
	sample_rate=X['sr'],
	scores=y.detach()[0],
	frame_start=float(embedding.sliding_window.start),
	frame_step=float(embedding.sliding_window.step),
	label_names=['VAD', 'Speaker change', 'Overlap'],
	selected_indices=(1, 2, 3),     
)



