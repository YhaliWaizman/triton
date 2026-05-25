import os
import tritonclient.grpc.aio as triton_grpc_aio
from tritonclient.grpc import InferInput, InferRequestedOutput
import numpy as np
import wave

import torch
from transformers.modeling_outputs import BaseModelOutput

from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import librosa
import os
import wave
import warnings
# warnings.filterwarnings("ignore")

os.environ['HF_HUB_OFFLINE'] = '1'

processor = WhisperProcessor.from_pretrained(
    "./models/whisper-large-v3-turbo"
)
triton_model = WhisperForConditionalGeneration.from_pretrained(
    "./models/whisper-large-v3-turbo"
)
triton_model.generation_config.forced_decoder_ids = None
triton_model.generation_config.suppress_tokens = None
triton_model.generation_config.begin_suppress_tokens = None
triton_model.eval()

# /audio_20260513_174839.wav
AUDIO_DIR = 'saved_audio'
TEST_AUDIO_PATH = list(map(lambda x: f"{AUDIO_DIR}/{x}", os.listdir(AUDIO_DIR)))
print(TEST_AUDIO_PATH)
WINDOW_SIZE = int(16000 * 2)

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

def tokenize_audio(
    pcm_bytes: list[bytes],
    sample_rate: int = 16000,
    channels: int = 1,
    input_layer_name: str = "mel",
    input_data_type: str = "FP32"
):

    # Assuming 16-bit PCM, adjust if audio is different
    # How to know which dtype to use? check the sample_width from the incoming audio
    # sample_width is in bytes, and the dtype in bits.
    # 16-bit PCM is calculated using sample_width x 8 (8 bits per byte)
    audio_type = np.int16

    # audio is a 1D array of 16-bit signed integers.
    audio = np.array([np.frombuffer(e, dtype=audio_type) for e in pcm_bytes])

    # to normalize to [-1.0, 1.0) range, divide by the max absolute value of the min and max of the dtype. For int16, the range is -32768 to 32767, so we divide by 32768.
    # For 16-bit signed integers, the max value is (2 ^ (16 - 1)) - 1 = 32767, and the min value is -(2 ^ (16 - 1)) = -32768. To normalize, we divide by 32768, which is the absolute value of the min.
    audio = audio.astype(np.float32) / 32768.0

    # If the audio has multiple channels, reshape and average to mono
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    # Now `audio` is a 1D numpy array of float32 samples in the range [-1.0, 1.0).
    # Next step is to convert this "raw" audio array into tokenized vector that the model can understand.
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)
    input_features, attention_mask = inputs.input_features, inputs.get("attention_mask")

    # input_features = input_feature.astype(np.float32)
    # infer_input = InferInput(input_layer_name, input_features.shape, input_data_type)
    # infer_input.set_data_from_numpy(input_features)

    # return infer_input
    return input_features, attention_mask

tokenized_batch, attention_mask = tokenize_audio(list(map(lambda x: x['data'][:WINDOW_SIZE], batch)))
hidden_state = triton_model.generate(
    tokenized_batch, 
    attention_mask=attention_mask,
    language="he",
    task="transcribe"
)
print(hidden_state)
# print(f"hidden state shape: {hidden_state.shape}")
# output = processor.batch_decode(hidden_state, skip_special_tokens=True)  #[0].strip()
# print(output)


# with torch.no_grad():
#     predicted_ids = triton_model.generate(tokenized_input)

# print(predicted_ids)