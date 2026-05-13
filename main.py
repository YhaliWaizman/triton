import asyncio
import os
import tritonclient.grpc.aio as triton_grpc_aio
import numpy as np
import wave

TEST_AUDIO_PATH = "saved_audio/audio_20260511_063454.wav"

with wave.open(TEST_AUDIO_PATH, "rb") as wav_file:
    sample_rate = wav_file.getframerate()
    channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    n_frames = wav_file.getnframes()

    print(f"Audio file info: sample_rate={sample_rate}, channels={channels}, sample_width={sample_width}, n_frames={n_frames}")
    wav_data = wav_file.readframes(n_frames)
    print(f"Read {len(wav_data)} bytes of audio data from {TEST_AUDIO_PATH}")

def stream_audio_chunks(wav_data, chunk_size=4096):
    for i in range(0, len(wav_data), chunk_size):
        yield wav_data[i:i+chunk_size]

def tokenize_audio(
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int,
):

    # Assuming 16-bit PCM, adjust if audio is different
    # How to know which dtype to use? check the sample_width from the incoming audio
    # sample_width is in bytes, and the dtype in bits.
    # 16-bit PCM is calculated using sample_width x 8 (8 bits per byte)
    audio_type = np.int16

    # audio is a 1D array of 16-bit signed integers.
    audio = np.frombuffer(pcm_bytes, dtype=audio_type)

    # to normalize to [-1.0, 1.0) range, divide by the max absolute value of the min and max of the dtype. For int16, the range is -32768 to 32767, so we divide by 32768.
    # For 16-bit signed integers, the max value is (2 ^ (16 - 1)) - 1 = 32767, and the min value is -(2 ^ (16 - 1)) = -32768. To normalize, we divide by 32768, which is the absolute value of the min.
    audio = audio.astype(np.float32) / 32768.0

    # If the audio has multiple channels, reshape and average to mono
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    # Now `audio` is a 1D numpy array of float32 samples in the range [-1.0, 1.0).
    # Next step is to convert this "raw" audio array into tokenized vector that the model can understand.
    # tokenized = 


# async def main():
#     try:
#         triton_client = triton_grpc_aio.InferenceServerClient(url="localhost:8001")
#         is_live = await triton_client.is_server_live()
#         if not is_live:
#             print("Triton server is not live")
#             return

#         print(f"Triton server live: {is_live}")

#         infer_input = triton_grpc_aio.InferInput("mel", [1, 80, 300], "FP32")
#         infer_input.set_data_from_numpy(np.random.rand(1, 80, 300).astype(np.float32))  # Example input

#         requested_output = triton_grpc_aio.InferRequestedOutput("encoder_hidden_states")

#         result = await triton_client.infer(
#             model_name="onnx_whisper",
#             inputs=[
#                 infer_input
#             ],
#             outputs=[
#                 requested_output
#             ]
#         )

#         print("Inference result:", result)

#     except Exception as e:
#         print(f"Error connecting to Triton server: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())


























# import onnx

# model = onnx.load("models/onnx_whisper/1/model.onnx")

# for node in model.graph.node:
#     print(f"Node name: {node.name}, Op type: {node.op_type}")

# for inp in model.graph.input:
# 	print(inp.name)

# for inp in model.graph.output:
# 	print(inp.name)

