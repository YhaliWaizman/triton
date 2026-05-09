from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
from fastapi import WebSocketDisconnect
import asyncio
import json
import os
from datetime import datetime
import wave
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import tritonclient.grpc.aio as triton_grpc_aio
from tritonclient.grpc import InferInput, InferRequestedOutput

app = FastAPI()

templates = Jinja2Templates(directory="frontend")

# Create directory for saving audio files
AUDIO_DIR = "saved_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Defaults; frontend sends actual config when websocket starts.
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2

TRITON_GRPC_ADDR = os.getenv("TRITON_GRPC_ADDR", "localhost:8001")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "onnx_whisper")
PARTIAL_EMIT_THRESHOLD_BYTES = int(os.getenv("PARTIAL_EMIT_THRESHOLD_BYTES", "64000"))

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
decoder_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
decoder_model.config.forced_decoder_ids = None
decoder_model.eval()


def decode_from_encoder_hidden(encoder_hidden_states: np.ndarray) -> str:
	encoder_outputs = BaseModelOutput(
		last_hidden_state=torch.from_numpy(encoder_hidden_states).float()
	)
	with torch.no_grad():
		predicted_ids = decoder_model.generate(encoder_outputs=encoder_outputs)
	text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
	return text.strip()


async def transcribe_with_triton(
	triton_client: triton_grpc_aio.InferenceServerClient,
	pcm_bytes: bytes,
	sample_rate: int,
	channels: int,
) -> str:
	if not pcm_bytes:
		return ""

	audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
	if channels > 1:
		audio = audio.reshape(-1, channels).mean(axis=1)

	features = processor(audio, sampling_rate=sample_rate, return_tensors="np").input_features
	features = features.astype(np.float32)

	infer_input = InferInput("mel", features.shape, "FP32")
	infer_input.set_data_from_numpy(features)
	requested_output = InferRequestedOutput("encoder_hidden_states")

	result = await triton_client.infer(
		model_name=TRITON_MODEL_NAME,
		inputs=[infer_input],
		outputs=[requested_output],
	)
	encoder_hidden_states = result.as_numpy("encoder_hidden_states")
	if encoder_hidden_states is None:
		return ""

	return await asyncio.to_thread(decode_from_encoder_hidden, encoder_hidden_states)

@app.get("/")
def index(request: Request):
	return templates.TemplateResponse("index.html", { "request": request })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
	await websocket.accept()

	# Create timestamped audio file
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	audio_file_path = os.path.join(AUDIO_DIR, f"audio_{timestamp}.wav")
	wav_file = None
	sample_rate = DEFAULT_SAMPLE_RATE
	channels = DEFAULT_CHANNELS
	sample_width = DEFAULT_SAMPLE_WIDTH
	pcm_buffer = bytearray()
	last_partial = ""
	triton_client = None

	async def send_transcript(kind: str, text: str):
		await websocket.send_text(
			json.dumps(
				{
					"type": "transcript",
					"kind": kind,
					"text": text,
					"transcript": text,
				}
			)
		)

	try:
		triton_client = triton_grpc_aio.InferenceServerClient(url=TRITON_GRPC_ADDR)
		is_live = await triton_client.is_server_live()
		if not is_live:
			await send_transcript("error", f"Triton server at {TRITON_GRPC_ADDR} is not live")
			return

		while True:
			message = await websocket.receive()

			if "text" in message and message["text"] is not None:
				try:
					payload = json.loads(message["text"])
				except json.JSONDecodeError:
					continue

				if payload.get("type") == "audio_config":
					sample_rate = int(payload.get("sampleRate", DEFAULT_SAMPLE_RATE))
					channels = int(payload.get("channels", DEFAULT_CHANNELS))
					sample_width = int(payload.get("sampleWidth", DEFAULT_SAMPLE_WIDTH))
					print(
						f"Audio config received: sample_rate={sample_rate}, channels={channels}, sample_width={sample_width}"
					)

					if wav_file is None:
						wav_file = wave.open(audio_file_path, "wb")
						wav_file.setnchannels(channels)
						wav_file.setsampwidth(sample_width)
						wav_file.setframerate(sample_rate)

			if "bytes" in message and message["bytes"] is not None:
				audio_chunk = message["bytes"]
				pcm_buffer.extend(audio_chunk)
				if wav_file is None:
					wav_file = wave.open(audio_file_path, "wb")
					wav_file.setnchannels(channels)
					wav_file.setsampwidth(sample_width)
					wav_file.setframerate(sample_rate)

				wav_file.writeframes(audio_chunk)
				print(f"Received PCM chunk: {len(audio_chunk)} bytes - saving to {audio_file_path}")

				if len(pcm_buffer) >= PARTIAL_EMIT_THRESHOLD_BYTES:
					partial_text = await transcribe_with_triton(
						triton_client,
						bytes(pcm_buffer),
						sample_rate,
						channels,
					)
					if partial_text and partial_text != last_partial:
						last_partial = partial_text
						await send_transcript("partial", partial_text)

	except WebSocketDisconnect:
		print("Connection closed")
	except Exception as e:
		print("Connection closed:", e)
		await send_transcript("error", str(e))
	finally:
		if triton_client is not None:
			try:
				final_text = await transcribe_with_triton(
					triton_client,
					bytes(pcm_buffer),
					sample_rate,
					channels,
				)
				if final_text:
					await send_transcript("final", final_text)
			except Exception as e:
				print("Final Triton inference failed:", e)
				await send_transcript("error", str(e))
			await triton_client.close()
		if wav_file is not None:
			wav_file.close()
		print(f"Audio saved to {audio_file_path}")