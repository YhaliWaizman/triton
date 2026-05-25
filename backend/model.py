import asyncio
import json
import os
from typing import AsyncIterator

import grpc
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

GRPC_BIND_ADDR = os.getenv("GRPC_BIND_ADDR", "0.0.0.0:50051")
MODEL_ID = os.getenv("WHISPER_MODEL_ID", "openai/whisper-tiny")

REQ_KIND_CONFIG = 1
REQ_KIND_AUDIO = 2
REQ_KIND_END = 3


class StreamingASRService:
	def __init__(self) -> None:
		print(f"Loading Whisper model: {MODEL_ID}")
		self.processor = WhisperProcessor.from_pretrained(MODEL_ID)
		self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
		self.model.config.forced_decoder_ids = None
		self.model.eval()
		print("Whisper model loaded")

	def _transcribe_sync(self, pcm_bytes: bytes, sample_rate: int, channels: int) -> str:
		if not pcm_bytes:
			return ""

		audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
		if channels > 1:
			audio = audio.reshape(-1, channels).mean(axis=1)

		input_features = self.processor(
			audio,
			sampling_rate=sample_rate,
			return_tensors="pt",
		).input_features

		# print(input_features)
		print(type(input_features))

		with torch.no_grad():
			predicted_ids = self.model.generate(input_features)

		text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

		print(text.strip())
		return text.strip()

	async def stream_transcribe(self, request_iterator: AsyncIterator[bytes], context):
		sample_rate = 16000
		channels = 1
		pcm_buffer = bytearray()
		last_partial = ""
		chunk_counter = 0

		# Every ~2 seconds at 16kHz mono 16-bit.
		partial_emit_threshold_bytes = 64000

		async for message in request_iterator:
			if not message:
				continue

			kind = message[0]
			payload = message[1:]

			if kind == REQ_KIND_CONFIG:
				try:
					cfg = json.loads(payload.decode("utf-8"))
				except Exception:
					cfg = {}

				sample_rate = int(cfg.get("sampleRate", sample_rate))
				channels = int(cfg.get("channels", channels))
				await asyncio.sleep(0)
				continue

			if kind == REQ_KIND_AUDIO:
				pcm_buffer.extend(payload)
				chunk_counter += 1

				if len(pcm_buffer) >= partial_emit_threshold_bytes:
					partial_text = await asyncio.to_thread(
						self._transcribe_sync,
						bytes(pcm_buffer),
						sample_rate,
						channels,
					)
					if partial_text and partial_text != last_partial:
						last_partial = partial_text
						yield json.dumps(
							{"kind": "partial", "text": partial_text, "chunks": chunk_counter}
						).encode("utf-8")
				continue

			if kind == REQ_KIND_END:
				break

		final_text = await asyncio.to_thread(
			self._transcribe_sync,
			bytes(pcm_buffer),
			sample_rate,
			channels,
		)
		yield json.dumps({"kind": "final", "text": final_text}).encode("utf-8")


async def serve() -> None:
	service = StreamingASRService()

	server = grpc.aio.server()
	handler = grpc.stream_stream_rpc_method_handler(
		service.stream_transcribe,
		request_deserializer=lambda x: x,
		response_serializer=lambda x: x,
	)
	generic_handler = grpc.method_handlers_generic_handler(
		"asr.ASRService",
		{"StreamTranscribe": handler},
	)
	server.add_generic_rpc_handlers((generic_handler,))
	server.add_insecure_port(GRPC_BIND_ADDR)

	await server.start()
	print(f"gRPC ASR service listening on {GRPC_BIND_ADDR}")
	await server.wait_for_termination()


if __name__ == "__main__":
	asyncio.run(serve())
