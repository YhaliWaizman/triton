#!/usr/bin/env python3
"""Run speaker diarization fully offline using a local pyannote model.

Examples:
  python run_offline_diarization.py \
	  --audio /path/to/audio.wav \
	  --output diarization.rttm \
	  --model models/pyannote-speaker-diarization-community-1

  python run_offline_diarization.py \
	  --audio /path/to/audio.wav \
	  --model models/pyannote-speaker-diarization-community-1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pyannote.audio import Pipeline


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run offline speaker diarization with pyannote")
	parser.add_argument("--audio", required=True, type=Path, help="Path to input audio file")
	parser.add_argument(
		"--model",
		default="models/pyannote-speaker-diarization-community-1",
		help=(
			"Local model path only (offline). "
			"Example: models/pyannote-speaker-diarization-community-1"
		),
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("diarization.rttm"),
		help="Output RTTM file path",
	)
	parser.add_argument(
		"--output-txt",
		type=Path,
		default=Path("diarization_segments.txt"),
		help="Optional human-readable segment file",
	)
	parser.add_argument("--min-speakers", type=int, default=None, help="Minimum number of speakers")
	parser.add_argument("--max-speakers", type=int, default=None, help="Maximum number of speakers")
	parser.add_argument("--num-speakers", type=int, default=None, help="Fixed number of speakers")
	parser.add_argument(
		"--device",
		choices=["cpu", "cuda"],
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device to run inference on",
	)
	return parser.parse_args()


def load_pipeline(model: str, device: str) -> Pipeline:
	model_path = Path(model)
	if not model_path.exists() or not model_path.is_dir():
		raise FileNotFoundError(
			"Offline mode requires a local model directory. "
			f"Not found: {model_path}"
		)

	pipeline = Pipeline.from_pretrained(str(model_path))

	pipeline.to(torch.device(device))
	return pipeline


def write_txt_output(diarization, output_txt: Path) -> None:
	output_txt.parent.mkdir(parents=True, exist_ok=True)
	with output_txt.open("w", encoding="utf-8") as f:
		for segment, _, speaker in diarization.itertracks(yield_label=True):
			f.write(f"{segment.start:.3f}\t{segment.end:.3f}\t{speaker}\n")


def main() -> None:
	args = parse_args()

	if not args.audio.exists():
		raise FileNotFoundError(f"Audio file not found: {args.audio}")

	if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
		raise ValueError("Use either --num-speakers OR --min-speakers/--max-speakers, not both.")

	pipeline = load_pipeline(args.model, args.device)

	diarization_kwargs = {}
	if args.num_speakers is not None:
		diarization_kwargs["num_speakers"] = args.num_speakers
	else:
		if args.min_speakers is not None:
			diarization_kwargs["min_speakers"] = args.min_speakers
		if args.max_speakers is not None:
			diarization_kwargs["max_speakers"] = args.max_speakers

	diarization = pipeline(str(args.audio), **diarization_kwargs)

	print(diarization)

	# args.output.parent.mkdir(parents=True, exist_ok=True)
	# with args.output.open("w", encoding="utf-8") as f:
	# 	diarization.write_rttm(f)

	# if args.output_txt:
	# 	write_txt_output(diarization, args.output_txt)

	# print(f"Diarization complete. RTTM saved to: {args.output}")
	# if args.output_txt:
	# 	print(f"Segment list saved to: {args.output_txt}")


if __name__ == "__main__":
	main()
