import json
import os
import inspect

import numpy as np
import torch
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        from pyannote.audio import Pipeline

        pipeline_source = os.getenv(
            "PYANNOTE_PIPELINE_PATH", "pyannote/speaker-diarization-3.1"
        )
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        # Prefer local path in air-gapped deployments.
        if os.path.exists(pipeline_source):
            self.pipeline = Pipeline.from_pretrained(pipeline_source)
        else:
            if not hf_token:
                raise RuntimeError(
                    "Set PYANNOTE_PIPELINE_PATH to a local pipeline file for offline use, "
                    "or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN for HuggingFace download."
                )

            params = inspect.signature(Pipeline.from_pretrained).parameters
            auth_kw = "token" if "token" in params else "use_auth_token"
            self.pipeline = Pipeline.from_pretrained(
                pipeline_source,
                **{auth_kw: hf_token},
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = self.pipeline.to(torch.device(device))
        print(f"pyannote pipeline loaded on {device}")

    def execute(self, requests):
        responses = []
        for request in requests:
            audio = pb_utils.get_input_tensor_by_name(request, "audio_float32").as_numpy()
            sample_rate = int(
                pb_utils.get_input_tensor_by_name(request, "sample_rate").as_numpy()[0]
            )

            # pyannote expects waveform shape (channels, samples)
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate}
            )

            segments = [
                {
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker,
                }
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]

            out_tensor = pb_utils.Tensor(
                "segments_json",
                np.array([json.dumps(segments)], dtype=object),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        self.pipeline = None
