# from tokenizers import Tokenizer

# tokenizer = Tokenizer.from_file("./models/whisper/tokenizer.json")

# print(tokenizer.encode("gigga nigga").ids)
# print(tokenizer.decode([50258, 50363, 70, 328, 3680, 41626, 50257]))




import os
# os.environ["HF_HUB_OFFLINE"] = "1"

from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="hf_****"
)

diarization = pipeline("./saved_audio/audio_20260513_125646.wav")
print(diarization)

