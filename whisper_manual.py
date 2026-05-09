from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

import warnings
warnings.filterwarnings("ignore")

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None


audio, sr = librosa.load("saved_audio/audio_20260509_162506.wav", sr=16000)  # Whisper expects 16kHz

print(audio, sr)

input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features)

# decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)
