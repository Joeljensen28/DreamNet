import os
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from tqdm import tqdm

AUDIO_DIR = "data/original"
SAVE_DIR = "data/tokens"
CHUNK_DURATION = 10
SAMPLE_RATE = 16000
CHUNK_SAMPLES = CHUNK_DURATION * SAMPLE_RATE

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

audio_files = [
    f for f in os.listdir(AUDIO_DIR)
]

print(f"Found {len(audio_files)} files to tokenize...")

for filename in tqdm(audio_files, desc="Tokenizing"):
    audio_path = os.path.join(AUDIO_DIR, filename)
    save_path = os.path.join(SAVE_DIR, os.path.splitext(filename)[0] + ".pt")

    try:
        waveform, sr = torchaudio.load(audio_path)
        waveform = convert_audio(waveform, sr, model.sample_rate, model.channels)

        with torch.no_grad():
            tokens = []
            for start in range(0, waveform.shape[1], CHUNK_SAMPLES):
                chunk = waveform[:, start:start+CHUNK_SAMPLES]
                if chunk.shape[1] < CHUNK_SAMPLES // 2:
                    continue
                
                encoded_frames = model.encode(chunk.unsqueeze(0))
                codebooks = [cb for (cb, _) in encoded_frames]
                chunk_codes = torch.cat(codebooks, dim=2).squeeze(0).T 
                tokens.append(chunk_codes)

            codes = torch.cat(tokens, dim=0)

        torch.save(codes, save_path)

    except Exception as e:
        print(f"Failed on {filename}: {e}")
