import wave
import torch
import torchaudio
import silero_vad
import numpy as np
from numpy.matlib import empty
from pydub import AudioSegment
from speechbrain.inference.speaker import EncoderClassifier
import torch.nn.functional as F
from speechbrain.utils.fetching import LocalStrategy
model = silero_vad.load_silero_vad()
import os
import parselmouth
from parselmouth.praat import call
import scipy.io.wavfile as wav
import noisereduce as nr
os.add_dll_directory(r"C:\Users\yona\Downloads\ffmpeg-8.1.2-full_build-shared\ffmpeg-8.1.2-full_build-shared\bin")
with wave.open(r"C:\Users\yona\Downloads\harvard.wav", "rb") as wav_file:
    sample_rate = wav_file.getframerate()
    audio = wav_file.readframes(wav_file.getnframes())
    channels = wav_file.getnchannels()
    song = AudioSegment.from_wav(r"C:\Users\yona\Downloads\harvard.wav").set_channels(1)
    sounds = parselmouth.Sound(r"C:\Users\yona\Downloads\harvard.wav")
split1 = song[:5000]
split2 = song[6000:]
trainer = torch.from_numpy(np.array(split1.get_array_of_samples(), dtype=np.float32))/32768
splittrain = trainer.unsqueeze(0)




tester = torch.from_numpy(np.array(split2.get_array_of_samples(), dtype=np.float32))/32768

splittest = tester.unsqueeze(0)
persontalking = False

pausefloor = 0.2
pauseroof = 4
wav = torch.frombuffer(bytearray(audio), dtype=torch.int16).float() / 32768.0
if channels >= 2:
    wav = wav.reshape(-1, 2).mean(dim=1)
if sample_rate != 16000:
    wav = torchaudio.functional.resample(wav, sample_rate, 16000)

speech_timestamps = silero_vad.get_speech_timestamps(wav, model)

pauses = []
listaudio = [[i['start']/16000, i['end']/16000] for i in speech_timestamps]
print(listaudio)
for i in range(len(listaudio)-1):
    if pausefloor <= np.abs(listaudio[i][1] - listaudio[i+1][0]) <= pauseroof:
        pauses.append(np.abs(listaudio[i][1] - listaudio[i+1][0]))
if not pauses:
    print(pauses)
else:
    average = np.mean(pauses)
    persontalking = True
    print(pauses,"PAUSES")
    print(average)

if persontalking:


  classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb", local_strategy=LocalStrategy.COPY)


  training = classifier.encode_batch(trainer).flatten()
  testing = classifier.encode_batch(tester).flatten()

  score = F.cosine_similarity(training,testing,dim=0)

  print(f"Similarity Score: {score.item():.2f}")
  if score.item() > 0.45:
    print("User identified: Match confirmed!")
    usertalking = True
  else:
      print("User identified: Unknown speaker.")
if usertalking:
  sound = parselmouth.Sound(sounds)
  pitch = sound.to_pitch()

  pitch_values = pitch.selected_array['frequency']

  voiced_pitches = pitch_values[pitch_values > 0]
  std = np.std(voiced_pitches)
  mean = np.mean(voiced_pitches)
  print("std",std, "mean",mean)

algorithm_path = r"C:\Users\yona\Downloads\syllable_nuclei.py"

with open(r"C:\Users\yona\Downloads\syllable_nuclei.py", encoding="utf-8") as f:
    nuclei = f.read()

algorithm_namespace = {}
exec(nuclei, algorithm_namespace)

preresults = algorithm_namespace["speech_rate"]

results = preresults(r"C:\Users\yona\Downloads\harvard.wav")

print("\nSyllable Nuclei Results:")
print(f"Pauses: {results['npause']}")
print(f"Speech rate: "f"{results['speechrate(nsyll / dur)']:.2f} syllables/second")
print(f"Articulation rate: "f"{results['articulation rate(nsyll / phonationtime)']:.2f} syllables/second")
print(f"Average syllable duration: "f"{results['ASD(speakingtime / nsyll)']:.3f} seconds")
print("std", std, "mean", mean)
