import torch

torch.set_num_threads(1)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

wav = read_audio('datasets/speech16.wav')
speech_timestamps = get_speech_timestamps(
  wav,
  model,
  return_seconds=False,  # Return speech timestamps in seconds (default is samples)
  visualize_probs=True
)

print(speech_timestamps)