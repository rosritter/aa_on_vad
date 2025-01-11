from dotenv.main import load_dotenv
import os
import torchaudio as ta


# Load huggingface access token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")


from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token=hf_token)
output = pipeline("datasets/speech16.wav")
print(output)
print(pipeline._inferences['_segmentation'].model)