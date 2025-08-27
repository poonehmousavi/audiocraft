import json
import os

# Replace with your input file
input_jsonl = "/home/jovyan/project/audiocraft/ttas/22k_samples.json"
output_jsonl = "dataset.jsonl"
info_dir = "/home/jovyan/project/audiocraft/dataset"
os.makedirs(info_dir, exist_ok=True)

def get_audio_metadata(audio_path):
    # Dummy function for illustration. Replace with real code if needed.
    # Use torchaudio.info or librosa.get_duration, etc.
    # For now, returns (duration, sample_rate)
    return 15.0, 48000  # Example values

with open(input_jsonl, "r") as f_in, open(output_jsonl, "w") as f_out:
    for line in f_in:
        ex = json.loads(line)
        # Define paths
        audio_path = ex["audio_file"]
        speech_path = ex["speech_file"]

        # Get duration and sample_rate (you should use a function for this)
        duration, sample_rate = get_audio_metadata(audio_path)

        # Save info file
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        info_path = os.path.join(info_dir, f"{base_name}.json")
        # All fields except the audio/speech paths
        info_fields = {k: v for k, v in ex.items() if k not in ["audio_file", "speech_file"]}
        with open(info_path, "w") as info_out:
            json.dump(info_fields, info_out, ensure_ascii=False, indent=2)

        # First row: audio as path, speech as input_path
        row1 = {
            "path": audio_path,
            "input_path": speech_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "amplitude": None,
            "weight": None,
            "info_path": info_path
        }
        f_out.write(json.dumps(row1) + "\n")

        # Second row: speech as path, audio as input_path
        row2 = {
            "path": speech_path,
            "input_path": audio_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "amplitude": None,
            "weight": None,
            "info_path": info_path
        }
        f_out.write(json.dumps(row2) + "\n")
