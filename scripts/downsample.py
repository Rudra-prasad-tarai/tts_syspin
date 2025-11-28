import os
import argparse
import librosa
import soundfile as sf
from tqdm import tqdm


def downsample_folder(in_dir: str, out_dir: str, target_sr: int = 16000):
    os.makedirs(out_dir, exist_ok=True)

    wav_files = [
        f for f in os.listdir(in_dir)
        if f.lower().endswith(".wav")
    ]
    print(f"Found {len(wav_files)} wav files in {in_dir}")

    for fname in tqdm(wav_files, desc="Downsampling"):
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)

        # Load with original sampling rate
        audio, sr = librosa.load(in_path, sr=None, mono=True)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        sf.write(out_path, audio, target_sr)

    print("Done downsampling to", target_sr, "Hz.")


def main():
    parser = argparse.ArgumentParser(description="Downsample wav files to target sample rate.")
    parser.add_argument("--in_dir", type=str, required=True, help="Input wav directory (48k).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output wav directory (16k).")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate.")
    args = parser.parse_args()

    downsample_folder(args.in_dir, args.out_dir, args.sr)


if __name__ == "__main__":
    main()
