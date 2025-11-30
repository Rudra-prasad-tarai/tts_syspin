import json
import os
import torchaudio
import glob
from tqdm import tqdm
from pathlib import Path
import torch

# --- CONFIGURATION ---
# We assume you are running this from the 'tts_syspin' folder
# Path to the specific speaker folder shown in your screenshot
SYSPIN_SPEAKER_DIR = "./IISc_SYSPIN_Data/IISc_SYSPINProject_Hindi_Female_Spk001_HC"

OUTPUT_DIR = "./processed_data"
TARGET_SAMPLE_RATE = 24000  # Required for Orpheus/Canopy

def process_data():
    # 1. Verification
    if not os.path.exists(SYSPIN_SPEAKER_DIR):
        print(f"❌ ERROR: Could not find folder at: {os.path.abspath(SYSPIN_SPEAKER_DIR)}")
        print("   Make sure you are running this script from the 'tts_syspin' directory.")
        return

    os.makedirs(f"{OUTPUT_DIR}/wavs", exist_ok=True)

    # 2. Locate the WAV folder (Explicitly 'wav' based on your screenshot)
    wav_folder = os.path.join(SYSPIN_SPEAKER_DIR, "wav")
    if not os.path.exists(wav_folder):
        print(f"❌ ERROR: Could not find 'wav' folder inside {SYSPIN_SPEAKER_DIR}")
        return
    
    # 3. Locate the JSON Transcript
    # We look for any JSON file in the speaker dir
    json_files = glob.glob(f"{SYSPIN_SPEAKER_DIR}/*.json")
    if not json_files:
        print("❌ ERROR: No .json transcript file found in the speaker directory.")
        return
    
    # Take the first JSON found (there should only be one per speaker)
    json_path = json_files[0]
    print(f"📄 Found Transcript: {os.path.basename(json_path)}")

    # 4. Process Data
    train_entries = []
    test_entries = []
    
    unique_domains_found = set()
    skipped_count = 0
    processed_count = 0

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return

    transcripts = data.get("Transcripts", {})
    
    print(f"�� Processing {len(transcripts)} items...")

    for key, content in tqdm(transcripts.items(), desc="Converting Audio"):
        text = content.get("Transcript")
        
        # DOMAIN HANDLING
        # Your JSON has "BOOKS", "WEATHER", etc.
        # We need to filter out "EVALUATION"
        raw_domain = content.get("Domain", "UNKNOWN")
        domain = raw_domain.strip().upper() # Normalize to uppercase
        unique_domains_found.add(domain)

        if not text:
            continue

        # Construct wav path: Key + .wav
        wav_filename = f"{key}.wav"
        wav_path = os.path.join(wav_folder, wav_filename)

        if not os.path.exists(wav_path):
            # Sometimes keys differ slightly from filenames, skip if missing
            skipped_count += 1
            continue

        # --- AUDIO PROCESSING (48k -> 24k) ---
        try:
            waveform, sample_rate = torchaudio.load(wav_path)

            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)

            # Ensure Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Save Processed File
            new_wav_path = os.path.join(OUTPUT_DIR, "wavs", wav_filename)
            torchaudio.save(new_wav_path, waveform, TARGET_SAMPLE_RATE)

            entry = {
                "audio": os.path.abspath(new_wav_path), 
                "text": text,
                "domain": raw_domain
            }
            
            # THE HACKATHON FILTER
            if domain == "EVALUATION":
                test_entries.append(entry)
            else:
                train_entries.append(entry)
                
            processed_count += 1

        except Exception as e:
            print(f"Error converting {wav_filename}: {e}")
            skipped_count += 1

    # 5. Save Manifests
    # Training Data
    with open(f"{OUTPUT_DIR}/metadata_train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    # Test Data (Held out)
    with open(f"{OUTPUT_DIR}/metadata_test.jsonl", "w", encoding="utf-8") as f:
        for entry in test_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print("\n" + "="*60)
    print(f"🎉 PROCESSING COMPLETE")
    print(f"📂 WAVs Saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("-" * 30)
    print(f"📚 TRAINING Set (metadata_train.jsonl): {len(train_entries)} samples")
    print(f"🧪 TESTING Set  (metadata_test.jsonl):  {len(test_entries)} samples")
    print("-" * 30)
    print(f"🔎 Domains Found: {sorted(list(unique_domains_found))}")
    print(f"⚠️ Skipped Files: {skipped_count}")
    print("="*60)
    
    if len(test_entries) == 0:
        print("⚠️ WARNING: No 'EVALUATION' samples found! Check the 'Domains Found' list above.")

if __name__ == "__main__":
    process_data()
