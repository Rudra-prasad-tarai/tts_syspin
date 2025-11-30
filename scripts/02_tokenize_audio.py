import torch
import torchaudio
from snac import SNAC
import json
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# We use the training metadata created in Step 1
INPUT_METADATA = "./processed_data/metadata_train.jsonl" 
OUTPUT_DATASET = "./dataset_tokenized_train.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def flatten_snac_codes(codes):
    """
    Converts SNAC's 3-layer codebooks into a single flattened list of integers.
    Pattern: [L0, L1, L1, L2, L2, L2, L2] repeated.
    """
    c0, c1, c2 = codes
    
    # Move to CPU list
    c0 = c0.squeeze(0).cpu().tolist()
    c1 = c1.squeeze(0).cpu().tolist()
    c2 = c2.squeeze(0).cpu().tolist()

    flattened_sequence = []
    time_steps = len(c0)
    
    for t in range(time_steps):
        # The specific interleaving pattern: 1 coarse + 2 mid + 4 fine = 7 tokens
        flattened_sequence.append(c0[t])      # Layer 0
        flattened_sequence.append(c1[2*t])    # Layer 1 part a
        flattened_sequence.append(c1[2*t+1])  # Layer 1 part b
        flattened_sequence.append(c2[4*t])    # Layer 2 part a
        flattened_sequence.append(c2[4*t+1])  # Layer 2 part b
        flattened_sequence.append(c2[4*t+2])  # Layer 2 part c
        flattened_sequence.append(c2[4*t+3])  # Layer 2 part d
        
    return flattened_sequence

def main():
    print(f"--- 🚀 Starting Tokenization on {DEVICE} ---")
    
    # 1. Load SNAC Model
    try:
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE)
        snac_model.eval()
        print("✅ SNAC Model Loaded")
    except Exception as e:
        print(f"❌ Failed to load SNAC: {e}")
        return

    # 2. Check Input
    if not os.path.exists(INPUT_METADATA):
        print(f"❌ Missing {INPUT_METADATA}. Run scripts/01_process_audio.py first!")
        return
        
    with open(INPUT_METADATA, 'r', encoding='utf-8') as f:
        data_entries = [json.loads(line) for line in f]
    
    print(f"📂 Tokenizing {len(data_entries)} training samples...")
    print("   (This will take 30-45 minutes on your GPU)")
    
    # 3. Process
    processed_count = 0
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as outfile:
        for entry in tqdm(data_entries, desc="Encoding Audio"):
            wav_path = entry['audio']
            text = entry['text']
            
            try:
                # Load & Encode
                wav, sr = torchaudio.load(wav_path)
                wav = wav.to(DEVICE).unsqueeze(0) # Add batch dim
                
                with torch.no_grad():
                    codes = snac_model.encode(wav)
                
                audio_tokens = flatten_snac_codes(codes)
                
                # Save
                final_entry = {"text": text, "audio_tokens": audio_tokens}
                json.dump(final_entry, outfile, ensure_ascii=False)
                outfile.write("\n")
                processed_count += 1
                
            except Exception as e:
                # Skip corrupt files silently
                continue

    print(f"\n✅ DONE! Successfully tokenized {processed_count} files.")
    print(f"💾 Ready for training: {OUTPUT_DATASET}")

if __name__ == "__main__":
    main()
