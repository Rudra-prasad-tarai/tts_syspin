import torch
import torchaudio
from unsloth import FastLanguageModel
from snac import SNAC

print("--- SYSTEM CHECK ---")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n--- 1. LOADING AUDIO CODEC (SNAC) ---")
try:
    # Canopy models use the 24kHz SNAC model
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")
    print("✅ SNAC 24kHz loaded successfully.")
except Exception as e:
    print(f"❌ SNAC Load Failed: {e}")

print("\n--- 2. LOADING LLM IN 4-BIT (UNSLOTH) ---")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "canopylabs/3b-hi-pretrain-research_release",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    print("✅ Canopy 3B Model loaded in 4-bit mode.")
    
    # CRITICAL: Check for the special separator token
    # Orpheus models usually use a specific token ID to separate text from audio
    # We need to find the 'CODE_START_TOKEN' (Usually ID 128257 or similar)
    print(f"Vocab Size: {len(tokenizer)}")
    
except Exception as e:
    print(f"❌ Model Load Failed: {e}")

print("\nIf you see two Green Checks, you are ready for Phase 2 (Data Processing).")