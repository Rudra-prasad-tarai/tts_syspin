from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

# --- CONFIGURATION ---
MODEL_NAME = "canopylabs/3b-hi-pretrain-research_release"
DATASET_FILE = "dataset_tokenized_train.jsonl"
OUTPUT_DIR = "./lora_syspin_hindi"

# 16GB VRAM Settings
MAX_SEQ_LENGTH = 2048 # Reduce to 1024 if you get OOM (Out of Memory)
BATCH_SIZE = 2        # Small batch size per step
GRAD_ACCUM = 4        # Accumulate gradients to simulate Batch Size = 8

def main():
    print(f"--- 🚀 Initializing Training on {torch.cuda.get_device_name(0)} ---")

    # 1. Load Model & Tokenizer in 4-bit (Memory Efficient)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # Auto-detects float16/bfloat16
        load_in_4bit = True,
    )

    # 2. Add LoRA Adapters
    # This makes the model "learnable" without updating all 3B parameters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Higher rank = better quality, slightly more VRAM
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0, 
        bias = "none",   
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 3. Load Your Tokenized Dataset
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
    print(f"📚 Loaded {len(dataset)} training samples.")

    # 4. Formatting Function
    # We need to combine TEXT + SEPARATOR + AUDIO into one long sequence
    # Note: Orpheus models usually just predict the audio after the text.
    # We will use a standard separator.
    
    # Let's find a good separator. Usually, models have a specific one.
    # For now, we will use the EOS token or a newline as a soft separator.
    text_audio_separator = "\n" 

    def formatting_prompts_func(examples):
        texts = examples["text"]
        audio_tokens_list = examples["audio_tokens"]
        outputs = []
        
        for text, audio_tokens in zip(texts, audio_tokens_list):
            # Convert audio integers to string tokens (if needed) or keep as IDs.
            # However, SFTTrainer expects text strings. 
            # TRICK: We rely on the tokenizer to handle the text, 
            # but we need to inject the audio tokens manually in the collator usually.
            
            # SIMPLIFIED APPROACH for Unsloth SFT:
            # We format the input as: "Text content... \n <audio_codes...>"
            # Note: This implies the tokenizer can decode the audio codes. 
            # If the audio codes are distinct IDs (e.g. 32000+), we need to ensure 
            # they are treated correctly.
            
            # Since 'audio_tokens' are already integers (IDs) from SNAC,
            # and 'text' is raw string, we need to do this carefully.
            
            # For this specific script, we will format it so the model learns:
            # Input: Text
            # Output: Audio Tokens
            
            # We construct a string representation (This is a bit hacky but works for SFT)
            # A better way is a custom DataCollator, but let's try the text-based approach first
            # assuming the tokenizer can handle the raw text.
            
            part = f"{text}{text_audio_separator}"
            # We append the audio tokens as a string of IDs? No, that breaks.
            
            # actually, standard SFTTrainer is tricky with pre-tokenized audio.
            # Let's return just the text for now and let the Collator handle the audio?
            # No, we will use a raw text approach:
            
            outputs.append(part) 

        return outputs

    # --- CRITICAL FIX FOR AUDIO LLMS ---
    # Since we have pre-tokenized audio (integers) and raw text,
    # Standard SFTTrainer is hard to use out-of-the-box.
    # We will use a custom Data Collator approach in a more advanced script if this fails.
    # BUT for now, let's trust Unsloth's ability to handle mixed inputs if we format correctly.
    
    # 5. Training Arguments
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text", # This is just a placeholder
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps = 100,
            max_steps = 1000, # Train for 1000 steps (approx 1 epoch or less)
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR,
            save_steps = 200, # Save every 200 steps
        ),
    )
    
    # CUSTOM COLLATOR HOOK (Advanced)
    # We need to override the inputs to include our audio tokens
    # This is where the magic happens to merge Text IDs + Audio IDs
    def custom_collate_fn(features):
        batch_text = [f["text"] + "\n" for f in features]
        batch_audio = [f["audio_tokens"] for f in features]
        
        # 1. Tokenize Text
        text_inputs = tokenizer(batch_text, padding=False, truncation=False, add_special_tokens=False)
        
        input_ids = []
        labels = []
        
        for i in range(len(features)):
            t_ids = text_inputs["input_ids"][i]
            a_ids = batch_audio[i]
            
            # Combine: [Text IDs] + [Audio IDs]
            combined_ids = t_ids + a_ids
            
            # Truncate if too long
            if len(combined_ids) > MAX_SEQ_LENGTH:
                combined_ids = combined_ids[:MAX_SEQ_LENGTH]
            
            # Labels: We mask the text part (set to -100) so we only learn to predict Audio
            # Label = [-100, -100, ... -100, Audio_ID, Audio_ID...]
            label_ids = [-100] * len(t_ids) + a_ids
            if len(label_ids) > MAX_SEQ_LENGTH:
                label_ids = label_ids[:MAX_SEQ_LENGTH]
                
            input_ids.append(torch.tensor(combined_ids))
            labels.append(torch.tensor(label_ids))
            
        # Pad sequence
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

    # Inject our custom collator
    trainer.data_collator = custom_collate_fn

    # 6. TRAIN!
    print("🚂 Choo Choo! Training started...")
    trainer_stats = trainer.train()

    # 7. Save
    print(f"💾 Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
