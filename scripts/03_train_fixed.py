from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer # Switched to standard Trainer
import os

# --- CONFIGURATION ---
MODEL_NAME = "canopylabs/3b-hi-pretrain-research_release"
DATASET_FILE = "./dataset_tokenized_train.jsonl"
OUTPUT_DIR = "./lora_syspin_hindi_v1"

# 16GB VRAM Settings
MAX_SEQ_LENGTH = 2048 
BATCH_SIZE = 2        
GRAD_ACCUM = 4        

def main():
    print(f"--- 🚀 Initializing Training on {torch.cuda.get_device_name(0)} ---")

    # 1. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, 
        load_in_4bit = True,
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0, 
        bias = "none",   
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 3. Load Dataset
    # We load it, but we DON'T process it here. We let the collator do it live.
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
    print(f"📚 Loaded {len(dataset)} training samples.")

    # 4. Custom Collator (The "Glue")
    # This runs on the CPU right before the batch hits the GPU.
    def custom_collate_fn(features):
        # features is a list of dictionaries: [{"text": "...", "audio_tokens": [...]}, ...]
        
        # 1. Prepare Inputs
        batch_text = [f["text"] + "\n" for f in features]
        batch_audio = [f["audio_tokens"] for f in features]
        
        # 2. Tokenize Text (Dynamically pad to the longest text in this batch)
        # We explicitly return tensors='pt' (PyTorch)
        text_inputs = tokenizer(
            batch_text, 
            padding=True,          # Pad to longest in batch
            truncation=True,       # Truncate if huge
            max_length=MAX_SEQ_LENGTH,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        
        # 3. Merge Text + Audio
        for i in range(len(features)):
            # Get the IDs for this specific sentence
            t_ids = text_inputs["input_ids"][i] 
            # Remove padding from the text (we will pad the combined sequence later)
            # We look at attention_mask to find real tokens
            real_len = text_inputs["attention_mask"][i].sum()
            t_ids = t_ids[:real_len]
            
            a_ids = torch.tensor(batch_audio[i], dtype=torch.long)
            
            # Combine: [Text] + [Audio]
            combined_ids = torch.cat([t_ids, a_ids])
            
            # Truncate if the total is too long for the GPU
            if len(combined_ids) > MAX_SEQ_LENGTH:
                combined_ids = combined_ids[:MAX_SEQ_LENGTH]
            
            # Labels: -100 means "Ignore this part". We ignore the Text, we predict the Audio.
            # Label = [-100, -100, ... (text length) ..., Audio_ID, Audio_ID ...]
            label_ids = torch.cat([
                torch.full((len(t_ids),), -100, dtype=torch.long), 
                a_ids
            ])
            # Sync label length with combined_ids length (in case of truncation)
            label_ids = label_ids[:len(combined_ids)]

            input_ids_list.append(combined_ids)
            labels_list.append(label_ids)
            attention_mask_list.append(torch.ones_like(combined_ids))

        # 4. Final Padding (Make them all the same length for the GPU)
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        
        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": padded_attention_mask
        }

    # 5. Training Arguments
    # We switch to standard Trainer
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        data_collator = custom_collate_fn, # <--- Our custom logic works here
        args = TrainingArguments(
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps = 100,
            max_steps = 1000, 
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR,
            save_steps = 200,
            remove_unused_columns = False, # CRITICAL: Tells Trainer NOT to delete 'text' column
        ),
    )

    # 6. TRAIN!
    print("🚂 Choo Choo! Training started...")
    trainer.train()

    # 7. Save
    print(f"💾 Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
