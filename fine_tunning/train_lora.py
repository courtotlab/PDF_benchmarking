#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to use

#2 reasons for validation loss is consistantly lower than training loss
#1. training batch is much larger than validation batch, so training loss is training_batch * training_loss, validation loss is validation_batch * validation_loss
#2. checkout the dropout rate in the validation. The dropout rate is 0.05, so 5% of the model is not used in training, but all the model is used in validation, so the validation loss is lower than training loss

import pickle
import random
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig
from PIL import Image
from trl import SFTTrainer
import time
import json

wanted_count = 1000   #number of samples to use for training LoRA
#loading training data
pkl_dir = "/u/lsong/labspace/lei_notebook/data/" #directory to load input pkl files
# Hugging Face model id
model_name = "gemma-3-27b-it" # select from google/gemma-3-27-it, gemma-3n-E2B-it-finetuned
model_path = f"/.mounts/labs/courtotlab/scratch/{model_name}/" #directory to load base model, the model should be downloaded from Hugging Face by hf cli first
lora_output_dir = "/.mounts/labs/courtotlab/scratch/lora/" #directory to save LoRA adapter model
lora_name = f"{lora_output_dir}{model_name}{wanted_count}ct_lora" #name of the LoRA adapter model
print(model_path)


with open(f"{pkl_dir}mock_data_train_input_{wanted_count}ct.pkl", "rb") as f:
    dataset = pickle.load(f)

random.shuffle(dataset)

train_dataset = dataset[:int((0.9*len(dataset)))]
test_dataset = dataset[int((0.9*len(dataset))):]

print("train dataset length: ", len(train_dataset))
print("test dataset length: ", len(test_dataset))

# Check if GPU benefits from bfloat16
print(torch.cuda.get_device_capability())
if torch.cuda.get_device_capability()[0] < 8:
    bnb_4bit_compute_dtype = torch.float16
    attn_impl = "eager"
    #    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")
else:
    bnb_4bit_compute_dtype = torch.bfloat16 #bfloat16 is only supported on Ampere or newer GPU
    attn_impl = "eager"

# Define model init arguments
model_kwargs = dict(
    attn_implementation=attn_impl, # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=bnb_4bit_compute_dtype,
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig int-4 config
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs, local_files_only=True)
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

peft_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

#define special hyperparameters

args = SFTConfig(
    output_dir=lora_output_dir,             # directory to save and repository id
    max_length=None,                        # max sequence length for model and packing of the dataset
    packing=False,                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    per_device_eval_batch_size=2,           # batch size for evaluation
    gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # used to use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="steps",                  # save checkpoint every epoch
    save_steps = 100,                       # save checkpoint every 100 steps if save_strategy is "steps"
    learning_rate=1e-4,                     # learning rate, based on QLoRA paper
    fp16=True if bnb_4bit_compute_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if bnb_4bit_compute_dtype == torch.bfloat16 else False,   # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                      # push model to hub
    do_eval=True,                           # enable evaluation
    eval_strategy="steps",                  # after x steps do evaluation
    eval_steps=10,                          # evaluation steps
    report_to="tensorboard",                # report metrics to tensorboard
    dataset_kwargs={
        "add_special_tokens": False,        # We template with special tokens
        "append_concat_token": True,        # Add EOS token as separator token between examples
    }
)

args.remove_unused_columns = False # important for collator

def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element.keys() or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element.keys():
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs if image_inputs else None)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()
    tok = processor.tokenizer
    
    special_ids = set()

    for key in ["boi_token", "eoi_token", "image_token", "image_pad_token", "image_patch_token"]:
        tok_str = tok.special_tokens_map.get(key)
        if tok_str is not None:
            special_ids.add(tok.convert_tokens_to_ids(tok_str))

    # Some tokenizers expose many patch tokens; if there is a contiguous block or a getter, add them here.
    if hasattr(tok, "image_token_id") and tok.image_token_id is not None:
        special_ids.add(tok.image_token_id)

    # Mask tokens for not being used in the loss computation
    labels[labels == tok.pad_token_id] = -100
    for sid in special_ids:
        labels[labels == sid] = -100

    batch["labels"] = labels
    return batch

#start SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# Import the time library
start = time.time() 

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model
trainer.save_model(lora_name)

log_history = trainer.state.log_history

with open(f"{lora_output_dir}/training.log", "w") as f:
    for rec in log_history:
        f.write(json.dumps(rec) + "\n")

end = time.time()
length = end - start

print("trainning takes: ",length," seconds")

# free the memory
del model
del trainer
torch.cuda.empty_cache()

#test section
