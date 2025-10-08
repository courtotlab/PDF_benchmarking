import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

wanted_count = 200   #number of samples to run through the model, can be adjusted based on available memory

model_name = "gemma-3-27b-it" #choose from gemma-3n-E2B-it-finetuned, gemma-3-27b-it
GEMMA_PATH = f"/.mounts/labs/courtotlab/scratch/{model_name}/" #@param ["google/gemma-3n-E2B-it", "google/gemma-3n-E4B-it"]
LORA_PATH = "/.mounts/labs/courtotlab/scratch/lora/gemma-3-27b-it1000ct_lora/"  #location of the LoRA adapter model
output_dir = "/u/lsong/labspace/lei_notebook/data/" #directory to save the output pickle file
pickle_dir = f"/u/lsong/labspace/lei_notebook/data/output_general_{wanted_count}ct.pkl"    #location of the test dataset pickle file
max_tokens = 5000   #max tokens to generate, can be adjusted based on the length of the expected reports, if seeing truncated results, increase this number

print(f"{output_dir}output_{model_name}_{str(wanted_count)}ct.pkl")

import lei_prompts

from peft import LoraConfig, PeftModel
import torch
# load base model
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
#inport datasets types
from datasets import Dataset, Features, Value, Sequence, Image as HFImage
import copy
import pickle
from time import time

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

model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, **model_kwargs, local_files_only=True)
print("Model loaded", flush=True)

processor = AutoProcessor.from_pretrained(GEMMA_PATH, local_files_only=True, use_fast=True)

print("start loading LoRA adapter", flush=True)
model = PeftModel.from_pretrained(
    model,
    LORA_PATH,
    adapter_name="adapter_model",
    is_trainable=False,              # set True only if you intend to train
    local_files_only=True,
)
print("done loading LoRA adapter", flush=True)

print(f"Device: {model.device}")
print(f"DType: {model.dtype}")

#load testing datamock_data_dir = "/.mounts/labs/courtotlab/private/jweile/projects/lei_mockup_generator/out2/"

# Convert dataset to OAI messages
def format_data(sample):
    try:
        user_content = [{
                            "type": "text",
                            "text": lei_prompts.user_prompt(),
                        }]
        for im in sample["image"]:
            user_content.append({"type": "image", "image": im})
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": lei_prompts.system_message()}],
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }
    except Exception as e:
        print(e,sample)
        return None


#schema for dataset
features = Features({
    "user_prompt": Value("string"),
    "system_message": Value("string"),
    "expected_report": Value("string"),
    "image": Sequence(HFImage(decode=True)),
    "mock_uuids": Value("string")
})

#generate dataset
print("start loading dataset")

with open(pickle_dir, "rb") as f:
    dataset = pickle.load(f)

dataset_back = copy.deepcopy(dataset)

dataset = Dataset.from_list(dataset, features=features)

dataset = [format_data(sample) for sample in dataset]

class ChatState():
  #chat state to hold history and parameters
  def __init__(self, model, processor):
    self.model = model
    self.processor = processor

  def send_message(self, message, max_tokens=5000):

    input_ids = self.processor.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=True, #used to be True
        return_dict=True,
        return_tensors="pt",
    )
    input_len = input_ids["input_ids"].shape[-1]

    input_ids = input_ids.to(self.model.device, dtype=model.dtype)
    outputs = self.model.generate(
        **input_ids,
        max_new_tokens=max_tokens,
        disable_compile=True
    )
    text = self.processor.batch_decode(
        outputs[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    return text[0]

print("start running model", flush=True)
chat = ChatState(model, processor)

#loop through dataset and get responses
output_dict = {}
for i in range(len(dataset)):
    
    key = dataset_back[i]["mock_uuids"]
    print(f"Processing {i}th case with key {key}", flush=True)

    #initialize chat state  
    response = chat.send_message(dataset[i]["messages"], max_tokens=max_tokens)
    
    output_dict[key] = {
       "response": response,
       "expected_report": dataset_back[i]["expected_report"]
        }
    print(i)

print(f"done running {model_name} with {wanted_count} cases", flush=True)
print(f"with {max_tokens} as max_tokens", flush=True)

with open(f"{output_dir}output_{model_name}_{str(wanted_count)}ct.pkl", "wb") as f:
    pickle.dump(output_dict, f)