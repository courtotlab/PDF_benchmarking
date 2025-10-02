import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

wanted_count = 1000

model_name = "gemma-3-27b-it" #gemma-3n-E2B-it-finetuned #gemma-3-27b-it
GEMMA_PATH = f"/.mounts/labs/courtotlab/scratch/{model_name}/" #@param ["google/gemma-3n-E2B-it", "google/gemma-3n-E4B-it"]
LORA_PATH = f"/.mounts/labs/courtotlab/scratch/lora/gemma-3-27b-it_1000ct1_lora/"
output_dir = "/u/lsong/labspace/lei_notebook/data/"
pickle_dir = "/u/lsong/labspace/lei_notebook/data/output_general_1000ct.pkl"
max_tokens = 5000

print(f"{output_dir}output_{model_name}_{str(wanted_count)}ct.pkl")

import lei_prompts

from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

# load base model
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained(GEMMA_PATH, local_files_only=True)

model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto", local_files_only=True)

#load LoRA adapter
model.load_adapter(LORA_PATH, adapter_name="adapter_model", peft_config=peft_config)

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

#inport datasets types
from datasets import Dataset, Features, Value, Sequence, Image as HFImage

#schema for dataset
features = Features({
    "user_prompt": Value("string"),
    "system_message": Value("string"),
    "expected_report": Value("string"),
    "image": Sequence(HFImage(decode=True)),
    "mock_uuids": Value("string")
})

#generate dataset
import pickle
with open(pickle_dir, "rb") as f:
    dataset = pickle.load(f)

import copy
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

chat = ChatState(model, processor)

#loop through dataset and get responses
output_dict = {}
for i in range(len(dataset)):
    
    key = dataset_back[i]["mock_uuids"]
     
    #initialize chat state  
    response = chat.send_message(dataset[i]["messages"], max_tokens=max_tokens)
    
    output_dict[key] = {
       "response": response,
       "expected_report": dataset_back[i]["expected_report"]
        }
    print(i)

print(f"done running {model_name} with {wanted_count} cases")
print(f"with {max_tokens} as max_tokens")

import pickle
with open(f"{output_dir}output_{model_name}_{str(wanted_count)}ct1.pkl", "wb") as f:
    pickle.dump(output_dict, f)