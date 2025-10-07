import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

wanted_count = 500

model_name = "gemma-3n-E2B-it-finetuned" #gemma-3n-E2B-it-finetuned #gemma-3-27b-it
GEMMA_PATH = f"/.mounts/labs/courtotlab/scratch/{model_name}/" #@param ["google/gemma-3n-E2B-it", "google/gemma-3n-E4B-it"]
LORA_PATH = f"/.mounts/labs/courtotlab/scratch/{model_name}/1000ct/"
output_dir = "/u/lsong/labspace/lei_notebook/data/"
mock_data_dir = "/.mounts/labs/courtotlab/private/jweile/projects/lei_mockup_generator/out3/"

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
#pipeline.delete_adapters("adapter_model")
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
            user_content.append({"type": "image", "image": im.convert("RGB")})
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
    
def generating_dataset(wanted_count=9999):
    #This is for generating training data dict, done in local jupyter notebook
    #import section
    import json
    from pdf2image import convert_from_path

    with open(mock_data_dir+'mock_data.json', "r") as f:
        mock_report_json = json.load(f)

    #convert pdf to image
    dataset = []
    
    for keys in mock_report_json.keys():
        pdf_path = mock_data_dir+f"report_{keys}.pdf"
        # You can adjust dpi if necessary.
        pages = convert_from_path(pdf_path, poppler_path="/.mounts/labs/courtotlab/private/linghao/lei_notebook/notebook/.pixi/envs/default/bin", dpi=150, fmt='png')

        expected_report = json.dumps(mock_report_json[keys], ensure_ascii=False)

        # Convert dataset to OAI messages
        # need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
        dataset.append(
              {
                  "user_prompt": lei_prompts.user_prompt(),
                  "system_message": lei_prompts.system_message(),
                  "expected_report": expected_report,
                  "image": pages,
                  "mock_uuids": keys
              }
            )
        if len(dataset) >= wanted_count:
            break
        
    print(f"Generated {len(dataset)} samples.")
    return dataset

#inport datasets types
from datasets import Dataset, Features, Value, Sequence, Image as HFImage

#schema for dataset
features = Features({
    "user_prompt": Value("string"),
    "system_message": Value("string"),
    "expected_report": Value("string"),
    "image": Sequence(HFImage()),
    "mock_uuids": Value("string")
})

#generate dataset
dataset = generating_dataset(wanted_count=wanted_count)

import copy
dataset_back = copy.deepcopy(dataset)

dataset = Dataset.from_list(dataset, features=features)

dataset = [format_data(sample) for sample in dataset]


from IPython.display import Image, display
import io

class ChatState():
  #chat state to hold history and parameters
  def __init__(self, model, processor):
    self.model = model
    self.processor = processor

  def send_message(self, message, max_tokens=1024):

    def display_pil(pil_img):
      buf = io.BytesIO()
      pil_img.save(buf, format="PNG")
      display(Image(data=buf.getvalue()))

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
    response = chat.send_message(dataset[i]["messages"], max_tokens=1024)
    
    output_dict[key] = {
       "response": response,
       "expected_report": dataset_back[i]["expected_report"]
        }

print(f"done running {model_name} with {wanted_count} cases")

import pickle
with open(f"{output_dir}output_{model_name}_{str(wanted_count)}ct.pkl", "wb") as f:
    pickle.dump(output_dict, f)