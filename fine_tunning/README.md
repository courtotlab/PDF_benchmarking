# Fine tuning Gemma3

### 1. preparing input data
`/u/lsong/labspace/git_repo/PDF_benchmarking/fine_tunning/create_input_dict.py`

This script will generate a pickle file that stores 1000 training ready samples.
They are all in the format of 
``` 
    {           
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": sample["system_message"]}],
            },
            {
                "role": "user",
                "content": [{
                        "type": "text",
                        "text": sample["user_prompt"],
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    }, #more images here
                    ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["expected_report"]}], #this is expected results
            },
        ],
    }
```

user prompt and system message stored in 
`/u/lsong/labspace/git_repo/PDF_benchmarking/fine_tunning/lei_prompts.py`

### 2. training LoRA
`/u/lsong/labspace/git_repo/PDF_benchmarking/fine_tunning/train_lora.py`

This script train the Gemma3 base model and generate an adapter model to 
`/.mounts/labs/courtotlab/scratch/{model_name}/{lora_dir_name}`

### 3. prepare test dictionary
`/u/lsong/labspace/git_repo/PDF_benchmarking/fine_tunning/create_test_dict.py`

Even there is a test dataset generated during the training phase, if any evaluation is needed this is the step to generate another dataset similar to the input, except there is no assistant or expected_report given in the dictionary. The expected_report will be stored in a sepearated key in the final dictionary. Its also output a pickle file

### 4. running the base LLM with LoRA and test dictionary
`/u/lsong/labspace/git_repo/PDF_benchmarking/getJSON/compareJSON_linghao.py`

This script will generate a report in `/u/lsong/labspace/lei_notebook/data/`. It includes LLM, False Positives, False Negatives, Incorrect Extractions, Correct Matches, Precision, Recall, F1score, Accuracy, Parsed,H ospital, Prompt, Distressed.
These information is ready for further analysis.

