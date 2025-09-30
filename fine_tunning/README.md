# Fine tuning Gemma3

### 1. Preparing the input data
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

### 2. Training LoRA
`/u/lsong/labspace/git_repo/PDF_benchmarking/fine_tunning/train_lora.py`

This script train the Gemma3 base model and generate an adapter model to 
`/.mounts/labs/courtotlab/scratch/{model_name}/{lora_dir_name}`

### 3. Preparing the test dictionary
`/u/lsong/labspace/git_repo/PDF_benchmarking/fine_tunning/create_test_dict.py`

Even there is a test dataset generated during the training phase, if any evaluation is needed this is the step to generate another dataset similar to the input, except there is no assistant or expected_report given in the dictionary. The expected_report will be stored in a sepearated key in the final dictionary. Its also output a pickle file

### 4. Running the base LLM with LoRA and test dictionary
`/u/lsong/labspace/git_repo/PDF_benchmarking/fine_tunning/run_fine_tuning_model_only.py`

This script will send PDFs into the LEI and generate an output to `/u/lsong/labspace/lei_notebook/data/output_{model_name}_1000ct1.pkl`. This pickle file will include the prediction results in format of listed dictionaries:
```
[{
    "response": response,
    "expected_report": expected_report  # source from the json generating the PDF
},...]
```

### 5. Analyzing and visualizing of the results
`/u/lsong/labspace/git_repo/PDF_benchmarking/getJSON/compareJSON_linghao.py`

This script will generate a report in `/u/lsong/labspace/lei_notebook/data/`. It includes LLM, False Positives, False Negatives, Incorrect Extractions, Correct Matches, Precision, Recall, F1score, Accuracy, Parsed,H ospital, Prompt, Distressed.
These information is ready for further analysis.

### TODO:
Change expected values and input prompts with hospital specific templates or keys

For example:
The prompt for SickKids should only ask the LEI to extract
```num_tested_genes
reference_genome
gene_symbol
chromosome
hgvsg
hgvsc
hgvsp
transcript_id
exon
zygosity
interpretation
mafaf
mega_hgvs
type
```
As the expected json, it should also only include these information. Although, this change may cause the model not able to understand new hospital's records. But it will likely to increse the performance of the provided templates. 