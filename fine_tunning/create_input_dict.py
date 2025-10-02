#/u/jweile/labspace/projects/lei_mockup_generator/out2 # training data and answers

pkl_dir = "/u/lsong/labspace/lei_notebook/data/"
wanted_count = 1000

# Convert dataset to OAI messages
def format_data(sample):
    import lei_prompts
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
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["expected_report"]}], #this is expected results
                },
            ],
        }
    except Exception as e:
        print(e,sample)
        return None

def generating_dataset(wanted_count=9999):
    #This is for generating training data dict, done in local jupyter notebook
    #import section
    import json
    from pdf2image import convert_from_path
    from getJSON.compareJSON_linghao import filter_template, extract_hospital_from_template, template_to_string

    #reading section
    mock_data_dir = "/.mounts/labs/courtotlab/private/jweile/projects/lei_mockup_generator/out2/"

    with open(mock_data_dir+'mock_data.json') as f:
        mock_report_json = json.load(f)

    #convert pdf to image
    dataset = []
    
    for keys in mock_report_json.keys():
        pdf_path = mock_data_dir+f"report_{keys}.pdf"
        png_lst = []
        # You can adjust dpi if necessary.
        pages = convert_from_path(pdf_path, poppler_path="/.mounts/labs/courtotlab/private/linghao/lei_notebook/notebook/.pixi/envs/default/bin", dpi=150, fmt='png')
        
        for count, page in enumerate(pages):
            #convert to RGB first
            page = page.convert("RGB")
            # Save pages as images in the pdf
            png_path = f'{mock_data_dir}png/out_report_{keys}_{count}.png'
            page.save(png_path, 'PNG')
            png_lst.append(png_path)

        #filter the json to remove keys not included in the records
        expected_report = mock_report_json[keys]
        expected_report, hospital = extract_hospital_from_template(expected_report)
        filtered_report = filter_template(expected_report, hospital)   
        filtered_report = template_to_string(filtered_report)

        # Convert dataset to OAI messages
        # need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
        dataset.append(
              {
                  "expected_report": filtered_report,
                  "image": png_lst
              }
            )
        if len(dataset) >= wanted_count:
            break
        
    print(f"Generated {len(dataset)} samples.")
    return dataset

from datasets import Dataset, Features, Value, Sequence, Image as HFImage

features = Features({
    "expected_report": Value("string"),
    "image": Sequence(HFImage(decode=True))
})

dataset = generating_dataset(wanted_count=wanted_count)

dataset = Dataset.from_list(dataset, features=features)

dataset = [format_data(sample) for sample in dataset]  

import pickle

#write generated data to pkl file
with open(f'{pkl_dir}mock_data_train_input_{wanted_count}ct.pkl', 'wb') as f:
    pickle.dump(dataset, f)