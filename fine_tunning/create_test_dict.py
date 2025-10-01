mock_data_dir = "/.mounts/labs/courtotlab/private/jweile/projects/lei_mockup_generator/out3/"
output_dir = "/u/lsong/labspace/lei_notebook/data/"
wanted_count = 1000
model_name = "general"

print(f"{output_dir}output_{model_name}_{str(wanted_count)}ct.pkl")

import lei_prompts
    
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
        png_lst = []
        # You can adjust dpi if necessary.
        pages = convert_from_path(pdf_path, poppler_path="/.mounts/labs/courtotlab/private/linghao/lei_notebook/notebook/.pixi/envs/default/bin", dpi=150)
        for count, page in enumerate(pages):
            #convert to RGB first
            page = page.convert("RGB")
            # Save pages as images in the pdf
            png_path = f'{mock_data_dir}png/out_report_{keys}_{count}.png'
            page.save(png_path, 'PNG')
            png_lst.append(png_path)

        expected_report = json.dumps(mock_report_json[keys], ensure_ascii=False)

        # Convert dataset to OAI messages
        # need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
        dataset.append(
              {
                  "user_prompt": lei_prompts.user_prompt(),
                  "system_message": lei_prompts.system_message(),
                  "expected_report": expected_report,
                  "image": png_lst,
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
    "image": Sequence(HFImage(decode=True)),
    "mock_uuids": Value("string")
})

#generate dataset
dataset = generating_dataset(wanted_count=wanted_count)

import pickle

with open(f"{output_dir}output_{model_name}_{str(wanted_count)}ct.pkl", "wb") as f:
    pickle.dump(dataset, f)

print(f"generated {wanted_count} pickle for testing")
