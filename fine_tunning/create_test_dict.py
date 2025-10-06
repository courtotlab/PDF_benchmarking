#import section
import sys
#enable the scirpt to access local modules
script_path = "/u/lsong/labspace/git_repo/PDF_benchmarking" #current project path in order to import local modules
sys.path.append(script_path)

import json
from pdf2image import convert_from_path
from getJSON.compareJSON_linghao import extract_hospital_from_template
import lei_prompts
#import datasets types
from datasets import Features, Value, Sequence, Image as HFImage
import os
import pickle


mock_data_dir = "/.mounts/labs/courtotlab/private/jweile/projects/lei_mockup_generator/out3/" #directory including mockup pdfs and jsons
output_dir = "/u/lsong/labspace/lei_notebook/data/" #directory to save the output pickle file   
wanted_count = 1000     #number of samples to convert
model_name = "general"  #name of the model. because this is generating test dataset for all model uses, so it is "general"
template_path=f"{script_path}/getJSON/hospitals/" #path to the hospital keys templates


print(f"{output_dir}output_{model_name}_{str(wanted_count)}ct.pkl")
    
def generating_dataset(wanted_count=9999):
    #This is for generating training data dict, done in local jupyter notebook

    with open(mock_data_dir+'mock_data.json', "r") as f:
        mock_report_json = json.load(f)

    #convert pdf to image
    dataset = []
    
    for keys in mock_report_json.keys():
        pdf_path = mock_data_dir+f"report_{keys}.pdf"
        png_lst = []
        convert = False
        # You can adjust dpi if necessary.
        if convert == True:
            pages = convert_from_path(pdf_path, poppler_path="/.mounts/labs/courtotlab/private/linghao/lei_notebook/notebook/.pixi/envs/default/bin", dpi=150)
            for count, page in enumerate(pages):
                #convert to RGB first
                page = page.convert("RGB")
                # Save pages as images in the pdf
                png_path = f'{mock_data_dir}png/out_report_{keys}_{count}.png'
                page.save(png_path, 'PNG')
                png_lst.append(png_path)
        else:
            # Use pre-converted images if available
            count = 0
            while True:
                png_path = f'{mock_data_dir}png/out_report_{keys}_{count}.png'
                if os.path.exists(png_path):
                    png_lst.append(png_path)
                    count += 1
                else:
                    break

        #filter the json to remove keys not included in the records
        expected_report = mock_report_json[keys]
        filtered_report, hospital = extract_hospital_from_template(expected_report, template_path=template_path)

        # Convert dataset to OAI messages
        # need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
        dataset.append(
              {
                  "user_prompt": lei_prompts.user_prompt(),
                  "system_message": lei_prompts.system_message(),
                  "expected_report": filtered_report,
                  "image": png_lst,
                  "mock_uuids": keys
              }
            )
        if len(dataset) >= wanted_count:
            break
        
    print(f"Generated {len(dataset)} samples.")
    return dataset

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

with open(f"{output_dir}output_{model_name}_{str(wanted_count)}ct.pkl", "wb") as f:
    pickle.dump(dataset, f)

print(f"generated pickle include {wanted_count} samples for testing")
