#/u/jweile/labspace/projects/lei_mockup_generator/out2 # training data and answers

system_message = """
You are a high-accuracy extractor for clinical, genomic, and diagnostic data from germline lab reports. Your output must follow the JSON schema below exactly. No extra fields, comments, or explanations.
You are efficient and precise, extracting only the required fields from the provided text. You do not want to use extra tokens for explanations or summaries.
"""

user_prompt = """
You are a high-accuracy extractor for clinical, genomic, and diagnostic data from germline lab reports. Your output must follow the JSON schema below exactly. No extra fields, comments, or explanations.
You are efficient and precise, extracting only the required fields from the provided text. You do not want to use extra tokens for explanations or summaries.
Extract and return the following fields from the text or images:

A. Clinical Testing Info
- Sequencing Scope - One or more of: Gene panel, Targeted variant testing, WES, WGS, WTS
- Tested Genes - Gene names tested
- RefSeq mRNA - Ordered with genes; format: NM_000123.3
- Sample Type - One of: Amplified DNA, ctDNA, Total DNA, Total RNA, etc.
- Analysis Type - One or more of: Variant analysis, Karyotyping, Microarray, etc.

B. Report Metadata
- Report Dates - Collected, Received, Verified in YYYY-MM-DD
- Report Type - Pathology or Molecular Genetics
- Testing Context - Clinical or Research
- Ordering Clinic - Include city (e.g. Mount Sinai Hospital (Toronto))
- Testing Laboratory - Include city (e.g. Ontario Cancer Hospital (Toronto))

C. Variant Details
- Variant ID - e.g., OMIM, ClinVar, dbSNP (must match \\d+ or \\w+)
- Gene Symbol - HGNC format
- Transcript ID - e.g., NM_000123.3
- HGVS - Genomic: g., Coding: c., Protein: p.
- Chromosome - chr1~22, chrX, chrY
- Exon - Number(s)
- Zygosity - Homozygous, Heterozygous, etc.
- Interpretation - "Variant of [clinical significance...]"
- MAF - mafac, mafan, mafaf (decimal)
- Type - frameshift, nonsense, synonymous, missense
- mega_hgvs - paste0(
    transcript_id, "(", gene_symbol, "):[",
    hgvsc, "(", hgvsp, ")]:[",
    switch(zygosity,
      homozygous = paste0(hgvsc, "(", hgvsp, ")"),
      heterozygous = "="
    ),
    "]"
  )

Output must match this JSON structure exactly. All fields must be included, even if empty (use ""). Do NOT add summaries or comments. Validate your output for format errors.
Format:
```json
{{
  "report_id": {{
    "date_collected": "",
    "date_received": "",
    "date_verified": "",
    "report_type": "",
    "testing_context": "",
    "ordering_clinic": "",
    "testing_laboratory": "",
    "sequencing_scope": "",
    "tested_genes": {{
      "GENE1": {{
        "gene_symbol": "GENE1",
        "refseq_mrna": "NM_xxxxxxx.x"
      }}
    }},
    "num_tested_genes": "",
    "sample_type": "",
    "analysis_type": "",
    "variants": [
      {{
        "gene_symbol": "",
        "variant_id": "",
        "chromosome": "",
        "hgvsg": "",
        "hgvsc": "",
        "hgvsp": "",
        "transcript_id": "",
        "exon": "",
        "zygosity": "",
        "interpretation": "",
        "mafac": "",
        "mafan": "",
        "mafaf": "",
        "type": "",
        "mega_hgvs": ""
      }}
    ],
    "num_variants": "",
    "reference_genome": ""
  }}
}}
```"""

pkl_dir = "/u/lsong/labspace/lei_notebook/data/"
wanted_count = 1000

# Convert dataset to OAI messages
def format_data(sample):
    try:
        user_content = [{
                            "type": "text",
                            "text": sample["user_prompt"],
                        }]
        for im in sample["image"]:
            user_content.append({"type": "image", "image": im})
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sample["system_message"]}],
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

    #reading section
    mock_data_dir = "/.mounts/labs/courtotlab/private/jweile/projects/lei_mockup_generator/out2/"

    with open(mock_data_dir+'mock_data.json') as f:
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
                  "user_prompt": user_prompt,
                  "system_message": system_message,
                  "expected_report": expected_report,
                  "image": pages
              }
            )
        if len(dataset) >= wanted_count:
            break
        
    print(f"Generated {len(dataset)} samples.")
    return dataset

from datasets import Dataset, Features, Value, Sequence, Image as HFImage

features = Features({
    "user_prompt": Value("string"),
    "system_message": Value("string"),
    "expected_report": Value("string"),
    "image": Sequence(HFImage())
})

dataset = generating_dataset(wanted_count=wanted_count)

dataset = Dataset.from_list(dataset, features=features)

dataset = [format_data(sample) for sample in dataset]  

import pickle

#write generated data to pkl file
with open(f'{pkl_dir}mock_data_train_input_{wanted_count}ct.pkl', 'wb') as f:
    pickle.dump(dataset, f)