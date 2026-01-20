import json
import os
import re
import base64
from collections import defaultdict

def read_text_file(file_path):
    """Read the content of a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def read_prompt_file(prompt_path="getJSON/prompt.txt"):
    """Read the prompt file content."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def encode_image_to_base64(image_path):
    """Encode a single image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def group_images_by_source(directory="output_pdfs/images/"):
    """Group images by their source document (hospital report)."""
    try:
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        grouped_images = defaultdict(list)
        
        for image_file in image_files:
            if image_file.startswith("report_"):
                parts = image_file.split("_")
                if len(parts) >= 2:
                    source_name = parts[1]
                    if "distressed" in image_file:
                        source_name += "_distressed"
                    grouped_images[source_name].append(os.path.join(directory, image_file))
        
        # Sort images within each group by page number
        for source_name in grouped_images:
            grouped_images[source_name].sort(key=lambda x: int(x.split("_page_")[1].split(".")[0]) if "_page_" in x else 0)
        
        return dict(grouped_images)
    except Exception as e:
        print(f"Error grouping images from directory {directory}: {e}")
        return {}

def encode_image_group_to_base64(image_paths):
    """Encode multiple images to base64."""
    return [encode_image_to_base64(path) for path in image_paths if encode_image_to_base64(path)]

def extract_json_from_response(response):
    """
    take JSON out from response text looking for the common pattern
    if not available return None
    """
    if not response:
        return None
    
    # Method 1: Try direct parsing after basic cleaning
    cleaned = response.strip()
    
    # Handle common markdown formatting
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    cleaned = cleaned.strip()
    
    # Try direct parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Method 2: Remove comments and try again
    try:
        comment_pattern = r'^\s*//.*$'
        json_nocomment = re.sub(comment_pattern, '', cleaned, flags=re.MULTILINE)
        return json.loads(json_nocomment)
    except json.JSONDecodeError:
        pass
    
    # Method 3: Extract JSON from mixed content (brace matching)
    start_idx = response.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 0
    end_idx = start_idx
    
    for i, char in enumerate(response[start_idx:], start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    if brace_count == 0:
        json_str = response[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try cleaning formatting issues
            try:
                cleaned_json = json_str.replace('\n', ' ').replace('  ', ' ')
                return json.loads(cleaned_json)
            except json.JSONDecodeError:
                pass
    
    #4 try just straight to json raw
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    return None

def save_model_response(model, response, source_file, output_dir):
    """Save the response from a specific model to a JSON file with integrated JSON extraction."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean the source filename to get template name
    clean_name = source_file.split("_")[1]
    print(clean_name)
    append = "distressed" if "distressed" in source_file else ""
    # Create filename
    model_name = model.replace("/", "_").replace(":", "_")
    filename = f"{output_dir}/{clean_name}_{model_name}__{append}__response.json"
    
    try:
        if response is None:
            output_data = {
                "model": model,
                "status": "failed",
                "error": "Model returned None response",
                "timestamp": "2025-06-19",
                "source_file": source_file
            }
        else:
            # Try to extract JSON using unified extraction function
            parsed_json = extract_json_from_response(response)
            
            if parsed_json is not None:
                output_data = {
                    "model": model,
                    "status": "success",
                    "data": parsed_json,
                    "timestamp": "2025-06-19",
                    "source_file": source_file
                }
                print(f"✓ JSON response saved to {filename}")
            else:
                output_data = {
                    "model": model,
                    "status": "json_extraction_failed",
                    "error": "Could not extract valid JSON from response",
                    "raw_response": response,
                    "timestamp": "2025-06-19",
                    "source_file": source_file
                }
                print(f"✗ Raw response saved to {filename} (JSON extraction failed)")
        
        with open(filename, "w") as f:
            json.dump(output_data, f, indent=2)
            
    except Exception as e:
        print(f"Error saving response for {model}: {e}")

def get_text_files_from_directory(directory="output_pdfs/text/"):
    """Get all text files from the specified directory."""
    try:
        return [f for f in os.listdir(directory) if f.endswith('.txt')]
    except Exception as e:
        print(f"Error reading directory {directory}: {e}")
        return []

def process_text_files_with_models(models, output_dir, text_directory="../output_pdfs/text/", prompt_path="run_models/prompt.txt", llm_function=None):
    """Process all text files with the given models using the provided LLM function."""
    import random
    if llm_function is None:
        raise ValueError("llm_function must be provided")
    
    prompt = read_prompt_file(prompt_path)
    os.makedirs("outJSON", exist_ok=True)
    text_out_dirs = [
        "/Users/ayu/PDF_benchmarking/getJSON/outJSON/" + output_dir 
    ]
    output_dir = f"outJSON/{output_dir}"
    
    # Get list of already processed sources from all text output directories
    processed_sources = set()
    
    print(text_out_dirs[0])
    for text_dir in text_out_dirs:
        if os.path.exists(text_dir):
            try:
                text_files = os.listdir(text_dir)
                for filename in text_files:
                    if filename.endswith('.json'):
                        # Extract source ID (UUID) from filename
                        source_id = filename.split('_')[0]
                        if "distressed" in filename.lower():
                            source_id += "_distressed"
                    processed_sources.add(source_id)
            except Exception as e:
                print(f"Warning: Could not read {text_dir}: {e}")
    
    print(f"Found {len(processed_sources)} already processed sources, will skip them.")
    
    text_files = get_text_files_from_directory(text_directory)
    print(f"Found {len(text_files)} text files to process")
    print(f"Running extraction with {len(models)} models...")
    for text_file in text_files:
        print(f"\n{'='*60}")
        print(f"Processing file: {text_file}")
        
        # Extract source ID from text filename (e.g., "report_fakeHospital1__060f50fd...txt")
        s = text_file.replace("report_", "")  # Clean up source name if needed
        # Check if this source has already been processed
        if s in processed_sources:
            print(f"⏭️  Skipping {s} - already processed in vision output directories")
            continue
        
        print(f"{'='*60}")
        
        print(f"{'='*60}")
        
        text = read_text_file(os.path.join(text_directory, text_file))
        if text is None:
            print(f"Failed to read {text_file}, skipping...")
            continue
        
        for model in models:
            print(f"\nProcessing {text_file} with model: {model}")
            response = llm_function(prompt, text, model)
            save_model_response(model, response, text_file, output_dir)
    
    print(f"\n{'='*60}")
    print(f"All files and models completed. Check {output_dir} folder for results.")
    print(f"{'='*60}")
