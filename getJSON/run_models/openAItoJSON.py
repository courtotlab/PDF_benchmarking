from jsonLLM import(
    extract_json_from_response, 
    save_model_response, 
    process_text_files_with_models,
    group_images_by_source,
    encode_image_group_to_base64,
    read_prompt_file, 
)
import base64
from openai import OpenAI
import os
import time
from PIL import Image
import io

API = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=API)

def textToLLM(prompt, text, model):
    """
    Function to send text to a specific OpenAI model and get response.
    Returns the response text or None if failed.
    """
    print(f"Trying model: {model}")
            
    response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nText to analyze:\n{text}"
                    }
                ],
                temperature=0.0
            )
            
    print(f"Success with model: {model}")
    return response.choices[0].message.content
            
def reduce_image_token_size(image_path, max_dim=512, quality=20):
    """
    Resize image to fit within max_dim x max_dim and compress to reduce token size.
    """
    with Image.open(image_path) as img:
        # calculate resize ratio (only shrink)
        ratio = min(max_dim / img.width, max_dim / img.height, 1)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format='JPEG', quality=quality)
        return buffer.getvalue()

def imageGroupToLLM(prompt, image_group, model):
    """
    Function to send a group of images (multiple pages) to a vision-capable LLM model.
    Returns the response text or None if failed.
    """
    try:
        print(f"Trying model: {model} with {len(image_group)} images")
        
        images = []
        for image_path in image_group:
            # Reduce image token size by downsizing and compressing
            image_bytes = reduce_image_token_size(image_path)
            image_data = base64.b64encode(image_bytes).decode('utf-8')
            images.append(image_data)
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nImages to analyze:\n{images}"
                }
            ],
            "temperature": 0.0
        }
        response = client.chat.completions.create(**payload)
        print(f"Success with model: {model}")
        # use attribute access to avoid subscriptable error
        return response.choices[0].message.content
        
    except Exception as e:
        error_str = str(e)
        print(f"✗ Error with model {model}: {error_str}")
        return None

def process_grouped_images_with_models(models, output_dir, image_directory="../output_pdfs/images/", prompt_path="run_models/prompt.txt"):
    """    
    Args:
        models: List of vision-capable model names
        output_dir: Directory to save results
        image_directory: Directory containing images
        prompt_path: Path to the prompt file
    """
    # Read the prompt file content
    prompt = read_prompt_file(prompt_path)
    processed_sources = set()
    vision_out_dirs = [
        "/Users/ayu/PDF_benchmarking/getJSON/outJSON/" + output_dir
    ]
    for vision_dir in vision_out_dirs:
        if os.path.exists(vision_dir):
            try:
                vision_files = os.listdir(vision_dir)
                for filename in vision_files:
                    if filename.endswith('.json'):
                        # Extract source ID (UUID) from filename
                        source_id = filename.split('_')[0]
                        if "distressed" in filename.lower():
                            source_id += "_distressed"
                    processed_sources.add(source_id)
            except Exception as e:
                print(f"Warning: Could not read {vision_dir}: {e}")
    
    print(f"Found {len(processed_sources)} already processed sources, will skip them.")
    # Create output directory
    os.makedirs("outJSON", exist_ok=True)
    full_output_dir = "outJSON/" + output_dir
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Group images by source document
    grouped_images = group_images_by_source(image_directory)
    
    print(f"Found {len(grouped_images)} source documents with images")
    print(f"Running extraction with {len(models)} vision models...")
    
    # Process each group of images
    for source_name, image_paths in grouped_images.items():
        print(f"\n{'='*60}")
        print(f"Processing source: {source_name} ({len(image_paths)} pages)")
        print(f"{'='*60}")
        if source_name in processed_sources:
            print(f"Skipping already processed source: {source_name}")
            continue
        # Try each vision model for this image group
        for model in models:
            print(f"\nProcessing {source_name} with model: {model}")
            
            response = imageGroupToLLM(prompt, image_paths, model)
            # Use source name as the "file" identifier for consistent naming
            save_model_response(model, response, f"report_{source_name}", full_output_dir)
    
    print(f"\n{'='*60}")
    print(f"All image groups and models completed. Check {full_output_dir} folder for results.")
    print(f"{'='*60}")

def main():
    os.chdir("/Users/ayu/PDF_benchmarking/getJSON")  # Ensure we are in the correct directory
    # List of OpenAI models to try
    OPENAI_MODELS = [
        "gpt-4.1-mini",
        #"gpt-4.1-nano", 
        #"gpt-4.1"
        # Add more OpenAI models as needed
    ]
    
    print("Starting OpenAI model processing...")
    print(f"Available models: {OPENAI_MODELS}")
    
    # Use the shared processing function
    '''process_text_files_with_models(
        models=OPENAI_MODELS,
        output_dir="OpenAIOut/", 
        text_directory="../output_pdfs/text/", 
        llm_function=textToLLM
    )
    
    '''
    '''process_grouped_images_with_models(
        models=OPENAI_MODELS,
        output_dir="OpenAIVisionOutNP/",
        image_directory="../output_pdfs/images/",
        prompt_path="run_models/NERprompt.txt"
    )'''
    
    '''process_grouped_images_with_models(
        models=OPENAI_MODELS,
        output_dir="OpenAIVisionOut/",
        image_directory="../output_pdfs/images/",
    )'''
    process_text_files_with_models(
        models=OPENAI_MODELS,
        output_dir="OpenAIOutNP/", 
        text_directory="../output_pdfs/text/", 
        llm_function=textToLLM,
        prompt_path="run_models/NERprompt.txt"
    )
    print("Processing completed.")

def main3():
    #main()
    os.chdir("/Users/ayu/PDF_benchmarking/getJSON")  # Ensure we are in the correct directory
    # List of OpenAI models to try
    OPENAI_MODELS = [
        "gpt-4.1-mini",
        #"gpt-4.1-nano", 
        #"gpt-4.1"
        # Add more OpenAI models as needed
    ]
    process_text_files_with_models(
        models=OPENAI_MODELS,
        output_dir="OpenAIOutNP/", 
        text_directory="../output_pdfs/text/", 
        llm_function=textToLLM,
        prompt_path="run_models/NERprompt.txt"
    )
    print("Processing completed.")
main()
