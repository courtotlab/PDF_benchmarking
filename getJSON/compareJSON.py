import json
import os
import pprint
import copy as c
import pandas as pd  
from collections import defaultdict
import re

def filter_template(template, reportName):
    """
    Return a copy of `template` with only those entries whose key
    appears in reportName.txt (nested dicts/lists pruned similarly).
    """
    reportName = reportName.lower()
    if reportName == "fakehospital2":
        return template
    # Apply filtering to both hospitals based on their respective content files
    report_file = f"hospitals/{reportName}.txt"
    
    # Check if the report file exists
    if not os.path.exists(report_file):
        print(f"Warning: {report_file} not found, using full template")
        return template

    # read the report text once
    with open(report_file, "r", encoding="utf-8") as f:
        report_data = f.read()

    filtered = c.deepcopy(template)

    def recurse(obj):
        if isinstance(obj, dict):
            for k in list(obj):
                v = obj[k]
                if isinstance(v, (dict, list)):
                    recurse(v)
                    if not v:
                        obj[k] = "" #make the not listed keys an empty string
                else:
                    # drop any leaf whose key is not in the report text
                    if k not in report_data:
                        obj[k] = ""

        elif isinstance(obj, list):
            for item in list(obj):
                if isinstance(item, (dict, list)):
                    recurse(item)
                    if not item:
                        index = obj.index(item)
                        obj[index] = ""  # make the not listed items an empty string

    recurse(filtered)
    return filtered

def key_num(d):
    return sum(len(d) for d in d.values() if isinstance(d, dict))

def template_to_string(template):
    for a,v in template.items():
        if isinstance(v, int):
            template[a] = str(v).lower()
        if isinstance(v, list):
            for variantsdic in v:
                for key, value in variantsdic.items():
                    if isinstance(value, int) or isinstance(value, float):
                        variantsdic[key] = str(value).lower()
    #print("Template values converted to strings where applicable.")
    return template

def dict_to_lowercase(obj):
    """
    Recursively convert all string values in a dictionary/list structure to lowercase.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                obj[key] = value.lower()
            elif isinstance(value, (dict, list)):
                dict_to_lowercase(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                obj[i] = item.lower()
            elif isinstance(item, (dict, list)):
                dict_to_lowercase(item)
    return obj

def count_all_template_values(template):

    """
    Count ALL string and numeric values in the template, including empty ones and those in nested structures.
    This gives us the true total that should be used as the denominator for accuracy.
    """
    def count_recursive(obj):
        count = 0
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, (str, int, float)):
                    count += 1  # Count all strings, numbers, even empty ones
                elif isinstance(value, (dict, list)):
                    count += count_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (str, int, float)):
                    count += 1
                elif isinstance(item, (dict, list)):
                    count += count_recursive(item)
        return count
    
    total = count_recursive(template)
    #print(f"Total template values (including empty): {total}")
    return total

def normalizeNames(x):
    import re
    '''
    Normalize names by removing special characters and converting to lowercase. 
    bAsed off of ohcrn_lei eval compare_json.py same function
    '''
    # normalize hgvs by removing prefixes and brackets
    x = re.sub(r"Chr.+:g\.", "", x)
    x = re.sub(r"^g\.|^c\.|^p\.", "", x)
    x = re.sub(r"^\(|\)$", "", x)
    if re.match(r"^\d+-\d+$", x):
      x = re.sub(r"-\d+$", "", x)
    if re.match(r"^g+[0-9]", x):
        x = re.sub(r"^g","", x)

    if re.match(r"^c+[0-9]", x):
        x = re.sub(r"^c","", x)

    if re.match(r"^p+[0-9]", x):
        x = re.sub(r"^c","", x)
    # normalize omim, clinvar, dbsnp
    x = re.sub(r"^OMIM\D+", "", x)
    x = re.sub(r"^Clinvar[^V]*", "", x, flags=re.IGNORECASE)
    x = re.sub(r"^dbSNP[^r]*", "", x, flags=re.IGNORECASE)
    # normalize chromosomes
    if re.match(r"^ChrX$|^ChrY$|^Chr\d$", x, flags=re.IGNORECASE):
      x = re.sub(r"Chr", "", x, flags=re.IGNORECASE)
    # remove location tags
    x = re.sub(
      r" ?\(Toronto$| ?\(Kingston$| ?\(Ottawa| ?\(London| ?\(Orillia.*| ?\(Mississauga",
      "",
      x,
      flags=re.IGNORECASE,
    )
    # convert everything to uppercase for case insensitive matching
    x = x.lower()
    # remove extra spaces
    x = re.sub(r"\s+", " ", x).strip()

    if "e-" in x or "e+" in x:
        # Handle scientific notation by converting to float and back to string
        try:
            num = float(x)
            x = str(num)
        except ValueError:
            pass
        
    return x


def compare_dict_keys_and_values(dict1, dict2, path=""):
    """
    Compare two dictionaries by checking key matches and recursively comparing values.
    Enhanced to better handle missing keys, null/None values, and comprehensive error categorization.
    """
    differences = []
    if not (isinstance(dict1, dict) and isinstance(dict2, dict)):
        differences.append(f"Type mismatch at {path}: expected dictionaries")
        return differences
    
    # Get all keys from both dictionaries
    template_keys = set(dict1.keys())
    extracted_keys = set(dict2.keys())
    all_keys = template_keys | extracted_keys
    
    # First, identify missing keys explicitly
    missing_in_extraction = template_keys - extracted_keys
    extra_in_extraction = extracted_keys - template_keys
    
    # Handle missing keys in extraction (these are always false negatives)
    for key in missing_in_extraction:
        current_path = f"{path}.{key}" if path else key
        template_value = dict1[key]
        differences.append(f"FALSE NEGATIVE at {current_path}: missing key '{key}' with template value '{template_value}'")
    
    # Handle extra keys in extraction (these are potential false positives)
    for key in extra_in_extraction:
        if key == "report_id":
            # Skip report_id as it's not a meaningful extraction key
            continue
        current_path = f"{path}.{key}" if path else key
        differences.append(f"FALSE POSITIVE at {current_path}: extra key with structure")
    
    # Now compare keys that exist in both dictionaries
    common_keys = template_keys & extracted_keys
    
    for key in common_keys:
        current_path = f"{path}.{key}" if path else key
        val1 = dict1[key]
        val2 = dict2[key]
        
        # Handle null/None values consistently
        val1 = val1 if val1 is not None else ""
        val2 = val2 if val2 is not None else ""
        
        # Both values are strings (or converted from null) - exact comparison
        differences = compare_vals(val1, val2, current_path, differences)
    
    return differences

def compare_string(str1, str2, path=""):
    """
    Compare two strings for equality with better handling of empty values and placeholders.
    Treats null/None values as equivalent to empty strings.
    """
    differences = []
    
    # Handle null/None values - treat as empty strings
    str1 = str1 if str1 is not None else ""
    str2 = str2 if str2 is not None else ""
    
    # Normalize strings for comparison
    norm_str1 = normalizeNames(str1.strip()) if str1 else ""
    norm_str2 = normalizeNames(str2.strip()) if str2 else ""
    
    # Perfect match (including both empty/null)
    if norm_str1 == norm_str2:
        differences.append(f"EXACT MATCH at {path}: both are '{str1}'")
        return differences
    
    # Handle scientific notation comparison
    try:
        if norm_str1 and norm_str2 and abs(float(norm_str1) - float(norm_str2)) < 1e-10:
            differences.append(f"EXACT MATCH at {path}: both are '{str1}' (scientific notation match)")
            return differences
    except ValueError:
        pass
    
    # Categorize mismatches more precisely
    if not norm_str1 and not norm_str2:
        # Both empty/null - this is a match
        differences.append(f"EXACT MATCH at {path}: both empty/null")
    elif not norm_str1 and norm_str2:
        # Template empty/null but extraction has value
       differences.append(f"FALSE POSITIVE at {path}: expected empty/null but got '{str2}'")    
    elif norm_str1 and not norm_str2:
        # Template has value but extraction empty/null
        differences.append(f"FALSE NEGATIVE at {path}: expected '{str1}' but got empty/null")
    elif norm_str2 in norm_str1:
        differences.append(f"{len(norm_str2)/len(norm_str1):.2f} PARTIAL MATCH at {path}: '{norm_str2}' found in '{norm_str1}'")
    else:
        differences.append(f"VALUE MISMATCH at {path}: expected '{str1}' but got '{str2}'")
    
    return differences
    
def compare_list_values(list1, list2, path=""):
    """
    Compare two lists by comparing values at corresponding positions.
    """
    differences = []

    
    # Compare corresponding elements
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        current_path = f"{path}[{i}]"
        val1 = list1[i]
        val2 = list2[i]
        differences = compare_vals(val1, val2, current_path, differences)
    
    return differences

def compare_vals(val1, val2, current_path, differences):
    if isinstance(val1, (str,int,float)) and isinstance(val2, (str, int, float)):
        val1 = str(val1).strip() if val1 is not None else ""
        val2 = str(val2).strip() if val2 is not None else ""
        differences.extend(compare_string(val1, val2, current_path))
    elif isinstance(val1, dict) and isinstance(val2, dict):
        differences.extend(compare_dict_keys_and_values(val1, val2, current_path))
    elif isinstance(val1, list) and isinstance(val2, list):
        differences.extend(compare_list_values(val1, val2, current_path))
    elif isinstance(val1, list) and isinstance(val2, dict):
        vals = []
        for k,v in val2.items():
            if isinstance(v,dict):
                vals.append(v)
            differences.extend(compare_list_values(val1, vals, current_path))
    elif isinstance(val1, dict) and isinstance(val2, list): 
        vals = []
        for k,v in val1.values():
            if isinstance(v,dict):
                vals.append(v)
            differences.extend(compare_list_values(vals, val2, current_path))
    elif isinstance(val1, (str, int, float)) and isinstance(val2, list):
        if len(val2) == 1 and isinstance(val2[0], (str, int, float)):
            # If list has one item, compare directly
            differences.extend(compare_string(str(val1), str(val2[0]), current_path))
        elif len(val2) > 1:
            if val1 in val2:
                temp = 1/len(val2)
                differences.append(f"{temp} PARTIAL MATCH at {current_path}: '{val1}' found in list")
            else:
                differences.append(f"VALUE MISMATCH at {current_path}: expected '{val1}' but got list containing {val2}")
    else:
        differences.append(f"Type mismatch at {current_path}: {type(val1).__name__} vs {type(val2).__name__}")
    return differences

def compare_values_with_template(template, data):
    """
    Compare data with template using strict dictionary key matching.
    Returns count of matching vs mismatching values with proper categorization.
    Only EXACT matches count as correct extractions - missing keys are always false negatives.
    """
    differences = compare_dict_keys_and_values(template, data)
    
    # Calculate all template values
    total_template_values = count_all_template_values(template)
    
    fp = 0  # False positives: extra fields, placeholder values, or wrong values where template is empty
    fn = 0  # False negatives: missing fields or empty values where template has content
    ic = 0  # Incorrect extractions: wrong values where both template and extraction have content
    correct_matches = 0  # ONLY exact matches of non-empty values
    count = 0.0

    
    for diff in differences:
        diff_lower = diff.lower()
        
        # Only count as correct if it's an exact match of meaningful content
        if "exact match" in diff_lower:
            correct_matches += 1
            count += 1
        elif "false positive" in diff_lower:
            fp += 1
        elif "false negative" in diff_lower:
            fn += 1
            count += 1
        elif any(term in diff_lower for term in ["value mismatch", "type mismatch"]):
            ic += 1
            count += 1
        elif "partial match" in diff_lower:
            #print(diff_lower)
            #print(diff.lower().split(" "))
            location = diff_lower.split(" at ")[1]
            temp = 0.0
            valid = ["date", 'report_type', "testing context", 
             'ordering_clinic', 'testing_laboratory', "sequencing_scope", 
             "sample_type", "analysis_type", "reference_genome"]
            if location in valid:
                temp = float(diff_lower.split(" ")[0])
                print(f"Partial match at {location} counted as {temp}")
            correct_matches += temp
            ic += 1 - temp
            count += temp
    
    # Debug information
    #print(f"  Categorization: Correct={correct_matches}, FP={fp}, FN={fn}, IC={ic}")
    #print(f"  Total accounted: {count}/{total_template_values}")
    
    # Any unaccounted template values are missing fields (false negatives)
    if count < total_template_values:
        missing_fields = total_template_values - count
        fn += missing_fields
        #print(f"  Added {missing_fields} missing fields as false negatives")
    
    return correct_matches, fp, fn, ic, total_template_values, differences

def compare_gliner_output(template, data, hospital, source, model_name, json_file):
    """
    Fixed GLiNER comparison function that handles GLiNER's actual output format.
    GLiNER outputs flat key-value pairs with concatenated values.
    """
    # Create flattened template for GLiNER comparison
    flat_template = flatten_template_for_gliner2(template)
    
    # Get the actual GLiNER data (not nested under report_id)
    gliner_data = data.get("data", {}) if isinstance(data, dict) else data
    
    total_values = len(flat_template)
   #print(f"Flattened template values for GLiNER: {total_values}")
    #print(f"GLiNER extracted fields: {len(gliner_data)}")
    
    # Special GLiNER comparison - look for partial matches across all values
    perfect_matches = 0
    partial_matches = 0
    
    # For each template field, check if any GLiNER value contains relevant info
    for template_key, template_val in flat_template.items():
        if not template_val or template_val.strip() == "":
            continue  # Skip empty template values
            
        found_match = False
        template_norm = normalizeNames(template_val.strip().lower())
        
        # Check for exact matches first
        for gliner_key, gliner_val in gliner_data.items():
            if not gliner_val:
                continue
            gliner_norm = normalizeNames(str(gliner_val).strip().lower())
            
            if template_norm == gliner_norm:
                perfect_matches += 1
                found_match = True
                #print(f"PERFECT: {template_key} = '{template_val}' found in {gliner_key}")
                break
        
        # If no perfect match, look for partial matches
        if not found_match:
            for gliner_key, gliner_val in gliner_data.items():
                if not gliner_val:
                    continue
                gliner_text = str(gliner_val).lower()
                
                # Check if template value appears anywhere in GLiNER output
                if template_norm in gliner_text or any(word in gliner_text for word in template_norm.split() if len(word) > 2):
                    partial_matches += 1/len(gliner_text.split())
                    found_match = True
                    break
        
        #if not found_match:
            #print(f"MISSING: {template_key} = '{template_val}' not found anywhere")
    #partial matches count as proportion with an exact match
    correct_matches = perfect_matches + partial_matches  # Each partial match is porportional to its length
    fn = max(0, total_values - perfect_matches - partial_matches)  # Missing template fields
    fp = max(0, len(gliner_data) - perfect_matches - partial_matches)  # Extra GLiNER fields
    ic = 0  # For GLiNER, we don't count incorrect extractions separately
    
    # Calculate metrics using standard formulas.
    # Precision = (weighted correct) / (total extracted)
    # Recall = (weighted correct) / (total in template)
    accuracy = (correct_matches / total_values * 100) if total_values > 0 else 0
    precision = (correct_matches / len(gliner_data) * 100) if len(gliner_data) > 0 else 0
    recall = (correct_matches / total_values * 100) if total_values > 0 else 0
    f1score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    #print(f"FP: {fp}, FN: {fn}, IC: {ic}, Correct: {correct_matches:.2f}/{total_values}")
    #print(f"Perfect: {perfect_matches}, Partial: {partial_matches}")
    #print(f"Accuracy: {accuracy:.1f}%, Precision: {precision:.1f}%, Recall: {recall:.1f}%, F1: {f1score:.1f}%")
    
    return {
        "LLM": model_name,
        "False Positives": fp,
        "False Negatives": fn,
        "Incorrect Extractions": ic,
        "Correct Matches": correct_matches,
        "Precision": precision,
        "Recall": recall,
        "F1score": f1score,
        "Accuracy": accuracy,
        "Parsed": None,
        "Hospital": hospital,
        # GLiNER does not use prompts in the same way, so we set it to
        "Prompt": "None",
        "Distressed": True if "distressed" in json_file.lower() else False

    }

def is_partial_match(template_val, gliner_val):
    """
    Check if there's a partial match between template and GLiNER values.
    Returns True if any single word from template is found in GLiNER output.
    """
    if not template_val or not gliner_val:
        return False
    
    # Split template into individual words and check if any exist in GLiNER output
    template_words = template_val.lower().split()
    gliner_text = gliner_val.lower()
    
    # If any template word is found in the GLiNER output, count as partial match
    for word in template_words:
        if len(word) > 2 and word in gliner_text:  # Only count words longer than 2 characters
            return True
    
    return False

def flatten_template_for_gliner2(template):
    """
    Recursively flatten the nested template structure
    into a flat dict of exactly one entry per leaf.
    """
    flattened = {}
    def recurse(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}_{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    recurse(v, new_key)
                else:
                    flattened[new_key] = "" if v is None else str(v)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                recurse(item, f"{prefix}_{i}" if prefix else str(i))
    recurse(template)
    return flattened

def determine_model_name(directory, json_data, filename=""):
    """
    Determine the appropriate model name based on directory, JSON content, and filename.
    Handles vision detection and different model types across various sources.
    
    Args:
        directory (str): The directory name (e.g., "OpenAIVisionOut", "localout")
        json_data (dict): The JSON data containing model information
        filename (str): The filename for additional context
    
    Returns:
        str: The formatted model name with vision indicator if applicable
    """
    model_name = json_data.get("model", "Unknown")
    
    # Model name normalization - handle colon separated names
    if ":" in model_name:
        t = model_name.split(":")
        model_name = t[0] + t[1]
    
    # Specific model mappings
    if "NuExtract-1.5-tiny" in model_name:
        model_name =  "NuExtract:0.5B"
    elif "NuExtract-2.0-2B" in model_name:
        model_name = "NuExtract:2B"
    elif "NuExtract-2.0-4B" in model_name:
        model_name = "NuExtract:4B"
    elif "qwen/qwen2.5-vl-72b-instruct" in model_name:
        model_name = "qwen2.5:72b"
    elif "meta-llama/llama-4-scout" in model_name:
        model_name = "llama-4:17B"
    elif "google/gemini-2.0-flash-exp" in model_name:
        model_name = "gemini-2.0"
    elif "mistral-small3.1" in model_name:
        model_name = "mistral(24b)"
    elif "granite3.2-vision" in model_name:
        model_name = "granite3.2"
    elif "llama3.2_1b" in model_name:
        model_name = "llama3.2:1b"
    elif "llama3.2_3b" in model_name:
        model_name = "llama3.2:3b"
    elif "numind/NuNerZero" in model_name: 
        model_name = "GLiNER"
    elif "llava-llama-3.2-vision" in model_name:
        model_name = "llava-llama3.2:8b"
    elif "mistral_latest" in model_name:
        model_name = "mistral:7b"
    elif "qwen2.5-vl-7b" in model_name:
        model_name = "qwen2.5:7b"
    # Add vision indicator if this is a vision-enabled directory
    elif "biomed_base_v1.0" in model_name:
        model_name = "biomed_GliNER:"
    if "Vision" in directory:
        model_name = model_name + "*ImageInput*"
    
    return model_name
 

def is_valid_gene_name(gene_name):
    """
    Check if a string represents a valid gene symbol.
    Valid gene symbols should follow HGNC nomenclature patterns.
    
    Args:
        gene_name (str): The gene name to validate
        
    Returns:
        bool: True if the gene name appears valid, False otherwise
    """
    if not gene_name or not isinstance(gene_name, str):
        return False
    
    gene_name = gene_name.strip()
    
    # Check for obvious invalid patterns
    invalid_patterns = [
        # Contains spaces (except for some rare cases)
        r'\s',
        # Contains lowercase letters mixed with uppercase improperly
        r'[a-z].*[A-Z]|[A-Z].*[a-z].*[A-Z]',
        # Contains punctuation other than hyphens
        r'[^\w\-]',
        # Contains words like "gene", "genes", "and", "or"
        r'\b(gene|genes|and|or|the|with|protein|transcript)\b',
        # Contains HGVS notation
        r'[cp]\.',
        # Contains RefSeq identifiers
        r'NM_|XM_|NR_',
        # Contains obvious OCR corruption (repeated characters, gibberish)
        r'(.)\1{3,}|[bcdfghjklmnpqrstvwxyz]{6,}',
        # Too long for a gene name (most are under 15 characters)
        r'^.{20,}$',
        # Contains obvious non-gene content
        r'(variant|mutation|clinical|significance|pathogenic|benign)',
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, gene_name, re.IGNORECASE):
            return False
    
    # Additional checks for completely invalid content
    if len(gene_name) < 2:  # Too short
        return False
    
    if gene_name.lower() in ['', 'null', 'none', 'unknown', 'na', 'n/a']:
        return False
    
    return True

def is_gene_field(field_name):
    """
    Determine if a field name represents a gene symbol, regardless of validity.
    This is more liberal than is_valid_gene_name and catches both valid and invalid gene attempts.
    
    Args:
        field_name (str): The field name to check
        
    Returns:
        bool: True if this looks like an attempt to extract a gene name
    """
    if not field_name or not isinstance(field_name, str):
        return False
    
    field_name = field_name.strip()
    
    # Known non-gene fields - these should NOT be consolidated
    non_gene_fields = {
        'gene_symbol', 'transcript_id', 'hgvsg', 'hgvsc', 'hgvsp', 'chromosome', 'exon', 
        'zygosity', 'interpretation', 'mafac', 'mafan', 'mafaf', 'type', 'mega_hgvs',
        'variant_id', 'testing_laboratory', 'ordering_clinic', 'sample_type', 
        'analysis_type', 'sequencing_scope', 'reference_genome', 'report_type',
        'testing_context', 'date_collected', 'date_received', 'date_verified',
        'num_variants', 'num_tested_genes', 'variants', 'tested_genes', 'report_id'
    }
    
    if field_name.lower() in non_gene_fields:
        return False
    
    # If it's 2-15 characters, mostly uppercase letters/numbers, it's likely a gene attempt
    if (2 <= len(field_name) <= 15 and 
        re.match(r'^[A-Z0-9][A-Z0-9\-]*[A-Z0-9]?$', field_name)):
        return True
    
    # Catch other gene-like patterns (even if malformed)
    gene_like_patterns = [
        r'^[A-Z]{2,}[0-9]*$',  # Like TP53, BRCA1, etc.
        r'^[A-Z]+[0-9]+[A-Z]*$',  # Mixed letters and numbers
        r'^[A-Z]+\-[A-Z0-9]+$',  # Hyphenated
    ]
    
    for pattern in gene_like_patterns:
        if re.match(pattern, field_name):
            return True
    
    return False

def analyze_fp_fn_by_field(differences, hospital, model_name, template_data, extracted_data):
    """
    Analyze false positives and false negatives by individual extraction fields.
    Creates a single DataFrame with FP/FN counts and tracks hallucinated/missing keys.
    Consolidates ALL gene-related fields under the general gene_symbol field.
    
    Args:
        differences (list): List of difference strings from comparison
        hospital (str): Hospital name
        model_name (str): Model name
        template_data (dict): Original template data to identify required keys
        extracted_data (dict): Extracted data to identify hallucinated keys
    
    Returns:
        pd.DataFrame: Single DataFrame with all field-level error data
    """
    
    # Initialize tracking dictionaries
    fp_counts = defaultdict(int)
    fn_counts = defaultdict(int)
    all_fields = set()
    
    # Process differences to extract field names and categorize errors
    for diff in differences:
        diff_lower = diff.lower()
        
        # Skip non-error differences
        if not ("false positive" in diff_lower or "false negative" in diff_lower):
            continue
            
        # Extract the path/field name from the difference string
        path_match = re.search(r'at ([\w\.\_\[\]\/\-]+):', diff)
        if path_match:
            path = path_match.group(1)
            # Get the leaf field name (last part after dots, before brackets)
            field_name = path.split('.')[-1].split('[')[0]
            
            # Consolidate ALL gene-like fields under gene_symbol
            if is_gene_field(field_name):
                field_name = 'gene_symbol'
            
            all_fields.add(field_name)
            
            # Count the error type
            if "false positive" in diff_lower:
                fp_counts[field_name] += 1
            elif "false negative" in diff_lower:
                fn_counts[field_name] += 1
    
    # Also process structural differences (missing/extra keys at dict level)
    def extract_all_leaf_keys(obj, prefix=""):
        """Extract all leaf keys from nested structure"""
        keys = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)) and v:  # Non-empty nested structures
                    keys.update(extract_all_leaf_keys(v, f"{prefix}.{k}" if prefix else k))
                else:  # Leaf nodes (including empty values)
                    leaf_key = f"{prefix}.{k}" if prefix else k
                    # Get just the field name
                    field_name = leaf_key.split('.')[-1].split('[')[0]
                    
                    # Consolidate gene-like keys under gene_symbol
                    if is_gene_field(field_name):
                        keys.add('gene_symbol')
                    else:
                        keys.add(field_name)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    keys.update(extract_all_leaf_keys(item, f"{prefix}[{i}]" if prefix else str(i)))
                else:
                    # Leaf item in list
                    field_name = f"{prefix}[{i}]" if prefix else str(i)
                    keys.add(field_name.split('.')[-1].split('[')[0])
        return keys
    
    # Get all leaf keys from both template and extracted data
    template_keys = extract_all_leaf_keys(template_data)
    extracted_keys = extract_all_leaf_keys(extracted_data)
    
    # Add any keys that might have been missed in the differences processing
    all_possible_keys = template_keys | extracted_keys
    for key in all_possible_keys:
        if is_gene_field(key):
            all_fields.add('gene_symbol')
        else:
            all_fields.add(key)
    
    # Create comprehensive field list
    all_error_fields = sorted(list(all_fields))
    
    # Calculate totals - these should match the main comparison totals
    total_fp = sum(fp_counts.values())
    total_fn = sum(fn_counts.values())
    
    # Create single row of data
    row_data = {
        'hospital': hospital,
        'model': model_name,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'template_keys_count': len(template_keys),
        'extracted_keys_count': len(extracted_keys)
    }
    
    # Add columns for each field's FP/FN counts
    for field in all_error_fields:
        row_data[f'fp_{field}'] = fp_counts.get(field, 0)
        row_data[f'fn_{field}'] = fn_counts.get(field, 0)
    
    return pd.DataFrame([row_data])


def main():
    import numpy as np
    import os
    os.chdir("/Users/ayu/PDF_benchmarking/getJSON")
    
    '''if os.path.exists("../graphs/Hospitaltest.csv"):
        ovr = pd.read_csv("../graphs/Hospitaltest.csv") 
    else:'''
    ovr = pd.DataFrame(columns = ["LLM","False Positives","False Negatives","Incorrect Extractions","Correct Matches","Precision","Recall","F1score","Accuracy","Parsed","Hospital", "Prompt", "Distressed"])

    # Initialize single DataFrame for field-level analysis
    '''if os.path.exists("../graphs/field_analysisfinal.csv"):
        field_analysis_df = pd.read_csv("../graphs/field_analysisfinal.csv") 
    else:'''
    field_analysis_df = pd.DataFrame()

    json_direcs = [
        "localout",
        "glinerOut", 
       "OllamaOut",
        "OllamaOutNP",
       "OllamaVisionOut",
        "OllamaVisionOutNP",
        "OpenAIOut", 
        "OpenAIOutNP",
        "OpenAIVisionOut",
        "OpenAIVisionOutNP"
    ]
    
    lab_to_hospital = {
        "children's hospital of eastern ontario (ottawa)": "CHEO",
        "hamilton health sciences (hamilton)": "Hamilton",
        "hospital for sick children (toronto)": "SickKids",
        "kingston health sciences centre (kingston)": "Kingston",
        "london health sciences centre (london)": "LHSC",
        "north york general hospital (toronto)": "NYGH",
        "sinai health system (toronto)": "MtSinai"
    }
    
    sources = {
        "localout": "huggingface",
        "glinerOut": "gliner",
        "OllamaOut": "ollama",
        "OllamaOutNP": "ollama",
        "OllamaVisionOut": "ollama_vision", 
        "OpenAIOut": "openai",
        "OpenAIVisionOut": "openai_vision",
        "OpenRouter": "openrouter",
        "OpenRouterVisionOut": "openrouter_vision"
    }
    
    for direc in json_direcs:
        d = direc
        if "NP" in direc:
            d = d.replace("NP","")
        source = sources[d]
        direc_path = "outJSON/" + direc
        
        if not os.path.exists(direc_path):
            continue
            
        all_files = [f for f in os.listdir(direc_path) if f.endswith('.json')]
        if not all_files:
            continue
        
        for json_file in all_files:
            with open("../makeTemplatePDF/out/mock_data.json", "r") as f:
                temp = json.load(f)
            
            template = template_to_string(temp)
            report_id = json_file.split("_")[0].replace(".txt","")
            hospital = template[report_id].get("testing_laboratory", "").lower()
            hospital = lab_to_hospital[hospital]
            copy = template_to_string(filter_template(template, hospital))
            copy = dict_to_lowercase(copy)
            
            file_path = os.path.join(direc_path, json_file)
            try:
                with open(file_path, "r") as f:
                    dtemp = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                continue
                
            temp_num = json_file.split("_")[0].replace(".txt","")
            template = template[temp_num] if isinstance(template, dict) and temp_num in template else template
            copy = template_to_string(filter_template(template, hospital))
            copy = dict_to_lowercase(copy)
            
            if direc == "glinerOut":
                model_name = determine_model_name(direc, dtemp, json_file)
                result = compare_gliner_output(copy, dtemp, hospital, source, model_name, json_file)
                ovr = pd.concat([ovr, pd.DataFrame([result])], ignore_index=True)
            else:
                model_name = determine_model_name(direc, dtemp, json_file)
                prompt = "LTNER/GPT-NER" if 'NP' in direc else "Normal"
                prompt = "None" if "localout" == direc else prompt
                
                if dtemp.get("status") != "success":
                    temp_row = {
                        "LLM": model_name,
                        "False Positives": np.nan,
                        "False Negatives": np.nan,
                        "Incorrect Extractions": np.nan,
                        "Correct Matches": np.nan,
                        "Precision": np.nan,
                        "Recall": np.nan,
                        "F1score": np.nan,
                        "Accuracy": np.nan,
                        "Parsed": False,
                        "Hospital": hospital,
                        "Prompt": prompt, 
                        "Distressed": True if "distressed" in json_file.lower() else False
                    }
                    ovr = pd.concat([ovr, pd.DataFrame([temp_row])], ignore_index=True)
                    continue
                
                try:
                    data = dict_to_lowercase(dtemp["data"])
                    if isinstance(data, dict):
                        for k,v in data.items():
                            data = v if isinstance(v, dict) else data
                            break
                except KeyError:
                    data = {}
                
                correct_matches, fp, fn, ic, total_values, differences = compare_values_with_template(copy, data)
                
                total_extracted = correct_matches + fp + ic
                total_expected = correct_matches + fn
                
                accuracy = (correct_matches / total_values * 100) if total_values > 0 else 0
                precision = (correct_matches / total_extracted * 100) if total_extracted > 0 else 0
                recall = (correct_matches / total_expected * 100) if total_expected > 0 else 0
                f1score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if differences:
                    field_analysis = analyze_fp_fn_by_field(differences, hospital, model_name, copy, data)
                    field_analysis_df = pd.concat([field_analysis_df, field_analysis], ignore_index=True)
                
                temp_row = {
                    "LLM": model_name.split("+")[0],
                    "False Positives": fp,
                    "False Negatives": fn,
                    "Incorrect Extractions": ic,
                    "Correct Matches": correct_matches,
                    "Precision": precision,
                    "Recall": recall,
                    "F1score": f1score,
                    "Accuracy": accuracy,
                    "Parsed": True,
                    "Hospital": hospital, 
                    "Prompt": prompt,
                    "Distressed": True if "distressed" in json_file.lower() else False
                }
                ovr = pd.concat([ovr, pd.DataFrame([temp_row])], ignore_index=True)

    # Save results
    ovr.to_csv("../graphs/Hospitalfinal.csv", index=False)
    
    if not field_analysis_df.empty:
        field_analysis_df.to_csv("../graphs/field_analysisfinal.csv", index=False)


if __name__ == "__main__":
    main()