#!/bin/bash

# Script to run all LLM processing scripts
# This will run jsonOllama, openAItoJSON, openRouterLLMs, and localLLM scripts

echo "Starting LLM processing pipeline..."
echo "======================================="

# Change to the getJSON directory
cd getJSON/run_models|| { echo "Directory getJSON not found! Exiting. Must start in PDF_benchmarking"; exit 1; }

# Run jsonOllama.py
echo "1. Running Ollama models..."
echo "----------------------------"
python3 jsonOllama.py
echo ""

# Run openAItoJSON.py
echo "2. Running OpenAI models..."
echo "---------------------------"
python3 openAItoJSON.py
echo ""


echo "5. Running gliner"
echo "-------------------------------"
/Users/ayu/.pyenv/versions/PDF_benchmarking_py312/bin/python /Users/ayu/PDF_benchmarking/getJSON/run_models/glinerJSON.py
echo "Gliner processing completed"
echo ""

echo "4. Running local HuggingFace models..."
echo "______________________________________"
python3 localLLM.py
echo "Local models completed"
echo ""


echo "======================================="
echo "All LLM processing scripts completed!"
echo "Check the respective output directories:"
echo "- outJSON/OllamaOut/ for Ollama results"
echo "- outJSON/OpenAIOut/ for OpenAI results" 
echo "- outJSON/OpenRouter/ for OpenRouter results"
echo "- outJSON/localout/ for local HuggingFace results"
echo "- outJSON/gliner/ for Gliner results"
echo ""

echo "5. Checking accuracy..."
echo "---------------------"
cd .. 
# Run accuracy check script
python3 compareJSON.py
echo "Accuracy check completed! Check the Hospital.csv file for results."