#!/usr/bin/env bash
# Wrapper script to generate mock data and PDFs
# Run this from the main PDF_benchmarking directory
if [ -d "output_pdfs" ]; then
    echo "Removing output_pdfs/*"
    rm -rf output_pdfs/images/*
    rm -rf output_pdfs/text/*
fi
if [ -d "makeTemplatePDF/out" ]; then
    echo "Removing makeTemplatePDF/out/*"
    rm -rf makeTemplatePDF/out/*
fi

echo "PDF Benchmarking - Automatic Report Generator"
echo "=============================================="

# Check if we're in the right directory

# Change to scripts directory and run the pipeline
cd makeTemplatePDF
cd scripts

echo "Changing to scripts directory..."
echo "Running pipeline to generate sets of mock data and PDFs..."
echo ""

# Execute the main pipeline
./run.sh "$@"

# Return to original directory
cd ..

echo ""
echo "Pipeline complete! Check the scripts/out/ directory for generated PDFs."

cd ..
echo "=============================================="
echo "All tasks completed successfully!"
echo "You can now review the generated reports in the 'out' directory."
echo "=============================================="
echo "Generating text files for each PDF..."
# Generate text files for each PDF in the out directory
cd getJSON
python3 pdfToText.py
echo "Text files generated successfully!"
echo "=============================================="
cd ..

