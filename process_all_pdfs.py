# Filename: process_all_pdfs.py

import os
import json
import argparse
import joblib
from utils.pdf_extractor import PdfExtractor
from utils.feature_extractor_lite import FeatureExtractorLite

def process_all_pdfs(pdf_dir, model_path, output_dir):
    """
    Automates the prediction process for an entire directory of PDF files.
    
    This script:
    1. Loads a single, speed-focused model once.
    2. Iterates through every PDF in the input directory.
    3. Generates a corresponding JSON output file for each PDF in the output directory.
    """
    print(f"--- Starting Batch Prediction Process ---")

    # --- 1. Load the single trained model and tools ONCE for efficiency ---
    print(f"Loading model bundle from: {model_path}...")
    try:
        model_bundle = joblib.load(model_path)
        model = model_bundle['model']
        inverse_label_map = {v: k for k, v in model_bundle['label_mapping'].items()}
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at '{model_path}'. Cannot proceed.")
        return
    except Exception as e:
        print(f"FATAL ERROR: Could not load model bundle. Error: {e}")
        return

    # --- 2. Prepare tools and output directory ---
    os.makedirs(output_dir, exist_ok=True)
    pdf_extractor = PdfExtractor()
    feature_extractor = FeatureExtractorLite()

    # --- 3. Loop through all PDFs in the input directory ---
    for pdf_filename in os.listdir(pdf_dir):
        if not pdf_filename.lower().endswith('.pdf'):
            continue
        
        print(f"\n-> Processing: {pdf_filename}")
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        
        # This try-except block ensures that if one PDF is corrupted or fails,
        # the entire batch process doesn't stop.
        try:
            # A. Extract richly-featured blocks
            blocks = pdf_extractor.extract_enriched_blocks(pdf_path)
            if not blocks:
                print("  -> No text blocks found. Skipping.")
                continue

            # B. Generate features using the fast "lite" extractor
            feature_matrix, _ = feature_extractor.extract_features(blocks)
            if feature_matrix.size == 0:
                print("  -> Could not generate features. Skipping.")
                continue
            
            # C. Predict all blocks at once
            predictions = model.predict(feature_matrix)

            # D. Assemble the final JSON output, applying business rules
            outline, title_parts = [], []
            for i, block in enumerate(blocks):
                label = inverse_label_map.get(predictions[i], 'NONE')
                
                # Apply business rules for highest accuracy
                if block.get('is_in_table', False) or block['text'].strip().isdigit():
                    label = 'NONE'

                if label == 'TITLE':
                    title_parts.append(block['text'])
                if label.startswith('H'):
                    outline.append({'level': label, 'text': block['text'], 'page': block['page_number']})
            
            doc_title = " ".join(title_parts) if title_parts else "Document Title Not Found"
            final_output = {"title": doc_title, "outline": outline}
            
            # E. Save the output JSON file
            output_filename = os.path.splitext(pdf_filename)[0] + '.json'
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=4, ensure_ascii=False)
            print(f"  -> Successfully saved prediction to: {output_path}")

        except Exception as e:
            print(f"  -> ERROR: Failed to process {pdf_filename}. Error: {e}")
            
    print("\n--- Batch Prediction Complete ---")

# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated PDF Heading Extraction Tool.")
    parser.add_argument('--pdf_dir', required=True, help="Directory containing the source PDF files to process.")
    parser.add_argument('--model', required=True, help="Path to the saved speed-focused model bundle (.joblib).")
    parser.add_argument('--output_dir', required=True, help="Directory to save the final JSON prediction files.")
    
    args = parser.parse_args()
    process_all_pdfs(args.pdf_dir, args.model, args.output_dir)