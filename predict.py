# Filename: predict.py (FINAL SPEED-FOCUSED VERSION)

import os
import json
import argparse
import joblib
import numpy as np
from utils.pdf_extractor import PdfExtractor
# ## CRITICAL CHANGE ##: Import the new, fast "Lite" feature extractor
from utils.feature_extractor_lite import FeatureExtractorLite

def predict_outline(pdf_path, model_path, output_path=None):
    """
    Loads a trained model to predict the hierarchical outline of a new PDF document.
    This version is OPTIMIZED FOR SPEED and uses the FeatureExtractorLite.
    """
    print(f"--- Running Prediction Phase ---")
    
    # --- 1. Load the trained XGBoost model and its associated metadata ---
    print(f"Loading model bundle from: {model_path}...")
    try:
        model_bundle = joblib.load(model_path)
        model = model_bundle['model']
        inverse_label_map = {v: k for k, v in model_bundle['label_mapping'].items()}
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at '{model_path}'")
        return
    except Exception as e:
        print(f"FATAL ERROR: Could not load model bundle. Error: {e}")
        return

    # --- 2. Extract enriched text blocks from the target PDF ---
    print(f"Extracting and processing blocks from: {pdf_path}...")
    pdf_extractor = PdfExtractor()
    blocks = pdf_extractor.extract_enriched_blocks(pdf_path)

    if not blocks:
        print("Warning: No text blocks were found in the PDF. Cannot generate an outline.")
        return

    # --- 3. Generate Features for ALL blocks at once using the LITE extractor ---
    # This will be extremely fast as it does NOT load or run the sentence model.
    print("Generating features for all blocks (LITE version)...")
    # ## CRITICAL CHANGE ##: Use the lightweight FeatureExtractorLite class
    feature_extractor = FeatureExtractorLite()
    feature_matrix, _ = feature_extractor.extract_features(blocks)

    if feature_matrix.size == 0:
        print("Warning: Could not generate features. Cannot make predictions.")
        return
        
    # --- 4. Make predictions for ALL blocks at once ---
    print("Classifying all blocks...")
    predictions = model.predict(feature_matrix)

    # --- 5. Assemble the final structured JSON output ---
    print("Assembling the final outline...")
    outline, title_parts = [], []
    for i, block in enumerate(blocks):
        predicted_label_int = predictions[i]
        label = inverse_label_map.get(predicted_label_int, 'NONE')
        
        if label == 'TITLE':
             title_parts.append(block['text'])
        
        if label.startswith('H'):
            outline.append({
                'level': label,
                'text': block['text'],
                'page': block['page_number']
            })
            
    doc_title = " ".join(title_parts) if title_parts else "Document Title Not Found"
    final_output = { "title": doc_title, "outline": outline }
    # Use ensure_ascii=False to correctly handle multilingual characters
    output_json_string = json.dumps(final_output, indent=4, ensure_ascii=False)

    # --- 6. Save or print the output ---
    if output_path:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_json_string)
        print(f"\n✅ Structured outline saved successfully to: {output_path}")
    else:
        print("\n--- Generated JSON Outline ---")
        print(output_json_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict PDF outline using a trained heading classification model.")
    # Using --input for consistency with main.py
    parser.add_argument('--input', required=True, help="Path to the new PDF file to process.")
    parser.add_argument('--model', required=True, help="Path to the saved model bundle (.joblib).")
    parser.add_argument('--output', help="Optional. Path to save the output as a JSON file. If not provided, prints to console.")
    
    args = parser.parse_args()
    
    predict_outline(args.input, args.model, args.output)