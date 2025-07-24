# Filename: predict.py

import os
import json
import argparse
import joblib
import numpy as np
from utils.pdf_extractor import PdfExtractor
from utils.feature_extractor import FeatureExtractor

def predict_outline(pdf_path, model_path, output_path=None):
    """
    Loads a trained model to predict the hierarchical outline of a new PDF document.
    This version includes a hard-coded rule to override model predictions for
    text blocks located within tables.
    """
    print(f"--- Running Prediction Phase ---")
    
    # --- 1. Load the trained model and its associated metadata ---
    print(f"Loading model bundle from: {model_path}...")
    try:
        # The .joblib file contains the trained model, the font map from training,
        # the label mapping, and the feature names list.
        model_bundle = joblib.load(model_path)
        model = model_bundle['model']
        font_map = model_bundle['font_map']
        inverse_label_map = {v: k for k, v in model_bundle['label_mapping'].items()}
        label_map = model_bundle['label_mapping']
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

    # --- 3. Make predictions for each block, applying business rules ---
    print("Classifying blocks with hierarchical context and business rules...")
    
    feature_extractor = FeatureExtractor()
    doc_font_sizes = [b.get('font_size', 0) for b in blocks if b.get('font_size', 0) > 6]
    median_font = np.median(doc_font_sizes) if doc_font_sizes else 12.0

    predictions = []
    # This context is passed to the feature extractor to help it create hierarchical features
    last_heading_info = {'index': -1, 'level': 0, 'font_size': median_font}

    for i, block in enumerate(blocks):
        # A. Get the feature vector for the current block
        current_features = feature_extractor._get_block_features(
            block, i, median_font, font_map, last_heading_info
        )
        
        # B. The model makes its best guess based on the features
        prediction_result = model.predict(np.array([current_features], dtype=np.float32))
        predicted_label_int = prediction_result[0]

        # C. *** CRITICAL BUSINESS RULE ***
        # Override the model's prediction if the block is flagged as being in a table.
        if block.get('is_in_table', False):
            # No matter what the model predicted, if it's in a table, it is not a heading.
            # We deterministically force the label to be 'NONE'.
            predicted_label_int = label_map['NONE']
            
        # D. Store the final, corrected prediction
        predictions.append(predicted_label_int)
        
        # E. Update the context for the *next* block's feature extraction
        predicted_label_str = inverse_label_map[predicted_label_int]
        if predicted_label_str.startswith('H'):
            try:
                level = int(predicted_label_str[1:])
                last_heading_info = {
                    'index': i,
                    'level': level,
                    'font_size': block.get('font_size', median_font)
                }
            except (ValueError, IndexError):
                # If the label is something weird like "H" or "Habc", ignore it
                pass

    # --- 4. Generate the final structured JSON output from the predictions ---
    print("Assembling the final outline...")
    outline, title_parts = [], []
    for i, block in enumerate(blocks):
        # Get the final label for the current block
        label = inverse_label_map[predictions[i]]
        
        if label == 'TITLE':
             title_parts.append(block['text'])
        
        # Only include headings in the final outline
        if label.startswith('H'):
            outline.append({
                'level': label,
                'text': block['text'],
                'page': block['page_number']
            })
            
    # Consolidate title and create the final JSON object
    doc_title = " ".join(title_parts) if title_parts else "Document Title Not Found"
    final_output = { "title": doc_title, "outline": outline }
    output_json_string = json.dumps(final_output, indent=4)

    # --- 5. Save or print the output ---
    if output_path:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Write with UTF-8 encoding to prevent errors with special characters
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_json_string)
        print(f"\n✅ Structured outline saved successfully to: {output_path}")
    else:
        # If no output path is provided, print to the console
        print("\n--- Generated JSON Outline ---")
        print(output_json_string)


if __name__ == "__main__":
    # Setup command-line argument parsing for easy use
    parser = argparse.ArgumentParser(description="Predict PDF outline using a trained heading classification model.")
    parser.add_argument('--pdf', required=True, help="Path to the new PDF file to process.")
    parser.add_argument('--model', required=True, help="Path to the saved model bundle (.joblib).")
    parser.add_argument('--output', help="Optional. Path to save the output as a JSON file. If not provided, prints to console.")
    
    args = parser.parse_args()
    
    predict_outline(args.pdf, args.model, args.output)