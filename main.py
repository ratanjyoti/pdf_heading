# Filename: main.py

import os
import json
import argparse
import numpy as np
import xgboost as xgb
import joblib
from utils.pdf_extractor import PdfExtractor
from utils.feature_extractor import FeatureExtractor

# --- Define Label Mapping ---
LABEL_MAPPING = {'NONE': 0, 'TITLE': 1, 'H1': 2, 'H2': 3, 'H3': 4, 'H4': 5}

def run_extraction(pdf_path, output_path):
    """Phase 1: Extracts enriched block data from a PDF and saves it as JSON."""
    print(f"--- Running Phase 1: Extraction ---")
    print(f"Processing PDF: {pdf_path}")
    
    extractor = PdfExtractor()
    enriched_blocks = extractor.extract_enriched_blocks(pdf_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Write with UTF-8 to be safe
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_blocks, f, indent=2)
        
    print(f"Successfully extracted {len(enriched_blocks)} blocks.")
    print(f"Raw data saved to: {output_path}")
    print("\nNext Step: Manually review and add the correct 'label' key to each block in the JSON file.")

def run_training(labeled_data_path, model_output_path):
    """Phase 2: Trains an XGBoost model on manually labeled data."""
    print(f"--- Running Phase 2: Training ---")
    print(f"Loading labeled data from: {labeled_data_path}")

    # 1. Load all labeled JSON files
    all_blocks = []
    for filename in os.listdir(labeled_data_path):
        if filename.endswith('.json'):
            file_path = os.path.join(labeled_data_path, filename)
            # FIX: Specify UTF-8 encoding to handle special characters from any OS
            with open(file_path, 'r', encoding='utf-8') as f:
                all_blocks.extend(json.load(f))
    
    if not all_blocks:
        print("Error: No labeled JSON files found in the directory.")
        return

    # 2. Extract features and labels
    feature_extractor = FeatureExtractor()
    feature_matrix, font_map = feature_extractor.extract_features(all_blocks)
    labels = np.array([LABEL_MAPPING[block['label']] for block in all_blocks])

    print(f"Training on {feature_matrix.shape[0]} samples with {feature_matrix.shape[1]} features.")

    # 3. Train the XGBoost model
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(LABEL_MAPPING),
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(feature_matrix, labels)
    
    # 4. Save the model and supporting files
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model_bundle = {
        'model': model,
        'font_map': font_map,
        'label_mapping': LABEL_MAPPING,
        'feature_names': feature_extractor.feature_names
    }
    joblib.dump(model_bundle, model_output_path)
    print(f"Training complete. Model bundle saved to: {model_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Outline Extraction Pipeline")
    parser.add_argument('phase', choices=['extract', 'train'], help="The pipeline phase to run.")
    parser.add_argument('--input', required=True, help="Input file (for extract) or directory (for train).")
    parser.add_argument('--output', required=True, help="Output file path.")
    
    args = parser.parse_args()

    if args.phase == 'extract':
        run_extraction(args.input, args.output)
    elif args.phase == 'train':
        run_training(args.input, args.output)