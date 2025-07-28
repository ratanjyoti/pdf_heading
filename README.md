# PDF Heading and Title Extractor

This project provides a high-performance, multilingual tool for automatically extracting a hierarchical outline (Title, H1, H2, etc.) from PDF documents. It is optimized for both speed and accuracy, capable of processing large, multi-page documents in under 10 seconds.

The system uses a sophisticated pipeline combining advanced document analysis with machine learning. It is packaged with Docker, making it portable and easy to run in any environment.

## Key Features

*   **High Speed:** Utilizes a two-pass cascade model to process large documents well within a 10-second time limit.
*   **High Accuracy:** Employs a powerful Sentence Transformer model for deep semantic understanding and applies business rules to eliminate common errors (e.g., page numbers, table content).
*   **Multilingual:** Built to understand over 50 languages, including Japanese, French, and German, out of the box.
*   **Automated & Portable:** Fully automated batch processing using a single command. Dockerization ensures it runs consistently anywhere.
*   **Robust Data Extraction:** Reliably detects tables and font styles (bold/italic) in a language-agnostic way using PyMuPDF's built-in tools.

## Project Structure

Adobe1A/
├── Dockerfile # Instructions to build the portable application
├── main.py # Script for data preparation and model training
├── process_all_pdfs.py # Script for running batch predictions (used by Docker)
├── requirements.txt # Project dependencies
├── models/ # Stores trained model files (.joblib)
└── utils/
├── pdf_extractor.py
├── feature_extractor.py
└── feature_extractor_lite.py


## Setup

### Prerequisites
*   Python 3.9+
*   Docker Desktop (must be running)

### Installation
1.  Clone the repository.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The primary way to use this project is through Docker, which handles all dependencies and provides a clean, automated workflow.

### Step 1: Build the Docker Image

From the project's root directory (`Adobe1A/`), run the build command once. This will package the entire application into a portable image named `heading-extractor`.

```bash
docker build -t heading-extractor .

Step 2: Prepare Your Data

In the project root, create a folder for your input files (e.g., input_pdfs).
Create an empty folder for the results (e.g., output_jsons).
Place all the PDF documents you want to process into the input_pdfs folder.

Step 3: Run Prediction

Execute the following single-line command from your terminal. This command runs the container, connects your local folders to it, and processes all the PDFs inside input_pdfs.

Generated bash

docker run --rm -v "$(pwd)/input_pdfs":/app/input_pdfs -v "$(pwd)/output_jsons":/app/output_jsons heading-extractor --pdf_dir /app/input_pdfs --model models/heading_model.joblib --output_dir /app/output_jsons
Use code with caution.

Bash
(Note: For Windows CMD, replace $(pwd) with %cd%)
After the command completes, the output_jsons folder will be populated with the structured JSON outlines, one for each input PDF.

Training a New Model (Advanced)
To improve accuracy or add support for new document styles, you can re-train the models.
Prepare Labeled Data: Use main.py extract to generate raw data, then manually label it and place it in a training folder (e.g., data/labeled_training).
Train Both Models: Use the smart main.py script to train both the "lite" and "full" models.

Generated bash
# Train the fast model
python main.py train --input data/labeled_training/ --output models/heading_model_speed_focused.joblib --model_type lite

# Train the accurate model
python main.py train --input data/labeled_training/ --output models/heading_model_high_accuracy.joblib --model_type full