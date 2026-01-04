Document Relevance Processor with PDF Heading Extraction
📋 Overview
The Document Relevance Processor is an advanced NLP-powered tool designed to intelligently rank and summarize document sections based on their relevance to user queries. This enhanced version includes robust PDF heading extraction capabilities, preserving document structure while performing semantic relevance analysis.

✨ Key Features
🔍 Smart PDF Parsing: Extract headings, subheadings, and document hierarchy from PDF files

🎯 Relevance Scoring: Combine BM25, TF-IDF, and semantic embeddings for accurate ranking

👥 Persona-Based Filtering: Tailor results for specific user profiles (e.g., "Health_Conscious", "Chef")

📝 Intelligent Summarization: Generate concise summaries using T5 transformer models

🔄 Structured Output: JSON-formatted results with preserved document hierarchy

🐳 Containerized Deployment: Full Docker support for easy deployment

📁 Project Structure
document-relevance-processor/
├── data/
│   ├── labeled_training/          # Training datasets
│   ├── models/                    # Pretrained models
│   ├── predictions/               # Prediction outputs
│   └── sample_dataset/           # Sample data for testing
├── utils/                         # Utility functions
├── .dockerignore                 # Docker ignore rules
├── Dockerfile                    # Container configuration
├── README.md                     # This file
├── Readme_1a.md                 # Detailed technical documentation
├── main.py                       # Main application entry point
├── predict.py                    # Prediction script
├── process_all_pdfs.py          # Batch PDF processing
├── requirements.txt             # Python dependencies
└── upgrade_data.py              # Data upgrade utilities

🚀 Quick Start
Prerequisites
Docker (recommended) or Python 3.9+

4GB RAM minimum (8GB recommended for large documents)

Git for cloning the repository

Installation
Option 1: Docker (Recommended)
# Clone the repository
git clone <repository-url>
cd document-relevance-processor

# Build the Docker image
docker build -t doc-relevance-processor .

# Run the container
docker run -v $(pwd)/data:/app/data -it doc-relevance-processor

Option 2: Local Installation
# Clone and setup
git clone <repository-url>
cd document-relevance-processor

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"

📖 Usage Examples
Basic PDF Processing# Process a single PDF with heading extraction
python main.py --input "documents/menu.pdf" \
               --query "vegetarian options" \
               --persona "Health_Conscious"

Batch Processing
# Process multiple PDFs
python process_all_pdfs.py --input-dir "documents/" \
                           --output-dir "results/" \
                           --query "extract dessert recipes"

Advanced Options
# With custom configuration
python main.py \
  --input "document.pdf" \
  --query "gluten-free alternatives" \
  --persona "Dietary_Restricted" \
  --extract-headings true \
  --hierarchy-depth 3 \
  --min-relevance 0.5 \
  --output-format "structured" \
  --save-headings "document_structure.json"

🛠️ Configuration

Key Configuration Files
config.json - Main configuration

relevant_terms.json - Priority terms for scoring

irrelevant_terms.json - Terms to filter out

synonyms.py - Synonym generation module

PDF Extraction Settings
Configure in config.json:

{
  "pdf_processing": {
    "heading_detection": {
      "font_size_threshold": 1.2,
      "bold_weight": 0.8,
      "min_length": 2,
      "max_length": 100
    },
    "chunking": {
      "max_chunk_size": 1000,
      "overlap": 100,
      "preserve_headings": true
    }
  }
}

📊 Output Format
The tool generates structured JSON output:
{
  "metadata": {
    "document": "menu.pdf",
    "query": "vegetarian options",
    "persona": "Health_Conscious",
    "processing_time": "2.34s"
  },
  "headings_hierarchy": [
    {
      "heading": "Main Menu",
      "level": 1,
      "relevance_score": 0.95,
      "sections": [...]
    }
  ],
  "ranked_sections": [
   {
      "section_id": "sec_001",
      "content": "Vegetarian pasta with fresh vegetables...",
      "summary": "Fresh vegetable pasta dish",
      "relevance_score": 0.92,
      "source_heading": "Vegetarian Options",
      "source_page": 5
    }
  ],
  "processing_stats": {
    "total_sections": 45,
    "relevant_sections": 12,
    "average_relevance": 0.78
  }
}

📈 Performance Tips
For large documents: Enable batch processing in config

For better accuracy: Adjust relevance thresholds based on your use case

For speed: Use smaller models or enable caching

For memory: Process documents in smaller batches

🤝 Contributing
Fork the repository

Create a feature branch: git checkout -b feature-name

Commit changes: git commit -m 'Add feature'

Push to branch: git push origin feature-name

Open a Pull Request

Development Setup
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Format code
black .

.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Transformers by Hugging Face for NLP models

PyMuPDF for PDF processing

NLTK for natural language tools

scikit-learn for machine learning utilities










                           
