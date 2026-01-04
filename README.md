# 📄 PDF Heading Extraction & Evaluation System

A fast, offline, CPU-only system to extract structured headings (Title, H1, H2, H3) from PDF documents using layout-aware heuristics — with optional ground-truth evaluation and HTML visualization.

---

## 🚀 Overview

This project automatically analyzes PDF documents and reconstructs their logical structure by detecting headings based on **font size**, **styling**, **page position**, and **layout frequency**.

It is designed for environments where:
- ❌ Internet access is unavailable
- ❌ Large ML / DL models are not allowed
- ✅ Deterministic, explainable output is required
- ✅ Speed and low memory usage matter

The system also supports **accuracy evaluation** against manually annotated ground-truth JSON files and generates **HTML comparison reports**.

---

## ✨ Key Features

- 📐 Layout-aware heading detection (Title, H1, H2, H3)
- ⚡ Fast processing using multiprocessing (CPU only)
- 🧠 No machine learning, no pretrained models
- 🌐 Fully offline
- 📄 JSON output for downstream processing
- 📊 Precision / Recall / F1 evaluation
- 🌍 HTML reports for visual inspection
- 🧹 Auto-clears old output on every run

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **PyMuPDF (fitz)** — fast and lightweight PDF parsing
- Standard Python libraries only  
  (`multiprocessing`, `collections`, `difflib`, `json`, `os`)

---

pdf-heading-extractor/
│
├── input/ # Input PDF files
│ └── sample.pdf
│
├── ground_truth/ # Ground truth annotations
│ └── sample.json
│
├── output/
│ ├── sample.json # Extracted headings
│ ├── accuracy_report.json # Precision / Recall / F1
│ └── html_report/
│ └── sample.html # HTML comparison report
│
├── main.py # Main script
└── README.md

markdown
Copy code

⚠️ **Important:**  
Ground truth filenames **must exactly match** the PDF name:

sample.pdf → sample.json

yaml
Copy code

---

## 🧠 Heading Detection Logic

The system uses a **rule-based scoring strategy**:

### Signals Used
- Font size hierarchy
- Font frequency across document
- Bold / Italic emphasis
- Vertical position on page
- Repeating footer/header pattern detection

### Classification Heuristic
| Pattern | Assigned Level |
|------|---------------|
| Largest font + top of page | Title |
| Large font + bold | H1 |
| Medium font + bold | H2 |
| Smaller recurring structured text | H3 |

This makes the system **explainable, stable, and fast**.

---

## 📄 Ground Truth Format

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 1
    },
    {
      "level": "H2",
      "text": "Background",
      "page": 2
    }
  ]
}
```

📊 Evaluation Metrics
Predicted headings are matched with ground truth using:

Heading level

Page number

Text similarity (> 0.8)

Metrics Generated
Precision

Recall

F1-Score

Saved to:

output/accuracy_report.json

##🌐 HTML Reports
HTML reports are generated only when valid ground truth exists.

They display:

🟢 Predicted headings

🔵 Ground truth headings

Open directly:
output/html_report/sample.html

Or run as a local website:
bash
Copy code
cd output/html_report
python -m http.server 8080

Then open:
http://localhost:8080

▶️ How to Run
python main.py

The script will:

Clear previous output

Process all PDFs in input/

Extract structured headings

Save JSON output

Evaluate against ground truth (if available)

Generate HTML reports for matched files

⚙️ Performance Characteristics
Constraint	Status
CPU-only	✅
Offline	✅
Large ML models	❌ Not used
Execution time	≤ 10 sec (50 pages)
Memory usage	< 1 GB
Parallel processing	✅

📌 Use Cases
Document structure extraction

PDF preprocessing for search / indexing

Legal & government document analysis

Offline document intelligence pipelines

Evaluation of heading detection algorithms

📄 License
This project is licensed under the MIT License.

