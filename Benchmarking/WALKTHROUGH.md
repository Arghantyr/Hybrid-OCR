# Old Assyrian Textbook OCR Benchmark Suite — User Walkthrough

A professional, modular benchmarking suite designed to evaluate open-weight HuggingFace OCR models on Old Assyrian textbook PDFs against ground-truth sidecar transcriptions.

---

## 1. Project Structure

```
OCR Benchmark/
├── RULES.md                      # Code quality and vertical structure guidelines
├── requirements.txt              # Dependency specifications
├── create_sample_pdf.py          # Script to generate sample PDF and sidecar TXT pairs
├── run_benchmark.py              # CLI entry point for benchmark execution
├── WALKTHROUGH.md                # Project documentation and usage guide
├── .venv/                        # Virtual environment directory
│   └── bin/activate.bat          # Environment activation script
├── oinput/                       # Input directory for paired PDFs and sidecar TXTs
│   ├── sample1.pdf               # Sample document PDF
│   └── sample1.txt               # Matching ground-truth transcription
├── ocr_benchmark/                # Core python package
│   ├── __init__.py               # Package initialization
│   ├── metrics.py                # Implementation of accuracy, domain, and performance metrics
│   ├── data_loader.py            # PDF renderer and sidecar transcription loader
│   ├── model_adapter.py          # HuggingFace model registry (GLM-OCR, TrOCR, Qwen2-VL, GOT-OCR, Donut)
│   └── evaluator.py              # Benchmark execution orchestrator and report generator
├── tests/                        # Automated unit tests
│   └── test_metrics.py           # Test suite for metric verification
└── benchmark_results/            # Generated evaluation reports (JSON, CSV)
```

---

## 2. Environment Setup

### Step 1: Create Virtual Environment
Create an isolated Python virtual environment named `.venv`:

```cmd
python -m venv .venv
```

### Step 2: Activate Virtual Environment
Activate the environment using the path mandated by `RULES.md`:

```cmd
call .venv/bin/activate.bat
```

### Step 3: Install Dependencies
Install all required packages (`torch`, `transformers`, `torchvision`, `pillow`, `pdf2image`, `jiwer`, `scikit-learn`, `rouge-score`, `scipy`, `psutil`, `sentencepiece`, `tiktoken`, `requests`, `verovio`, `accelerate`):

```cmd
pip install -r requirements.txt
```

---

## 3. Generating Sample Datasets

If you do not have dataset files ready, generate a sample PDF and paired sidecar `.txt` transcription in `oinput/`:

```cmd
python create_sample_pdf.py
```

This creates:
- `oinput/sample1.pdf`: Rendered document page.
- `oinput/sample1.txt`: Sidecar transcription text.

To evaluate custom datasets, place your `.pdf` and matching `.txt` files into `oinput/` using matching basenames (e.g. `doc01.pdf` and `doc01.txt`).

---

## 4. Running Benchmarks

### A. Dry-Run Verification (Fast)
Test the benchmark pipeline using the lightweight `dummy` model:

```cmd
python run_benchmark.py --input_dir oinput --models dummy --output_dir benchmark_results
```

### B. Full Model Suite Evaluation
Evaluate all supported HuggingFace open-weight models (`glm-ocr`, `trocr`, `qwen2-vl`, `got-ocr`, `donut`):

```cmd
python run_benchmark.py --input_dir oinput --models glm-ocr trocr qwen2-vl got-ocr donut --device cuda
```

### CLI Command Options

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--input_dir` | `str` | `oinput` | Path to directory containing `.pdf` and matching `.txt` sidecars. |
| `--models` | `list` | `dummy` | Models to evaluate (`dummy`, `glm-ocr`, `trocr`, `qwen2-vl`, `got-ocr`, `donut`). |
| `--output_dir` | `str` | `benchmark_results` | Directory where JSON and CSV reports are saved. |
| `--device` | `str` | `cuda` | Target compute device (`cuda` or `cpu`). |

---

## 5. Metrics Description

The benchmark suite computes metrics across five distinct categories:

### A. Sentence-Level Metrics
- **Ordered Sentence Accuracy (`ordered_sentence_exact` / `ordered_fuzzy_ratio`)**: Evaluates line-by-line accuracy at exact index $i$. Measures how accurately the model preserves line order and exact line text.
- **Unordered Sentence Accuracy (`unordered_sentence_exact` / `unordered_hungarian_ratio`)**: Uses multiset matching and the **Hungarian algorithm** to find optimal bipartite pairings between ground-truth and predicted lines. Evaluates text line recovery independent of reading-order shifts.
- **Word-per-Sentence Accuracy (`exact_word_accuracy_macro` / `micro`)**: Average ratio of 100% correctly transcribed words per sentence/line.

### B. Character-Level Metrics
- **Character Error Rate (CER)**: Levenshtein edit distance between ground truth and predicted text, normalized by ground truth length ($\frac{\text{Insertions} + \text{Deletions} + \text{Substitutions}}{\text{Total GT Characters}}$).
- **Case-Insensitive CER**: CER computed after converting text to lowercase, isolating character misidentifications from Sumerogram capitalization conventions.
- **Character F1 Score (`char_f1`)**: Harmonic mean of character-level precision and recall.

### C. Word-Level Metrics
- **Word Error Rate (WER)**: Word-level Levenshtein edit distance normalized by total word count.
- **Word F1 Score (`word_f1`)**: Harmonic mean of word token precision and recall.

### D. Block & Domain-Specific Metrics
- **Normalized Edit Distance (NED)**: Bounded similarity metric $1 - \frac{\text{Levenshtein}(GT, Pred)}{\max(|GT|, |Pred|)}$ (scale $0.0$ to $1.0$).
- **Diacritic Error Rate (DER)**: Character error rate calculated strictly on Old Assyrian diacritics (`š`, `ṣ`, `ṭ`, `ḫ`, `á`, `à`, `í`, `ù`).
- **Epigraphic Bracket F1 (`bracket_f1`)**: Precision and recall F1 score for bracket restorations (`[`, `]`, `(`, `)`).
- **ROUGE-L**: Longest Common Subsequence (LCS) ratio, measuring structural and layout sequence preservation.

### E. Efficiency & Performance Metrics
- **Inference Time (`inference_time_sec`)**: Total wall-clock execution time in seconds per document.
- **Latency per Page (`latency_per_page_sec`)**: Average processing time per PDF page.
- **Peak System RAM (`peak_ram_mb`)**: Maximum system RAM consumed (MB), measured via `psutil`.
- **Peak GPU VRAM (`peak_gpu_vram_mb`)**: Maximum VRAM allocated (MB), measured via `torch.cuda.max_memory_allocated()`.

---

## 6. Automated Testing

Verify metric logic and component execution by running the unit test suite:

```cmd
python -m unittest discover tests
```
