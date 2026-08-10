# 1. module imports
import io
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image

# 2. constant definitions
SUPPORTED_PDF_EXTS = (".pdf",)
SUPPORTED_TXT_EXTS = (".txt",)

# 3. class definitions
@dataclass
class BenchmarkSample:
    sample_id: str
    pdf_path: str
    txt_path: str
    ground_truth_text: str
    images: List[Image.Image]

# 4. function definitions
def load_pdf_as_images(pdf_path: str) -> List[Image.Image]:
    images = []
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path)
    except Exception:
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            for page in reader.pages:
                for img_file in page.images:
                    images.append(Image.open(io.BytesIO(img_file.data)))
        except Exception:
            pass
    if not images:
        images = [Image.new("RGB", (800, 1000), color=(255, 255, 255))]
    return images

def load_ground_truth(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()

def discover_dataset(input_dir: str) -> List[Tuple[str, str, str]]:
    if not os.path.exists(input_dir):
        return []
    samples = []
    for entry in os.listdir(input_dir):
        base, ext = os.path.splitext(entry)
        if ext.lower() in SUPPORTED_PDF_EXTS:
            txt_path = os.path.join(input_dir, f"{base}.txt")
            if os.path.exists(txt_path):
                pdf_path = os.path.join(input_dir, entry)
                samples.append((base, pdf_path, txt_path))
    return samples

def load_benchmark_samples(input_dir: str) -> List[BenchmarkSample]:
    sample_pairs = discover_dataset(input_dir)
    loaded_samples = []
    for sample_id, pdf_path, txt_path in sample_pairs:
        gt_text = load_ground_truth(txt_path)
        images = load_pdf_as_images(pdf_path)
        loaded_samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                pdf_path=pdf_path,
                txt_path=txt_path,
                ground_truth_text=gt_text,
                images=images,
            )
        )
    return loaded_samples

# 5. class instantiation
# Helper loader functions ready
