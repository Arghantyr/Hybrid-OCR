# 1. module imports
import json
import csv
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
from ocr_benchmark.data_loader import BenchmarkSample, load_benchmark_samples
from ocr_benchmark.metrics import DocumentMetrics, evaluate_transcription, get_current_memory_usage
from ocr_benchmark.model_adapter import BaseOCRModel, global_model_registry

# 2. constant definitions
DEFAULT_OUTPUT_DIR = "benchmark_results"

# 3. class definitions
@dataclass
class SampleEvaluationResult:
    sample_id: str
    model_name: str
    ground_truth_text: str
    predicted_text: str
    metrics: DocumentMetrics

@dataclass
class ModelBenchmarkSummary:
    model_name: str
    num_samples: int
    avg_cer: float
    avg_case_insensitive_cer: float
    avg_wer: float
    avg_ned: float
    avg_char_f1: float
    avg_word_f1: float
    avg_diacritic_error_rate: float
    avg_bracket_f1: float
    avg_rouge_l: float
    avg_ordered_sentence_exact: float
    avg_ordered_sentence_fuzzy: float
    avg_unordered_sentence_exact: float
    avg_unordered_sentence_hungarian: float
    avg_word_accuracy_per_sentence_macro: float
    avg_word_accuracy_per_sentence_micro: float
    avg_inference_time_sec: float
    avg_latency_per_page_sec: float
    peak_ram_mb: float
    peak_gpu_vram_mb: float
    sample_results: List[SampleEvaluationResult]

class BenchmarkEvaluator:
    def __init__(self, input_dir: str, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def evaluate_model(self, model: BaseOCRModel, model_name: str) -> ModelBenchmarkSummary:
        samples = load_benchmark_samples(self.input_dir)
        if not samples:
            raise ValueError(f"No valid PDF and sidecar TXT pairs found in '{self.input_dir}'.")
        
        sample_results = []
        for sample in samples:
            start_time = time.perf_counter()
            pred_text = model.transcribe_images(sample.images)
            elapsed_sec = time.perf_counter() - start_time
            
            ram_mb, vram_mb = get_current_memory_usage()
            num_pages = len(sample.images)
            metrics = evaluate_transcription(
                gt_text=sample.ground_truth_text,
                pred_text=pred_text,
                inference_time_sec=elapsed_sec,
                num_pages=num_pages,
                ram_mb=ram_mb,
                vram_mb=vram_mb,
            )
            sample_results.append(
                SampleEvaluationResult(
                    sample_id=sample.sample_id,
                    model_name=model_name,
                    ground_truth_text=sample.ground_truth_text,
                    predicted_text=pred_text,
                    metrics=metrics,
                )
            )

        num_samples = len(sample_results)
        avg_cer = sum(r.metrics.cer for r in sample_results) / num_samples
        avg_case_insens_cer = sum(r.metrics.domain_metrics.case_insensitive_cer for r in sample_results) / num_samples
        avg_wer = sum(r.metrics.wer for r in sample_results) / num_samples
        avg_ned = sum(r.metrics.ned for r in sample_results) / num_samples
        avg_char_f1 = sum(r.metrics.character_f1 for r in sample_results) / num_samples
        avg_word_f1 = sum(r.metrics.word_f1 for r in sample_results) / num_samples
        avg_der = sum(r.metrics.domain_metrics.diacritic_error_rate for r in sample_results) / num_samples
        avg_bracket = sum(r.metrics.domain_metrics.bracket_f1 for r in sample_results) / num_samples
        avg_rouge_l = sum(r.metrics.rouge_l for r in sample_results) / num_samples
        
        avg_ord_exact = sum(r.metrics.sentence_matching.ordered_exact_ratio for r in sample_results) / num_samples
        avg_ord_fuzzy = sum(r.metrics.sentence_matching.ordered_fuzzy_ratio for r in sample_results) / num_samples
        avg_unord_exact = sum(r.metrics.sentence_matching.unordered_exact_ratio for r in sample_results) / num_samples
        avg_unord_hung = sum(r.metrics.sentence_matching.unordered_hungarian_ratio for r in sample_results) / num_samples
        
        avg_word_acc_macro = sum(r.metrics.word_per_sentence.exact_word_accuracy_macro for r in sample_results) / num_samples
        avg_word_acc_micro = sum(r.metrics.word_per_sentence.exact_word_accuracy_micro for r in sample_results) / num_samples

        avg_time = sum(r.metrics.performance.inference_time_sec for r in sample_results) / num_samples
        avg_latency = sum(r.metrics.performance.latency_per_page_sec for r in sample_results) / num_samples
        peak_ram = max(r.metrics.performance.peak_ram_mb for r in sample_results)
        peak_vram = max(r.metrics.performance.peak_gpu_vram_mb for r in sample_results)

        return ModelBenchmarkSummary(
            model_name=model_name,
            num_samples=num_samples,
            avg_cer=avg_cer,
            avg_case_insensitive_cer=avg_case_insens_cer,
            avg_wer=avg_wer,
            avg_ned=avg_ned,
            avg_char_f1=avg_char_f1,
            avg_word_f1=avg_word_f1,
            avg_diacritic_error_rate=avg_der,
            avg_bracket_f1=avg_bracket,
            avg_rouge_l=avg_rouge_l,
            avg_ordered_sentence_exact=avg_ord_exact,
            avg_ordered_sentence_fuzzy=avg_ord_fuzzy,
            avg_unordered_sentence_exact=avg_unord_exact,
            avg_unordered_sentence_hungarian=avg_unord_hung,
            avg_word_accuracy_per_sentence_macro=avg_word_acc_macro,
            avg_word_accuracy_per_sentence_micro=avg_word_acc_micro,
            avg_inference_time_sec=avg_time,
            avg_latency_per_page_sec=avg_latency,
            peak_ram_mb=peak_ram,
            peak_gpu_vram_mb=peak_vram,
            sample_results=sample_results,
        )

    def save_reports(self, summaries: List[ModelBenchmarkSummary]) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        json_path = os.path.join(self.output_dir, "benchmark_report.json")
        csv_path = os.path.join(self.output_dir, "benchmark_summary.csv")

        data = [asdict(s) for s in summaries]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        headers = [
            "model_name", "num_samples", "avg_cer", "case_insensitive_cer", "avg_wer", "avg_ned",
            "avg_char_f1", "avg_word_f1", "diacritic_error_rate", "bracket_f1", "avg_rouge_l",
            "ordered_sentence_exact", "ordered_sentence_fuzzy",
            "unordered_sentence_exact", "unordered_sentence_hungarian",
            "word_acc_per_sentence_macro", "word_acc_per_sentence_micro",
            "avg_inference_time_sec", "avg_latency_per_page_sec", "peak_ram_mb", "peak_gpu_vram_mb"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for s in summaries:
                writer.writerow([
                    s.model_name, s.num_samples, f"{s.avg_cer:.4f}", f"{s.avg_case_insensitive_cer:.4f}",
                    f"{s.avg_wer:.4f}", f"{s.avg_ned:.4f}", f"{s.avg_char_f1:.4f}", f"{s.avg_word_f1:.4f}",
                    f"{s.avg_diacritic_error_rate:.4f}", f"{s.avg_bracket_f1:.4f}", f"{s.avg_rouge_l:.4f}",
                    f"{s.avg_ordered_sentence_exact:.4f}", f"{s.avg_ordered_sentence_fuzzy:.4f}",
                    f"{s.avg_unordered_sentence_exact:.4f}", f"{s.avg_unordered_sentence_hungarian:.4f}",
                    f"{s.avg_word_accuracy_per_sentence_macro:.4f}", f"{s.avg_word_accuracy_per_sentence_micro:.4f}",
                    f"{s.avg_inference_time_sec:.3f}", f"{s.avg_latency_per_page_sec:.3f}",
                    f"{s.peak_ram_mb:.1f}", f"{s.peak_gpu_vram_mb:.1f}"
                ])
        return json_path

# 4. function definitions
def generate_markdown_summary_table(summaries: List[ModelBenchmarkSummary]) -> str:
    table = [
        "| Model Name | CER | Case-Insens CER | Diacritic Error (DER) | Bracket F1 | Sentence (Unordered) | Latency (s/page) | Peak RAM |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for s in summaries:
        table.append(
            f"| **{s.model_name}** | {s.avg_cer:.4f} | {s.avg_case_insensitive_cer:.4f} | "
            f"{s.avg_diacritic_error_rate:.4f} | {s.avg_bracket_f1:.4f} | "
            f"{s.avg_unordered_sentence_hungarian * 100:.1f}% | {s.avg_latency_per_page_sec:.3f}s | "
            f"{s.peak_ram_mb:.1f} MB |"
        )
    return "\n".join(table)

# 5. class instantiation
# Evaluator module ready
