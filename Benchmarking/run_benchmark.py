# 1. module imports
import argparse
import sys
from ocr_benchmark.evaluator import BenchmarkEvaluator, generate_markdown_summary_table
from ocr_benchmark.model_adapter import global_model_registry, get_available_models

# 2. constant definitions
DEFAULT_INPUT_DIR = "oinput"
DEFAULT_OUTPUT_DIR = "benchmark_results"
DEFAULT_MODELS = ["dummy"]

# 3. class definitions
# Entry point CLI runner

# 4. function definitions
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR Benchmark Suite for Old Assyrian Textbooks (HuggingFace Open-Weight Models)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing PDF documents and sidecar transcription TXT files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"List of models to benchmark. Available registered models: {get_available_models()}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save JSON and CSV benchmark reports.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device ('cuda' or 'cpu').",
    )
    return parser.parse_args()

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    
    args = parse_args()

    print(f"=== Old Assyrian OCR Benchmark Suite ===")
    print(f"Input directory : {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Selected models : {args.models}")
    print(f"Device          : {args.device}\n")

    evaluator = BenchmarkEvaluator(input_dir=args.input_dir, output_dir=args.output_dir)
    summaries = []

    for model_key in args.models:
        print(f"--> Running benchmark for model: '{model_key}'...")
        try:
            model = global_model_registry.get_model(model_key, device=args.device)
            summary = evaluator.evaluate_model(model, model_name=model_key)
            summaries.append(summary)
            print(f"    Done. Processed {summary.num_samples} sample(s). Avg CER: {summary.avg_cer:.4f}, WER: {summary.avg_wer:.4f}\n")
        except Exception as e:
            print(f"    [Error benchmarking model '{model_key}']: {e}\n", file=sys.stderr)

    if summaries:
        json_report_path = evaluator.save_reports(summaries)
        print("=== Benchmark Summary Results ===")
        print(generate_markdown_summary_table(summaries))
        print(f"\nDetailed JSON report saved to: {json_report_path}")

# 5. class instantiation
if __name__ == "__main__":
    main()
