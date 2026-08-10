# 1. module imports
import collections
import difflib
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Sequence, TypeVar, Set

# 2. constant definitions
T = TypeVar('T')
DEFAULT_SMOOTHING_EPS = 1e-9
BYTES_PER_MB = 1024 * 1024
ASSYRIAN_DIACRITICS: Set[str] = {
    'š', 'ṣ', 'ṭ', 'ḫ', 'á', 'à', 'í', 'ì', 'ú', 'ù', 'é', 'è',
    'Š', 'Ṣ', 'Ṭ', 'Ḫ', 'Á', 'À', 'Í', 'Ì', 'Ú', 'Ù', 'É', 'È'
}
EPIGRAPHIC_BRACKETS: Set[str] = {'[', ']', '(', ')', '⸢', '⸣', '<', '>', '{', '}'}

# 3. class definitions
@dataclass
class PerformanceMetrics:
    inference_time_sec: float
    latency_per_page_sec: float
    peak_ram_mb: float
    peak_gpu_vram_mb: float

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class SentenceMatchResult:
    ordered_exact_ratio: float
    ordered_fuzzy_ratio: float
    unordered_exact_ratio: float
    unordered_hungarian_ratio: float
    total_gt_sentences: int
    total_pred_sentences: int

@dataclass
class WordPerSentenceResult:
    exact_word_accuracy_macro: float
    exact_word_accuracy_micro: float
    total_gt_words: int
    correct_gt_words: int

@dataclass
class AssyriologicalDomainMetrics:
    case_sensitive_cer: float
    case_insensitive_cer: float
    diacritic_error_rate: float
    bracket_f1: float

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class DocumentMetrics:
    cer: float
    wer: float
    ned: float
    character_f1: float
    word_f1: float
    rouge_l: float
    sentence_matching: SentenceMatchResult
    word_per_sentence: WordPerSentenceResult
    domain_metrics: AssyriologicalDomainMetrics
    performance: PerformanceMetrics

    def to_dict(self) -> dict:
        return asdict(self)

# 4. function definitions
def get_current_memory_usage() -> Tuple[float, float]:
    ram_mb = 0.0
    vram_mb = 0.0
    try:
        import psutil
        process = psutil.Process()
        ram_mb = process.memory_info().rss / BYTES_PER_MB
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / BYTES_PER_MB
    except Exception:
        pass
    return ram_mb, vram_mb

def compute_levenshtein(seq1: Sequence[T], seq2: Sequence[T]) -> int:
    if not seq1:
        return len(seq2)
    if not seq2:
        return len(seq1)
    dp = list(range(len(seq2) + 1))
    for i, char1 in enumerate(seq1):
        new_dp = [i + 1] * (len(seq2) + 1)
        for j, char2 in enumerate(seq2):
            cost = 0 if char1 == char2 else 1
            new_dp[j + 1] = min(dp[j + 1] + 1, new_dp[j] + 1, dp[j] + cost)
        dp = new_dp
    return dp[-1]

def split_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]

def split_words(text: str) -> List[str]:
    return text.strip().split()

def compute_cer(gt_text: str, pred_text: str, case_sensitive: bool = True) -> float:
    if not case_sensitive:
        gt_text = gt_text.lower()
        pred_text = pred_text.lower()
    if not gt_text:
        return 0.0 if not pred_text else 1.0
    dist = compute_levenshtein(gt_text, pred_text)
    return dist / len(gt_text)

def compute_wer(gt_text: str, pred_text: str, case_sensitive: bool = True) -> float:
    if not case_sensitive:
        gt_text = gt_text.lower()
        pred_text = pred_text.lower()
    gt_words = split_words(gt_text)
    pred_words = split_words(pred_text)
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    dist = compute_levenshtein(gt_words, pred_words)
    return dist / len(gt_words)

def compute_ned(gt_text: str, pred_text: str) -> float:
    max_len = max(len(gt_text), len(pred_text))
    if max_len == 0:
        return 1.0
    dist = compute_levenshtein(gt_text, pred_text)
    return 1.0 - (dist / max_len)

def compute_f1_score(gt_tokens: List[str], pred_tokens: List[str]) -> float:
    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0
    gt_counts = collections.Counter(gt_tokens)
    pred_counts = collections.Counter(pred_tokens)
    overlap = sum((gt_counts & pred_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

def compute_character_f1(gt_text: str, pred_text: str) -> float:
    return compute_f1_score(list(gt_text), list(pred_text))

def compute_word_f1(gt_text: str, pred_text: str) -> float:
    return compute_f1_score(split_words(gt_text), split_words(pred_text))

def compute_diacritic_error_rate(gt_text: str, pred_text: str) -> float:
    gt_diacritics = [char for char in gt_text if char in ASSYRIAN_DIACRITICS]
    pred_diacritics = [char for char in pred_text if char in ASSYRIAN_DIACRITICS]
    if not gt_diacritics:
        return 0.0 if not pred_diacritics else 1.0
    dist = compute_levenshtein(gt_diacritics, pred_diacritics)
    return dist / len(gt_diacritics)

def compute_bracket_f1(gt_text: str, pred_text: str) -> float:
    gt_brackets = [char for char in gt_text if char in EPIGRAPHIC_BRACKETS]
    pred_brackets = [char for char in pred_text if char in EPIGRAPHIC_BRACKETS]
    return compute_f1_score(gt_brackets, pred_brackets)

def compute_rouge_l(gt_text: str, pred_text: str) -> float:
    gt_words = split_words(gt_text)
    pred_words = split_words(pred_text)
    if not gt_words or not pred_words:
        return 1.0 if not gt_words and not pred_words else 0.0
    matcher = difflib.SequenceMatcher(None, gt_words, pred_words)
    lcs_len = sum(block.size for block in matcher.get_matching_blocks())
    prec = lcs_len / len(pred_words)
    rec = lcs_len / len(gt_words)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)

def compute_sentence_matching(gt_lines: List[str], pred_lines: List[str]) -> SentenceMatchResult:
    if not gt_lines:
        return SentenceMatchResult(1.0, 1.0, 1.0, 1.0, 0, len(pred_lines))
    
    min_len = min(len(gt_lines), len(pred_lines))
    exact_ordered_matches = sum(1 for i in range(min_len) if gt_lines[i] == pred_lines[i])
    ordered_exact_ratio = exact_ordered_matches / len(gt_lines)
    
    fuzzy_sim_sum = sum(
        difflib.SequenceMatcher(None, gt_lines[i], pred_lines[i]).ratio()
        for i in range(min_len)
    )
    ordered_fuzzy_ratio = fuzzy_sim_sum / len(gt_lines)
    
    gt_counts = collections.Counter(gt_lines)
    pred_counts = collections.Counter(pred_lines)
    unordered_exact_matches = sum((gt_counts & pred_counts).values())
    unordered_exact_ratio = unordered_exact_matches / len(gt_lines)
    
    matched_pred = set()
    total_hungarian_sim = 0.0
    for gt_line in gt_lines:
        best_sim = 0.0
        best_idx = -1
        for j, pred_line in enumerate(pred_lines):
            if j in matched_pred:
                continue
            sim = difflib.SequenceMatcher(None, gt_line, pred_line).ratio()
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_idx != -1 and best_sim > 0.3:
            matched_pred.add(best_idx)
            total_hungarian_sim += best_sim
    unordered_hungarian_ratio = total_hungarian_sim / len(gt_lines)
    
    return SentenceMatchResult(
        ordered_exact_ratio=ordered_exact_ratio,
        ordered_fuzzy_ratio=ordered_fuzzy_ratio,
        unordered_exact_ratio=unordered_exact_ratio,
        unordered_hungarian_ratio=unordered_hungarian_ratio,
        total_gt_sentences=len(gt_lines),
        total_pred_sentences=len(pred_lines),
    )

def compute_word_per_sentence(gt_lines: List[str], pred_lines: List[str]) -> WordPerSentenceResult:
    if not gt_lines:
        return WordPerSentenceResult(1.0, 1.0, 0, 0)
    
    min_len = min(len(gt_lines), len(pred_lines))
    sentence_accuracies = []
    total_gt_words = 0
    total_correct_gt_words = 0
    
    for i in range(len(gt_lines)):
        gt_words = split_words(gt_lines[i])
        if not gt_words:
            continue
        total_gt_words += len(gt_words)
        pred_words = split_words(pred_lines[i]) if i < min_len else []
        
        gt_counter = collections.Counter(gt_words)
        pred_counter = collections.Counter(pred_words)
        correct_words = sum((gt_counter & pred_counter).values())
        total_correct_gt_words += correct_words
        
        line_acc = correct_words / len(gt_words)
        sentence_accuracies.append(line_acc)
    
    macro_acc = sum(sentence_accuracies) / len(sentence_accuracies) if sentence_accuracies else 1.0
    micro_acc = total_correct_gt_words / total_gt_words if total_gt_words > 0 else 1.0
    
    return WordPerSentenceResult(
        exact_word_accuracy_macro=macro_acc,
        exact_word_accuracy_micro=micro_acc,
        total_gt_words=total_gt_words,
        correct_gt_words=total_correct_gt_words,
    )

def evaluate_transcription(
    gt_text: str,
    pred_text: str,
    inference_time_sec: float = 0.0,
    num_pages: int = 1,
    ram_mb: float = 0.0,
    vram_mb: float = 0.0,
) -> DocumentMetrics:
    gt_lines = split_lines(gt_text)
    pred_lines = split_lines(pred_text)
    
    cer_case_sens = compute_cer(gt_text, pred_text, case_sensitive=True)
    cer_case_insens = compute_cer(gt_text, pred_text, case_sensitive=False)
    wer = compute_wer(gt_text, pred_text)
    ned = compute_ned(gt_text, pred_text)
    char_f1 = compute_character_f1(gt_text, pred_text)
    word_f1 = compute_word_f1(gt_text, pred_text)
    rouge_l = compute_rouge_l(gt_text, pred_text)
    sentence_matching = compute_sentence_matching(gt_lines, pred_lines)
    word_per_sentence = compute_word_per_sentence(gt_lines, pred_lines)
    
    der = compute_diacritic_error_rate(gt_text, pred_text)
    bracket_f1 = compute_bracket_f1(gt_text, pred_text)
    
    domain_metrics = AssyriologicalDomainMetrics(
        case_sensitive_cer=cer_case_sens,
        case_insensitive_cer=cer_case_insens,
        diacritic_error_rate=der,
        bracket_f1=bracket_f1,
    )
    
    latency_per_page = inference_time_sec / num_pages if num_pages > 0 else inference_time_sec
    perf = PerformanceMetrics(
        inference_time_sec=inference_time_sec,
        latency_per_page_sec=latency_per_page,
        peak_ram_mb=ram_mb,
        peak_gpu_vram_mb=vram_mb,
    )
    
    return DocumentMetrics(
        cer=cer_case_sens,
        wer=wer,
        ned=ned,
        character_f1=char_f1,
        word_f1=word_f1,
        rouge_l=rouge_l,
        sentence_matching=sentence_matching,
        word_per_sentence=word_per_sentence,
        domain_metrics=domain_metrics,
        performance=perf,
    )

# 5. class instantiation
# Metrics module ready
