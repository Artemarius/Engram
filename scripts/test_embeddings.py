"""Validate an exported ONNX embedding model against the PyTorch reference.

Loads both the HuggingFace PyTorch model and the ONNX export, runs a battery
of code-oriented test inputs through each, and compares the resulting
embeddings via cosine similarity.  Also performs a semantic sanity check to
verify that similar inputs produce closer embeddings than dissimilar ones.

Usage examples:
    # Validate all-MiniLM-L6-v2 (recommended)
    python scripts/test_embeddings.py --onnx-path models/all-MiniLM-L6-v2.onnx

    # Validate with explicit model name
    python scripts/test_embeddings.py \\
        --model-name sentence-transformers/all-MiniLM-L6-v2 \\
        --onnx-path models/all-MiniLM-L6-v2.onnx
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

_MISSING: list[str] = []


def _check_import(module: str, pip_name: str | None = None) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        _MISSING.append(pip_name or module)
        return False


_check_import("torch", "torch")
_check_import("transformers", "transformers")
_check_import("onnxruntime", "onnxruntime-gpu")
_check_import("numpy", "numpy")

if _MISSING:
    print(
        "ERROR: Missing required Python packages:\n"
        + "".join(f"  - {p}\n" for p in _MISSING)
        + "\nInstall them with:\n"
        f"  pip install {' '.join(_MISSING)}\n",
        file=sys.stderr,
    )
    sys.exit(1)

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COSINE_THRESHOLD = 0.99

MODEL_ALIASES: dict[str, str] = {
    "nomic": "nomic-ai/nomic-embed-code",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

# ---------------------------------------------------------------------------
# Test inputs — a diverse set of code and natural-language snippets
# ---------------------------------------------------------------------------

TEST_SNIPPETS: list[tuple[str, str]] = [
    (
        "C++ function",
        textwrap.dedent("""\
            void depthFusion(const cv::Mat& depth, const cv::Mat& intrinsics,
                             VoxelGrid& grid, const Eigen::Matrix4f& pose) {
                for (int v = 0; v < depth.rows; ++v) {
                    for (int u = 0; u < depth.cols; ++u) {
                        float d = depth.at<float>(v, u);
                        if (d <= 0.0f) continue;
                        Eigen::Vector3f pt = unproject(u, v, d, intrinsics);
                        pt = pose.block<3,3>(0,0) * pt + pose.block<3,1>(0,3);
                        grid.integrate(pt, truncationDist);
                    }
                }
            }"""),
    ),
    (
        "Python class",
        textwrap.dedent("""\
            class CodeChunker:
                \"\"\"Split source code into semantic chunks for embedding.\"\"\"

                def __init__(self, max_tokens: int = 512):
                    self.max_tokens = max_tokens

                def chunk(self, source: str, language: str) -> list[Chunk]:
                    if language in self._tree_sitter_languages:
                        return self._chunk_tree_sitter(source, language)
                    return self._chunk_regex(source)

                def _chunk_regex(self, source: str) -> list[Chunk]:
                    chunks = []
                    for block in re.split(r'\\n{2,}', source):
                        if block.strip():
                            chunks.append(Chunk(text=block.strip()))
                    return chunks"""),
    ),
    (
        "Natural language query",
        "how is depth fusion implemented",
    ),
    (
        "Comment block",
        textwrap.dedent("""\
            // ---------------------------------------------------------------------------
            // TSDF volume integration
            //
            // For each pixel in the depth image we unproject into 3D, transform by the
            // camera pose, and integrate into the voxel grid using a truncated signed
            // distance function.  The truncation distance controls the surface thickness.
            // ---------------------------------------------------------------------------"""),
    ),
]

# Pairs for the semantic sanity check.
# (description, snippet_a, snippet_b, should_be_similar)
SEMANTIC_PAIRS: list[tuple[str, str, str, bool]] = [
    (
        "C++ depth fusion vs NL query about depth fusion",
        TEST_SNIPPETS[0][1],  # C++ function
        TEST_SNIPPETS[2][1],  # NL query
        True,
    ),
    (
        "C++ depth fusion vs Python chunker class",
        TEST_SNIPPETS[0][1],  # C++ function
        TEST_SNIPPETS[1][1],  # Python class
        False,
    ),
    (
        "Comment about TSDF vs NL query about depth fusion",
        TEST_SNIPPETS[3][1],  # Comment block
        TEST_SNIPPETS[2][1],  # NL query
        True,
    ),
]

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pool hidden states using the attention mask."""
    mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    summed = np.sum(last_hidden_state * mask, axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


# ---------------------------------------------------------------------------
# Data class for results
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingComparison:
    label: str
    snippet_preview: str
    cosine_sim: float
    max_abs_diff: float
    passed: bool


@dataclass
class SemanticCheck:
    description: str
    similarity: float
    expected_similar: bool
    passed: bool


# ---------------------------------------------------------------------------
# Core test logic
# ---------------------------------------------------------------------------


def _get_pytorch_embedding(
    text: str,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
) -> np.ndarray:
    """Compute the mean-pooled embedding using the PyTorch model."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    hidden = outputs.last_hidden_state.numpy()
    mask = tokens["attention_mask"].numpy()
    return _mean_pool(hidden, mask)


def _get_onnx_embedding(
    text: str,
    tokenizer: AutoTokenizer,
    session: ort.InferenceSession,
) -> np.ndarray:
    """Compute embedding using the ONNX Runtime session.

    Handles two output conventions:
      - If the model outputs ``embeddings`` (mean-pooled), use directly.
      - If the model outputs ``last_hidden_state``, apply mean pooling.
    """
    tokens = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
    ort_inputs = {
        "input_ids": tokens["input_ids"].astype(np.int64),
        "attention_mask": tokens["attention_mask"].astype(np.int64),
    }

    output_names = [o.name for o in session.get_outputs()]

    if "embeddings" in output_names:
        result = session.run(["embeddings"], ort_inputs)[0]
    else:
        # Fallback: model outputs last_hidden_state, pool manually
        result = session.run([output_names[0]], ort_inputs)[0]
        if result.ndim == 3:
            result = _mean_pool(result, tokens["attention_mask"].astype(np.float32))

    return result


def run_parity_tests(
    tokenizer: AutoTokenizer,
    pt_model: torch.nn.Module,
    ort_session: ort.InferenceSession,
) -> list[EmbeddingComparison]:
    """Compare PyTorch and ONNX embeddings for all test snippets."""
    results: list[EmbeddingComparison] = []

    for label, text in TEST_SNIPPETS:
        pt_emb = _get_pytorch_embedding(text, tokenizer, pt_model)
        ort_emb = _get_onnx_embedding(text, tokenizer, ort_session)

        cos = _cosine_similarity(pt_emb, ort_emb)
        max_diff = float(np.max(np.abs(pt_emb - ort_emb)))
        passed = cos > COSINE_THRESHOLD

        # Truncate long snippets for display
        preview = text.replace("\n", " ")[:80]
        if len(text) > 80:
            preview += "..."

        results.append(
            EmbeddingComparison(
                label=label,
                snippet_preview=preview,
                cosine_sim=cos,
                max_abs_diff=max_diff,
                passed=passed,
            )
        )

    return results


def run_semantic_tests(
    tokenizer: AutoTokenizer,
    ort_session: ort.InferenceSession,
) -> list[SemanticCheck]:
    """Verify that semantically similar inputs produce closer embeddings."""
    results: list[SemanticCheck] = []

    for desc, text_a, text_b, should_be_similar in SEMANTIC_PAIRS:
        emb_a = _get_onnx_embedding(text_a, tokenizer, ort_session)
        emb_b = _get_onnx_embedding(text_b, tokenizer, ort_session)
        sim = _cosine_similarity(emb_a, emb_b)

        # For "should be similar" pairs, we expect sim > 0.5
        # For "should not be similar" pairs, we just note the value
        # The real check: similar pairs should score higher than dissimilar ones
        if should_be_similar:
            passed = sim > 0.3  # relaxed: cross-modal (code/NL) sims are lower
        else:
            passed = True  # dissimilar pairs always "pass" — we just log the score

        results.append(
            SemanticCheck(
                description=desc,
                similarity=sim,
                expected_similar=should_be_similar,
                passed=passed,
            )
        )

    # Cross-check: each "similar" pair should have higher sim than each "not similar" pair
    similar_sims = [r.similarity for r in results if r.expected_similar]
    dissimilar_sims = [r.similarity for r in results if not r.expected_similar]

    if similar_sims and dissimilar_sims:
        min_similar = min(similar_sims)
        max_dissimilar = max(dissimilar_sims)
        if min_similar <= max_dissimilar:
            # Mark all as failed with a note
            for r in results:
                if r.expected_similar:
                    r.passed = False

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_parity_results(results: list[EmbeddingComparison]) -> int:
    """Print parity test results. Returns number of failures."""
    print("=" * 76)
    print("  PARITY TESTS: PyTorch vs ONNX")
    print("=" * 76)

    failures = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if not r.passed:
            failures += 1
        print(f"\n  [{status}] {r.label}")
        print(f"    Snippet: {r.snippet_preview}")
        print(f"    Cosine similarity:  {r.cosine_sim:.6f}  (threshold: {COSINE_THRESHOLD})")
        print(f"    Max absolute diff:  {r.max_abs_diff:.6e}")

    return failures


def _print_semantic_results(results: list[SemanticCheck]) -> int:
    """Print semantic test results. Returns number of failures."""
    print("\n" + "=" * 76)
    print("  SEMANTIC SANITY CHECKS")
    print("=" * 76)

    failures = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if not r.passed:
            failures += 1
        relation = "similar" if r.expected_similar else "dissimilar"
        print(f"\n  [{status}] {r.description}")
        print(f"    Cosine similarity: {r.similarity:.4f}  (expected: {relation})")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ONNX embedding model against PyTorch reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/test_embeddings.py --onnx-path models/all-MiniLM-L6-v2.onnx\n"
            "  python scripts/test_embeddings.py \\\n"
            "      --model-name minilm \\\n"
            "      --onnx-path models/all-MiniLM-L6-v2.onnx\n"
        ),
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help=(
            "HuggingFace model identifier for the PyTorch reference. "
            "Accepts aliases: nomic, minilm.  Default: sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="Path to the exported ONNX model file.",
    )
    args = parser.parse_args()

    # Resolve aliases
    model_name = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    onnx_path: Path = args.onnx_path

    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}", file=sys.stderr)
        print(
            "Run the export script first:\n"
            "  python scripts/export_model.py --validate",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Load models ----------------------------------------------------
    print(f"Loading PyTorch model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    pt_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    pt_model.eval()

    print(f"Loading ONNX model:    {onnx_path}")
    available_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    ort_session = ort.InferenceSession(str(onnx_path), providers=providers)

    active_provider = ort_session.get_providers()[0]
    print(f"ONNX Runtime provider: {active_provider}")

    # Print model info
    output_info = ort_session.get_outputs()[0]
    print(f"Output name: {output_info.name}, shape: {output_info.shape}")
    print()

    # ---- Run tests ------------------------------------------------------
    parity_results = run_parity_tests(tokenizer, pt_model, ort_session)
    parity_failures = _print_parity_results(parity_results)

    semantic_results = run_semantic_tests(tokenizer, ort_session)
    semantic_failures = _print_semantic_results(semantic_results)

    # ---- Summary --------------------------------------------------------
    total_tests = len(parity_results) + len(semantic_results)
    total_failures = parity_failures + semantic_failures

    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    print(f"  Parity tests:   {len(parity_results) - parity_failures}/{len(parity_results)} passed")
    print(f"  Semantic tests: {len(semantic_results) - semantic_failures}/{len(semantic_results)} passed")
    print(f"  Total:          {total_tests - total_failures}/{total_tests} passed")

    if total_failures > 0:
        print(f"\n  OVERALL: FAIL ({total_failures} test(s) failed)")
        sys.exit(1)
    else:
        print("\n  OVERALL: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
