"""Export an embedding model to ONNX format for use with ONNX Runtime C++ API.

Supports two models:
  - nomic-ai/nomic-embed-code-v1  (768-dim, code-optimized)
  - sentence-transformers/all-MiniLM-L6-v2  (384-dim, lightweight fallback)

The script loads the model via HuggingFace Transformers, traces it through
torch.onnx.export with dynamic axes, and optionally applies INT8 dynamic
quantization or FP16 conversion.  The tokenizer is saved alongside the ONNX
file so the C++ side can load it directly (tokenizer.json).

Usage examples:
    # Export the default nomic model to models/
    python scripts/export_model.py

    # Export MiniLM with INT8 quantization and validate
    python scripts/export_model.py --model minilm --quantize --validate

    # Export to a custom directory with FP16
    python scripts/export_model.py --output /tmp/models --fp16
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks — fail early with actionable messages
# ---------------------------------------------------------------------------

_MISSING: list[str] = []


def _check_import(module: str, pip_name: str | None = None) -> bool:
    """Return True if *module* is importable, else record the pip package name."""
    try:
        __import__(module)
        return True
    except ImportError:
        _MISSING.append(pip_name or module)
        return False


_check_import("torch", "torch")
_check_import("transformers", "transformers")
_check_import("onnx", "onnx")
_check_import("onnxruntime", "onnxruntime-gpu")  # recommend GPU build

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
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "nomic": "nomic-ai/nomic-embed-code-v1",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

# Nomic model expects the "search_document: " or "search_query: " prefix
# when encoding.  For the ONNX export we use a plain dummy; the C++ side
# should prepend the prefix before tokenizing.

OPSET_VERSION = 17
DEFAULT_SEQ_LEN = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_model_name(alias_or_name: str) -> str:
    """Map a short alias to the full HuggingFace model identifier."""
    return MODEL_ALIASES.get(alias_or_name.lower(), alias_or_name)


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply mean pooling over token embeddings, respecting the attention mask."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


# ---------------------------------------------------------------------------
# Wrapper that bakes mean-pooling into the traced graph
# ---------------------------------------------------------------------------


class _PooledModel(torch.nn.Module):
    """Thin wrapper around a transformer encoder that returns mean-pooled embeddings.

    ONNX export traces this module, so the resulting graph maps
    (input_ids, attention_mask) -> embeddings  directly.
    """

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, S, D)
        return _mean_pool(last_hidden, attention_mask)


# ---------------------------------------------------------------------------
# Export pipeline
# ---------------------------------------------------------------------------


def export_onnx(
    model_name: str,
    output_dir: Path,
    *,
    quantize: bool = False,
    fp16: bool = False,
    validate: bool = False,
) -> Path:
    """Export a HuggingFace transformer to ONNX with optional post-processing.

    Parameters
    ----------
    model_name:
        Full HuggingFace model identifier (e.g. ``nomic-ai/nomic-embed-code-v1``).
    output_dir:
        Directory in which to write the ``.onnx`` file and ``tokenizer.json``.
    quantize:
        If True, apply ONNX Runtime dynamic INT8 quantization.
    fp16:
        If True, convert the model weights to float16.
    validate:
        If True, load the exported model and run a sanity check.

    Returns
    -------
    Path
        The path to the final ``.onnx`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a filesystem-friendly stem from the model name
    model_stem = model_name.rsplit("/", maxsplit=1)[-1]
    onnx_path = output_dir / f"{model_stem}.onnx"

    print(f"[1/6] Loading model: {model_name}")
    t0 = time.perf_counter()

    # trust_remote_code=True is needed for nomic models
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    encoder.eval()

    elapsed = time.perf_counter() - t0
    print(f"      Model loaded in {elapsed:.1f}s")

    # Detect embedding dimension from a probe forward pass
    probe_tokens = tokenizer("hello", return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        probe_out = encoder(**probe_tokens)
    embed_dim = probe_out.last_hidden_state.shape[-1]
    print(f"      Embedding dimension: {embed_dim}")

    # ---- Wrap with mean pooling -----------------------------------------
    pooled_model = _PooledModel(encoder)
    pooled_model.eval()

    # ---- Dummy input for tracing ----------------------------------------
    print(f"[2/6] Creating dummy input (seq_len={DEFAULT_SEQ_LEN})")
    dummy_text = "def hello_world(): print('hello')"
    dummy = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=DEFAULT_SEQ_LEN,
    )
    dummy_input_ids = dummy["input_ids"]
    dummy_attention_mask = dummy["attention_mask"]

    # ---- ONNX export ----------------------------------------------------
    print(f"[3/6] Exporting ONNX (opset {OPSET_VERSION}) -> {onnx_path}")
    t0 = time.perf_counter()

    torch.onnx.export(
        pooled_model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_path),
        opset_version=OPSET_VERSION,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size"},
        },
    )

    elapsed = time.perf_counter() - t0
    print(f"      Exported in {elapsed:.1f}s  ({_file_size_mb(onnx_path):.1f} MB)")

    # ---- Optional INT8 dynamic quantization -----------------------------
    if quantize:
        print("[4/6] Applying INT8 dynamic quantization")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError:
            print(
                "WARNING: onnxruntime.quantization not available. "
                "Install 'onnxruntime' (>=1.14) for quantization support.",
                file=sys.stderr,
            )
        else:
            quantized_path = output_dir / f"{model_stem}_int8.onnx"
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(quantized_path),
                weight_type=QuantType.QInt8,
            )
            print(f"      Quantized model: {quantized_path} ({_file_size_mb(quantized_path):.1f} MB)")
            onnx_path = quantized_path
    else:
        print("[4/6] Skipping INT8 quantization (use --quantize to enable)")

    # ---- Optional FP16 conversion ---------------------------------------
    if fp16:
        print("[5/6] Converting to FP16")
        try:
            from onnxconverter_common import float16
        except ImportError:
            print(
                "WARNING: onnxconverter-common not installed. "
                "Install it with:  pip install onnxconverter-common",
                file=sys.stderr,
            )
        else:
            # If we already quantized, apply fp16 to the original instead
            base_path = output_dir / f"{model_stem}.onnx"
            fp16_path = output_dir / f"{model_stem}_fp16.onnx"
            model_proto = onnx.load(str(base_path))
            model_fp16 = float16.convert_float_to_float16(model_proto, keep_io_types=True)
            onnx.save(model_fp16, str(fp16_path))
            print(f"      FP16 model: {fp16_path} ({_file_size_mb(fp16_path):.1f} MB)")
            if not quantize:
                onnx_path = fp16_path
    else:
        print("[5/6] Skipping FP16 conversion (use --fp16 to enable)")

    # ---- Save tokenizer for C++ side ------------------------------------
    print("[6/6] Saving tokenizer")
    tokenizer_out = output_dir / "tokenizer.json"
    # save_pretrained writes tokenizer.json plus config files
    tokenizer.save_pretrained(str(output_dir))
    if tokenizer_out.exists():
        print(f"      Tokenizer saved: {tokenizer_out}")
    else:
        # Some tokenizers only produce tokenizer_config.json; warn the user
        saved = list(output_dir.glob("tokenizer*"))
        print(f"      Tokenizer files: {[p.name for p in saved]}")

    # ---- Optional validation --------------------------------------------
    if validate:
        print("\n--- Validation ---")
        _validate_onnx(onnx_path, tokenizer, pooled_model, embed_dim)

    # ---- Summary --------------------------------------------------------
    print("\n=== Export Summary ===")
    print(f"  Model:       {model_name}")
    print(f"  Dimension:   {embed_dim}")
    print(f"  ONNX file:   {onnx_path.resolve()}")
    print(f"  Size:        {_file_size_mb(onnx_path):.1f} MB")
    print(f"  Output dir:  {output_dir.resolve()}")

    return onnx_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_onnx(
    onnx_path: Path,
    tokenizer: AutoTokenizer,
    pooled_model: torch.nn.Module,
    embed_dim: int,
) -> None:
    """Load the exported ONNX model and compare its output to PyTorch."""
    print(f"  Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    test_text = "int main() { return 0; }"
    tokens = tokenizer(test_text, return_tensors="np", padding=True, truncation=True)

    ort_inputs = {
        "input_ids": tokens["input_ids"].astype(np.int64),
        "attention_mask": tokens["attention_mask"].astype(np.int64),
    }
    ort_output = session.run(["embeddings"], ort_inputs)[0]

    # PyTorch reference
    pt_tokens = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        pt_output = pooled_model(pt_tokens["input_ids"], pt_tokens["attention_mask"]).numpy()

    # Compare
    cos_sim = float(
        np.dot(ort_output.flatten(), pt_output.flatten())
        / (np.linalg.norm(ort_output) * np.linalg.norm(pt_output) + 1e-9)
    )
    max_diff = float(np.max(np.abs(ort_output - pt_output)))

    print(f"  Test input: {test_text!r}")
    print(f"  Output shape: {ort_output.shape}  (expected: (1, {embed_dim}))")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute diff: {max_diff:.6e}")

    if cos_sim > 0.99:
        print("  Result: PASS")
    else:
        print("  Result: FAIL (cosine similarity below 0.99 threshold)", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an embedding model to ONNX for the Engram MCP server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/export_model.py\n"
            "  python scripts/export_model.py --model minilm --quantize --validate\n"
            "  python scripts/export_model.py --fp16 --output /tmp/models\n"
        ),
    )
    parser.add_argument(
        "--model",
        default="nomic",
        help=(
            "Model to export. Use a short alias (nomic, minilm) or a full "
            "HuggingFace model identifier.  Default: nomic"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Output directory for the ONNX file and tokenizer. Default: models/",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 dynamic quantization after export.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert weights to FP16 (requires onnxconverter-common).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run a quick inference sanity check after export.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    model_name = _resolve_model_name(args.model)
    print(f"Engram Model Export — {model_name}\n")

    export_onnx(
        model_name=model_name,
        output_dir=args.output,
        quantize=args.quantize,
        fp16=args.fp16,
        validate=args.validate,
    )


if __name__ == "__main__":
    main()
