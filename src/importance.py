# Model-derived attention and token-importance analysis.
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence, TypedDict, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "hf-internal-testing/tiny-random-gpt2"


class _TokenizedInputs(dict[str, torch.Tensor]):
    def to(self, device: str) -> "_TokenizedInputs":
        # Move every tokenized tensor to the selected device.
        for key, value in list(self.items()):
            self[key] = value.to(device)
        return self


class _Tokenizer(Protocol):
    def __call__(self, text: str, *, return_tensors: str) -> _TokenizedInputs: ...


class _TokenizerFactory(Protocol):
    def from_pretrained(self, pretrained_model_name_or_path: str) -> _Tokenizer: ...


class _ModelConfig(Protocol):
    num_hidden_layers: int
    num_attention_heads: int


class _ModelOutput(Protocol):
    attentions: Sequence[torch.Tensor] | None


class _CausalLM(Protocol):
    config: _ModelConfig

    def to(self, device: str) -> "_CausalLM": ...

    def eval(self) -> "_CausalLM": ...

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> _ModelOutput: ...


class _ModelFactory(Protocol):
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        *,
        attn_implementation: str = "eager",
        **kwargs: object,
    ) -> _CausalLM: ...


class _PositionImportance(TypedDict):
    token_index: int
    attention_mass: float
    recency_signal: float


class _AttentionPayload(TypedDict):
    model_name: str
    prompt: str
    sequence_length: int
    layers: int
    heads: int
    total_attention_mass: float
    layer_summaries: list[float]
    position_importance: list[_PositionImportance]


@dataclass
class ImportanceResult:
    model_name: str
    prompt: str
    sequence_length: int
    layers: int
    heads: int
    total_attention_mass: float
    position_importance: list[_PositionImportance]

    def to_dict(self) -> dict[str, object]:
        # Convert the result dataclass into JSON-compatible data.
        return asdict(self)


def _load_model_and_tokenizer(model_name: str, device: str) -> tuple[_CausalLM, _Tokenizer]:
    # Load the model and tokenizer with materialized attention tensors.
    tokenizer = cast(_TokenizerFactory, AutoTokenizer).from_pretrained(model_name)
    model = cast(_ModelFactory, AutoModelForCausalLM).from_pretrained(
        model_name,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _attention_importance(
    model: _CausalLM, tokenizer: _Tokenizer, prompt: str, model_name: str
) -> _AttentionPayload:
    # Aggregate attention mass into a position-importance payload.
    inputs = tokenizer(prompt, return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_attentions=True,
            use_cache=False,
        )

    if outputs.attentions is None:
        raise ValueError("The model did not return attention tensors.")

    seq_len = int(inputs["input_ids"].shape[1])
    position_mass = [0.0 for _ in range(seq_len)]
    layer_summaries: list[float] = []

    for layer_attention in outputs.attentions:
        layer = layer_attention[0]
        mean_over_heads = layer.mean(dim=0)
        key_mass = mean_over_heads.mean(dim=0)
        layer_total = float(key_mass.sum().item())
        layer_summaries.append(layer_total)

        for token_index in range(seq_len):
            position_mass[token_index] += float(key_mass[token_index].item())

    total_mass = sum(position_mass)
    normalized: list[_PositionImportance] = [
        {
            "token_index": index,
            "attention_mass": (position_mass[index] / total_mass if total_mass > 0 else 0.0),
            "recency_signal": (
                ((index + 1) / max(seq_len, 1))
                * (position_mass[index] / total_mass if total_mass > 0 else 0.0)
            ),
        }
        for index in range(seq_len)
    ]

    return {
        "model_name": model_name,
        "prompt": prompt,
        "sequence_length": seq_len,
        "layers": len(outputs.attentions),
        "heads": int(model.config.num_attention_heads),
        "total_attention_mass": float(total_mass),
        "layer_summaries": layer_summaries,
        "position_importance": normalized,
    }


def run_importance_analysis(
    model_name: str = DEFAULT_MODEL,
    prompt: str = "The future of memory in transformers is",
) -> ImportanceResult:
    # Run importance analysis for one model and prompt.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_model_and_tokenizer(model_name, device)
    payload = _attention_importance(model, tokenizer, prompt, model_name)
    return ImportanceResult(
        model_name=payload["model_name"],
        prompt=payload["prompt"],
        sequence_length=payload["sequence_length"],
        layers=payload["layers"],
        heads=payload["heads"],
        total_attention_mass=payload["total_attention_mass"],
        position_importance=payload["position_importance"],
    )


def main() -> None:
    # Parse CLI arguments and write the importance result.
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute a simple attention-based token-importance signal for a prompt."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of memory in transformers is",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/importance"),
    )
    args = parser.parse_args()

    result = run_importance_analysis(model_name=args.model, prompt=args.prompt)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "importance_analysis.json"
    output_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result.to_dict(), indent=2))
    print(f"Saved importance analysis to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
