"""Model directories SGLang can serve.

cotorra (github.com/bbj-lab/cotorra) saves a model trained with secondary
objectives (``tte`` / ``tnt`` / ``disposition`` heads) as ``CotorraForCausalLM``:
``model_type: cotorra``, the causal-LM backbone's config nested under
``text_config``, and the head weights under ``heads.*``. SGLang has no such
architecture. The backbone is laid out exactly as the stock ``*ForCausalLM``
(``model.*``, ``lm_head.*``), so dropping ``heads.*`` and unwrapping
``text_config`` gives a model SGLang and transformers load as is, with the same
next-token distribution. A cotorra model trained without heads is saved as the
stock class already and needs nothing.
"""

from __future__ import annotations

import json
import pathlib
import shutil

WRAPPED_MODEL_TYPES = ("cotorra",)

# keys the wrapper mirrors from its backbone, or holds for it
_CARRIED_KEYS = ("dtype", "torch_dtype", "bos_token_id", "eos_token_id", "pad_token_id", "tie_word_embeddings")


def _config(model_path: str | pathlib.Path) -> dict | None:
    path = pathlib.Path(model_path) / "config.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def is_wrapped(model_path: str | pathlib.Path) -> bool:
    """Whether ``model_path`` holds a cotorra model with secondary heads."""
    cfg = _config(model_path)
    return cfg is not None and cfg.get("model_type") in WRAPPED_MODEL_TYPES


def require_servable(model_path: str | pathlib.Path) -> None:
    """Raise if SGLang cannot load ``model_path`` because it is a cotorra wrapper."""
    if is_wrapped(model_path):
        raise ValueError(
            f"{model_path} is a cotorra model with secondary heads (model_type 'cotorra'), which SGLang "
            "cannot load. Export its backbone first:\n"
            "    from quick_sco_re import export_backbone\n"
            f"    export_backbone({str(model_path)!r}, '<new directory>')\n"
            "and point SGLang and transformers at the new directory.")


def export_backbone(model_path: str | pathlib.Path, out_dir: str | pathlib.Path, overwrite: bool = False) -> pathlib.Path:
    """Write the causal-LM backbone of a cotorra model with heads to ``out_dir``.

    The config is the wrapper's ``text_config`` (with its architecture, dtype and
    special-token ids filled in from the wrapper); the weights are every tensor
    except ``heads.*``, unchanged. ``generation_config.json`` is copied. Needs
    ``safetensors`` and ``transformers`` (the ``hf`` extra).

    Returns ``out_dir``.
    """
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    except ImportError as e:
        raise ImportError("export_backbone needs torch, transformers and safetensors: "
                          "pip install 'quick-sco-re[hf]'") from e

    src, dst = pathlib.Path(model_path), pathlib.Path(out_dir)
    cfg = _config(src)
    if cfg is None:
        raise FileNotFoundError(f"no config.json in {src}")
    if cfg.get("model_type") not in WRAPPED_MODEL_TYPES:
        raise ValueError(f"{src} is not a cotorra model with heads (model_type {cfg.get('model_type')!r}); "
                         "SGLang can load it as is")
    if dst.resolve() == src.resolve():
        raise ValueError("out_dir must differ from model_path")
    if dst.exists() and any(dst.iterdir()) and not overwrite:
        raise FileExistsError(f"{dst} is not empty (pass overwrite=True)")

    text = dict(cfg["text_config"])
    mt = text.get("model_type")
    arch = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(mt)
    if arch is None:
        raise ValueError(f"transformers has no causal-LM class for backbone model_type {mt!r}")
    text["architectures"] = [arch]
    for k in _CARRIED_KEYS:
        if text.get(k) is None and cfg.get(k) is not None:
            text[k] = cfg[k]
    if "transformers_version" in cfg:
        text["transformers_version"] = cfg["transformers_version"]

    index = src / "model.safetensors.index.json"
    if index.is_file():
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    elif (src / "model.safetensors").is_file():
        shards = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"no safetensors weights in {src}")

    dst.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    n_dropped = 0
    for shard in shards:
        tensors = {}
        with safe_open(str(src / shard), framework="pt") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                if k.startswith("heads."):
                    n_dropped += 1
                    continue
                tensors[k] = f.get_tensor(k)
        if not tensors:
            continue
        save_file(tensors, str(dst / shard), metadata={**meta, "format": "pt"})
        weight_map.update({k: shard for k in tensors})
    if not any(k.startswith("model.") for k in weight_map):
        raise ValueError(f"{src} has no backbone weights under 'model.'")
    if len(shards) > 1:
        with open(dst / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
    with open(dst / "config.json", "w") as f:
        json.dump(text, f, indent=2)
    if (src / "generation_config.json").is_file():
        shutil.copy2(src / "generation_config.json", dst / "generation_config.json")
    return dst
