"""Model directories: refusing and unwrapping cotorra models with secondary heads.

The wrapped model is written here in cotorra 26.11's on-disk layout (model_type 'cotorra', the
backbone config under text_config with architectures and dtype unset, head weights under heads.*),
so the tests need transformers but not cotorra.
"""

import json
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from safetensors.torch import load_file, save_file  # noqa: E402

import quick_sco_re as q  # noqa: E402


def stock_model(tie):
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=128, bos_token_id=1, eos_token_id=2,
        tie_word_embeddings=tie)).eval()


def wrap_like_cotorra(model, path, shards=1):
    model.save_pretrained(path)
    cfg = json.load(open(path / "config.json"))
    text = {k: v for k, v in cfg.items() if k not in ("architectures", "dtype", "transformers_version")}
    text["architectures"] = None
    wrapped = {"architectures": ["CotorraForCausalLM"], "model_type": "cotorra", "dtype": cfg.get("dtype", "float32"),
               "heads": {"tte": {"weight": 1.0}}, "text_config": text, "vocab_size": cfg["vocab_size"],
               "hidden_size": cfg["hidden_size"], "bos_token_id": 1, "eos_token_id": 2,
               "tie_word_embeddings": cfg["tie_word_embeddings"], "transformers_version": cfg.get("transformers_version")}
    json.dump(wrapped, open(path / "config.json", "w"))
    sd = load_file(path / "model.safetensors")
    sd["heads.tte.linear.weight"] = torch.randn(1, 32)
    sd["heads.tte.linear.bias"] = torch.zeros(1)
    (path / "model.safetensors").unlink()
    keys = sorted(sd)
    if shards == 1:
        save_file(sd, path / "model.safetensors", metadata={"format": "pt"})
        return
    wm = {}
    for i in range(shards):
        name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        part = {k: sd[k] for k in keys[i::shards]}
        save_file(part, path / name, metadata={"format": "pt"})
        wm.update({k: name for k in part})
    json.dump({"metadata": {}, "weight_map": wm}, open(path / "model.safetensors.index.json", "w"))


@pytest.mark.parametrize("tie,shards", [(False, 1), (True, 1), (False, 3)])
def test_export_backbone_reproduces_the_backbone(tmp_path, tie, shards):
    m = stock_model(tie)
    wrap_like_cotorra(m, tmp_path / "wrapped", shards)
    assert q.is_wrapped(tmp_path / "wrapped")
    with pytest.raises(ValueError, match="export_backbone"):
        q.require_servable(tmp_path / "wrapped")
    with pytest.raises(ValueError, match="export_backbone"):
        q.check_rope_config(str(tmp_path / "wrapped"))

    out = q.export_backbone(tmp_path / "wrapped", tmp_path / "backbone")
    cfg = json.load(open(out / "config.json"))
    assert cfg["architectures"] == ["LlamaForCausalLM"] and cfg["model_type"] == "llama"
    assert cfg["tie_word_embeddings"] is tie
    assert not any(k.startswith("heads.") for f in out.glob("*.safetensors") for k in load_file(f))
    q.require_servable(out)
    q.check_rope_config(str(out))
    b = transformers.AutoModelForCausalLM.from_pretrained(out).eval()
    x = torch.randint(0, 40, (2, 17))
    with torch.no_grad():
        assert torch.equal(m(input_ids=x).logits, b(input_ids=x).logits)


def test_export_refuses_stock_models_and_overwrites(tmp_path):
    stock_model(False).save_pretrained(tmp_path / "stock")
    assert not q.is_wrapped(tmp_path / "stock")
    with pytest.raises(ValueError, match="as is"):
        q.export_backbone(tmp_path / "stock", tmp_path / "out")
    wrap_like_cotorra(stock_model(False), tmp_path / "wrapped")
    q.export_backbone(tmp_path / "wrapped", tmp_path / "out")
    with pytest.raises(FileExistsError):
        q.export_backbone(tmp_path / "wrapped", tmp_path / "out")
    q.export_backbone(tmp_path / "wrapped", tmp_path / "out", overwrite=True)


def test_create_engine_without_sglang(tmp_path, monkeypatch):
    stock_model(False).save_pretrained(tmp_path / "stock")
    monkeypatch.setitem(sys.modules, "sglang", None)          # import sglang -> ImportError
    with pytest.raises(ImportError, match=r"quick-sco-re\[sglang\]"):
        q.create_engine(str(tmp_path / "stock"), max_len=100)
