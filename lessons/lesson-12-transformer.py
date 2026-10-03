"""第十二节：小型 decoder-only Transformer 字符语言模型。

运行：uv run python lessons/lesson-12-transformer.py
语料内置，使用项目已有的 CPU PyTorch；不下载数据或模型。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


OUT = Path(__file__).with_name("lesson-12-output")
LINES = [
    "春天来了，花开了。",
    "夏天来了，蝉叫了。",
    "秋天来了，叶落了。",
    "冬天来了，雪下了。",
]
CONTEXT = 16
D_MODEL = 32
HEADS = 4
LAYERS = 2
FFN_WIDTH = 64
TRAIN_STEPS = 320
PREFIX = "春天来了，"


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("模型宽度必须能被 head 数整除")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.project = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.project(y)


class TransformerBlock(nn.Module):
    """Pre-LN：每个子层先归一化，再做残差相加。"""

    def __init__(self, width: int, heads: int, ffn_width: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width), nn.GELU(), nn.Linear(ffn_width, width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token = nn.Embedding(vocab_size, D_MODEL)
        self.position = nn.Embedding(CONTEXT, D_MODEL)
        self.blocks = nn.ModuleList(
            [TransformerBlock(D_MODEL, HEADS, FFN_WIDTH) for _ in range(LAYERS)]
        )
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, ids: torch.Tensor, trace: bool = False):
        batch, length = ids.shape
        if length > CONTEXT:
            raise ValueError(f"序列长度不能超过 {CONTEXT}")
        positions = torch.arange(length, device=ids.device)
        x = self.token(ids) + self.position(positions)
        shapes = {"ids": list(ids.shape), "embedded": list(x.shape)}
        for i, block in enumerate(self.blocks):
            x = block(x)
            shapes[f"block_{i + 1}"] = list(x.shape)
        logits = self.head(self.final_norm(x))
        shapes["logits"] = list(logits.shape)
        return (logits, shapes) if trace else logits


def make_windows(data: torch.Tensor, length: int):
    starts = torch.arange(len(data) - length)
    x = torch.stack([data[i:i + length] for i in starts])
    y = torch.stack([data[i + 1:i + length + 1] for i in starts])
    return x, y


def generation_trace(model: TinyTransformerLM, prefix: str, char_to_id: dict[str, int],
                     vocab: list[str], count: int):
    generated = prefix
    records = []
    with torch.inference_mode():
        for step in range(count):
            visible = generated[-CONTEXT:]
            ids = torch.tensor([[char_to_id[ch] for ch in visible]], dtype=torch.long)
            logits = model(ids)
            probs = logits[0, -1].softmax(-1)
            values, top_ids = probs.topk(min(4, len(vocab)))
            chosen_id = int(probs.argmax())
            chosen = vocab[chosen_id]
            records.append({
                "step": step + 1,
                "context": visible,
                "chosen": chosen,
                "chosen_probability": float(probs[chosen_id]),
                "top4": [{"char": vocab[int(i)], "probability": float(p)}
                         for p, i in zip(values, top_ids)],
            })
            generated += chosen
    return generated, records


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1212)
    OUT.mkdir(exist_ok=True)
    corpus = "\n".join(LINES) + "\n"
    vocab = sorted(set(corpus))
    lookup = {char: i for i, char in enumerate(vocab)}
    ids = torch.tensor([lookup[char] for char in corpus], dtype=torch.long)
    x, y = make_windows(ids, CONTEXT)
    assert x.shape == y.shape == (len(corpus) - CONTEXT, CONTEXT)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert "".join(vocab[i] for i in x[0].tolist()) == corpus[:CONTEXT]

    model = TinyTransformerLM(len(vocab))
    logits, shapes = model(x[:2], trace=True)
    assert shapes == {
        "ids": [2, CONTEXT], "embedded": [2, CONTEXT, D_MODEL],
        "block_1": [2, CONTEXT, D_MODEL], "block_2": [2, CONTEXT, D_MODEL],
        "logits": [2, CONTEXT, len(vocab)],
    }
    assert model.blocks[0].attention.head_width == D_MODEL // HEADS

    # LayerNorm(4) 默认逐位置沿最后四个特征求均值/方差。
    norm_input = torch.tensor([1., 2., 3., 4.])
    norm_layer = nn.LayerNorm(4)
    norm_result = norm_layer(norm_input)
    manual_norm = (norm_input - norm_input.mean()) / torch.sqrt(
        norm_input.var(correction=0) + norm_layer.eps
    )
    torch.testing.assert_close(norm_result, manual_norm)
    residual_branch = torch.tensor([.5, -.5, 1., -1.])
    residual_sum = norm_input + residual_branch
    torch.testing.assert_close(residual_sum, torch.tensor([1.5, 1.5, 4., 3.]))

    # 改动未来位置的 token 时，所有更早位置的 logits 必须不变。
    model.eval()
    probe = x[:1].clone()
    changed = probe.clone()
    changed[0, 12] = (changed[0, 12] + 1) % len(vocab)
    with torch.inference_mode():
        original = model(probe)
        altered = model(changed)
    torch.testing.assert_close(original[:, :12], altered[:, :12], rtol=0, atol=0)
    assert not torch.equal(original[:, 12], altered[:, 12])
    causal_max_diff = float((original[:, :12] - altered[:, :12]).abs().max())

    optimizer = torch.optim.AdamW(model.parameters(), lr=.004)
    history = []
    with torch.inference_mode():
        initial_loss = float(F.cross_entropy(model(x).reshape(-1, len(vocab)), y.reshape(-1)))
    history.append({"step": 0, "train_loss": initial_loss})
    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        selected = torch.randint(0, len(x), (32,))
        xb, yb = x[selected], y[selected]
        optimizer.zero_grad(set_to_none=True)
        z = model(xb)
        loss = F.cross_entropy(z.reshape(-1, len(vocab)), yb.reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            model.eval()
            with torch.inference_mode():
                train_loss = float(F.cross_entropy(
                    model(x).reshape(-1, len(vocab)), y.reshape(-1)))
            history.append({"step": step, "train_loss": train_loss})

    model.eval()
    generated, records = generation_trace(model, PREFIX, lookup, vocab, 12)
    with torch.inference_mode():
        final_logits = model(x)
        train_accuracy = float((final_logits.argmax(-1) == y).float().mean())

    checkpoint = OUT / "tiny-transformer.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = TinyTransformerLM(len(vocab))
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    reloaded.eval()
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(x[:2]), model(x[:2]))

    result = {
        "torch_version": torch.__version__, "device": "cpu",
        "lines": LINES, "corpus": corpus, "vocab": vocab,
        "configuration": {"context": CONTEXT, "d_model": D_MODEL, "heads": HEADS,
                          "head_width": D_MODEL // HEADS, "layers": LAYERS,
                          "ffn_width": FFN_WIDTH, "steps": TRAIN_STEPS,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "sampling": "greedy argmax"},
        "window_count": len(x), "shapes": shapes,
        "shift_example": {"input": list(corpus[:8]), "target": list(corpus[1:9])},
        "norm_example": {"input": norm_input.tolist(), "mean": float(norm_input.mean()),
                         "variance": float(norm_input.var(correction=0)),
                         "epsilon": norm_layer.eps, "normalized": norm_result.tolist(),
                         "branch": residual_branch.tolist(), "sum": residual_sum.tolist()},
        "causal_earlier_logits_max_diff": causal_max_diff,
        "history": history, "final_train_token_accuracy": train_accuracy,
        "generation": {"prefix": PREFIX, "text": generated, "steps": records},
    }
    (OUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("字符", len(vocab), "种；训练窗口", len(x), "个；参数", result["configuration"]["parameters"])
    print("形状", shapes)
    print("训练损失", f"{initial_loss:.3f}", "→", f"{history[-1]['train_loss']:.3f}")
    print("贪心生成", repr(generated))
    print("结果保存到", OUT)


if __name__ == "__main__":
    main()
