"""第十一节：从可手算的 QKV 到 CPU 多头因果自注意力。

运行：uv run python lessons/lesson-11-attention.py
只依赖项目已有的 CPU PyTorch，不下载数据或权重。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


OUT = Path(__file__).with_name("lesson-11-output")


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
    """显式展示每一步；最后两轴分别为时间 T、head 维度 d。"""
    d = q.shape[-1]
    raw = q @ k.transpose(-2, -1)
    scaled = raw / math.sqrt(d)
    block = torch.triu(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool), diagonal=1)
    masked = scaled.masked_fill(block, float("-inf")) if causal else scaled
    weights = masked.softmax(dim=-1)
    output = weights @ v
    return {"raw": raw, "scaled": scaled, "mask": block, "masked": masked,
            "weights": weights, "output": output}


class TinyMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 4, heads: int = 2):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model 必须能被 heads 整除")
        self.d_model, self.heads, self.d_head = d_model, heads, d_model // heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True):
        batch, length, width = x.shape
        if width != self.d_model:
            raise ValueError("输入最后一轴应为 d_model")

        def split(layer):
            return layer(x).reshape(batch, length, self.heads, self.d_head).transpose(1, 2)

        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        calculated = attention(q, k, v, causal)
        joined = calculated["output"].transpose(1, 2).contiguous().reshape(batch, length, width)
        output = self.out_proj(joined)
        return output, calculated["weights"], {
            "input": list(x.shape), "qkv_after_split": list(q.shape),
            "scores": list(calculated["scaled"].shape),
            "weights": list(calculated["weights"].shape),
            "per_head_output": list(calculated["output"].shape),
            "joined": list(joined.shape), "final": list(output.shape),
        }


def sinusoidal_positions(length: int, width: int) -> torch.Tensor:
    if width % 2:
        raise ValueError("本示例要求偶数维")
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    even = torch.arange(0, width, 2, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.0) * even / width)
    result = torch.zeros(length, width)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies)
    return result


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    names = ["小猫", "喜欢", "小鱼"]
    q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    k = q.clone()
    v = torch.tensor([[1., 0.], [0., 2.], [3., 3.]])
    full = attention(q, k, v, causal=False)
    causal = attention(q, k, v, causal=True)
    assert tuple(full["raw"].shape) == (3, 3)
    assert full["raw"].tolist() == [[1., 0., 1.], [0., 1., 1.], [1., 1., 2.]]
    torch.testing.assert_close(causal["weights"][0], torch.tensor([1., 0., 0.]))
    assert bool(torch.all(causal["weights"].triu(1) == 0))
    torch.testing.assert_close(causal["weights"].sum(-1), torch.ones(3))
    torch.testing.assert_close(causal["output"][0], v[0])
    # PyTorch 高级 API 使用 is_causal=True；布尔 attn_mask 的 True 表示“允许”，
    # 与本文件的 block=True 表示“屏蔽”相反。
    expected = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None],
                                               is_causal=True)[0, 0]
    torch.testing.assert_close(causal["output"], expected)
    allowed_mask = ~causal["mask"]
    checked_mask = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None],
                                                  attn_mask=allowed_mask)[0, 0]
    torch.testing.assert_close(causal["output"], checked_mask)

    torch.manual_seed(1111)
    model = TinyMultiHeadAttention(d_model=4, heads=2)
    x = torch.randn(2, 4, 4)
    y, weights, shapes = model(x)
    assert shapes == {
        "input": [2, 4, 4], "qkv_after_split": [2, 2, 4, 2],
        "scores": [2, 2, 4, 4], "weights": [2, 2, 4, 4],
        "per_head_output": [2, 2, 4, 2], "joined": [2, 4, 4], "final": [2, 4, 4],
    }
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2, 4))
    assert bool(torch.all(weights.triu(1) == 0))
    assert sum(p.numel() for p in model.parameters()) == 64
    try:
        TinyMultiHeadAttention(d_model=5, heads=2)
    except ValueError:
        pass
    else:
        raise AssertionError("无效 head 划分应报错")

    altered = x.clone()
    altered[:, 3] += torch.tensor([10., -7., 4., 6.])
    altered_causal, _, _ = model(altered, causal=True)
    torch.testing.assert_close(y[:, :3], altered_causal[:, :3], rtol=0, atol=0)
    assert not torch.allclose(y[:, 3], altered_causal[:, 3])
    full_y, _, _ = model(x, causal=False)
    altered_full, _, _ = model(altered, causal=False)
    unmasked_earlier_change = float((full_y[:, :3] - altered_full[:, :3]).abs().max().detach())
    assert unmasked_earlier_change > 1e-5
    gradient_input = x.detach().clone().requires_grad_(True)
    grad_output, _, _ = model(gradient_input, causal=True)
    grad_output[0, 1].sum().backward()
    assert torch.count_nonzero(gradient_input.grad[0, 2:]) == 0
    assert torch.count_nonzero(gradient_input.grad[0, :2]) > 0

    # 没有显式位置且不加因果 mask 时，对内容位置的置换是等变的。
    perm = torch.tensor([2, 0, 3, 1])
    permuted, _, _ = model(x[:, perm], causal=False)
    torch.testing.assert_close(permuted, full_y[:, perm])
    pe = sinusoidal_positions(4, 4)
    with_positions, _, _ = model(x + pe, causal=False)
    changed_content_order, _, _ = model(x[:, perm] + pe, causal=False)
    position_sensitive_difference = float((changed_content_order - with_positions[:, perm]).abs().max().detach())
    assert position_sensitive_difference > 1e-5

    result = {
        "torch_version": torch.__version__, "device": "cpu", "tokens": names,
        "tiny": {"q": q.tolist(), "k": k.tolist(), "v": v.tolist(),
                 "raw": full["raw"].tolist(), "scaled": full["scaled"].tolist(),
                 "full_weights": full["weights"].tolist(), "full_output": full["output"].tolist(),
                 "blocked_future": causal["mask"].tolist(),
                 "causal_weights": causal["weights"].tolist(),
                 "causal_output": causal["output"].tolist()},
        "multihead": {"shapes": shapes, "parameters": 64,
                      "weights_first_batch": weights[0].detach().tolist(),
                      "first_output": y[0].detach().tolist(),
                      "unmasked_earlier_change_after_future_edit": unmasked_earlier_change,
                      "causal_earlier_change_after_future_edit": float((y[:, :3]-altered_causal[:, :3]).abs().max().detach()),
                      "future_gradient_nonzero_count": int(torch.count_nonzero(gradient_input.grad[0, 2:])),
                      "past_gradient_nonzero_count": int(torch.count_nonzero(gradient_input.grad[0, :2]))},
        "positions": {"sinusoidal_4x4": pe.tolist(),
                      "unmasked_no_position_permutation_max_diff": float((permuted-full_y[:, perm]).abs().max().detach()),
                      "with_position_permutation_max_diff": position_sensitive_difference},
    }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("手算 QKV 与 PyTorch SDPA 一致；多头形状、因果隔离、梯度与位置编码检查通过。")
    print("形状", shapes)
    print("结果保存到", OUT)


if __name__ == "__main__":
    main()
