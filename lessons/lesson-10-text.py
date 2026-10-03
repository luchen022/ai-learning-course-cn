"""第十节：Token → ID → Embedding → RNN 的离线 CPU 演示。

运行：uv run python lessons/lesson-10-text.py
只使用项目已有的 CPU PyTorch；玩具语料内置，不下载文本或模型。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


OUT = Path(__file__).with_name("lesson-10-output")
SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
SENTENCES = [
    "小猫 喜欢 小鱼",
    "小猫 追 小球",
    "小狗 喜欢 骨头",
    "小狗 追 小猫",
    "小鸟 喜欢 天空",
    "小鸟 追 小虫",
    "小鸟 起飞",
]


def vocabulary(sentences: list[str]) -> list[str]:
    result = SPECIAL.copy()
    for sentence in sentences:
        for token in sentence.split():
            if token not in result:
                result.append(token)
    return result


def char_vocabulary(sentences: list[str]) -> list[str]:
    result = SPECIAL.copy()
    for sentence in sentences:
        for char in sentence.replace(" ", ""):
            if char not in result:
                result.append(char)
    return result


def encode(sentence: str, lookup: dict[str, int]) -> list[int]:
    return [lookup["<BOS>"]] + [lookup.get(t, lookup["<UNK>"]) for t in sentence.split()] + [lookup["<EOS>"]]


def make_batch(sentences: list[str], lookup: dict[str, int]):
    sequences = [encode(s, lookup) for s in sentences]
    max_steps = max(len(s) - 1 for s in sequences)
    x, y = [], []
    for sequence in sequences:
        length = max_steps - (len(sequence) - 1)
        x.append(sequence[:-1] + [lookup["<PAD>"]]*length)
        y.append(sequence[1:] + [lookup["<PAD>"]]*length)
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8, padding_idx=0)
        self.rnn = nn.RNN(8, 16, batch_first=True)
        self.head = nn.Linear(16, vocab_size)

    def forward(self, ids: torch.Tensor):
        vectors = self.embedding(ids)
        hidden_steps, final_hidden = self.rnn(vectors)
        logits = self.head(hidden_steps)
        return vectors, hidden_steps, final_hidden, logits


def token_accuracy(logits, targets, pad_id):
    predicted = logits.argmax(-1)
    valid = targets != pad_id
    return float(((predicted == targets) & valid).sum() / valid.sum())


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    vocab = vocabulary(SENTENCES)
    cvocab = char_vocabulary(SENTENCES)
    ids = {token: i for i, token in enumerate(vocab)}
    x, y = make_batch(SENTENCES, ids)
    assert tuple(x.shape) == tuple(y.shape) == (7, 4)
    assert x[-1].tolist() == [ids["<BOS>"], ids["小鸟"], ids["起飞"], ids["<PAD>"]]
    assert y[-1].tolist() == [ids["小鸟"], ids["起飞"], ids["<EOS>"], ids["<PAD>"]]
    assert encode("小猫 喜欢 火星", ids)[-2] == ids["<UNK>"]
    assert (y != ids["<PAD>"]).sum().item() == 27

    # 便于手算的固定查找表，仅用于展示“ID 取一行”；不是语义词向量。
    demo_table = torch.arange(len(vocab)*3, dtype=torch.float32).reshape(len(vocab), 3) / 10
    demo_table[ids["<PAD>"]] = 0
    demo_embedding = nn.Embedding(len(vocab), 3, padding_idx=ids["<PAD>"])
    with torch.no_grad():
        demo_embedding.weight.copy_(demo_table)
    demo_vector = demo_embedding(x[:2])
    assert tuple(demo_vector.shape) == (2, 4, 3)
    torch.testing.assert_close(demo_vector[0, 1], demo_table[ids["小猫"]])
    torch.testing.assert_close(demo_vector[1, 3], demo_table[ids["小球"]])
    assert demo_embedding(torch.tensor([ids["<PAD>"]])).sum().item() == 0

    torch.manual_seed(1010)
    model = TinyLanguageModel(len(vocab))
    embedded, hidden, final_hidden, logits = model(x)
    assert tuple(embedded.shape) == (7, 4, 8)
    assert tuple(hidden.shape) == (7, 4, 16)
    assert tuple(final_hidden.shape) == (1, 7, 16)
    assert tuple(logits.shape) == (7, 4, len(vocab))
    assert tuple(model.embedding.weight.shape) == (len(vocab), 8)
    assert tuple(model.head.weight.shape) == (len(vocab), 16)
    # <PAD> 位置不参与交叉熵；手算有效位置的均值核对 API。
    per_position = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
    valid = y != ids["<PAD>"]
    masked_manual = per_position[valid].mean()
    masked_api = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=ids["<PAD>"])
    torch.testing.assert_close(masked_manual, masked_api)

    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    history = []
    for epoch in range(0, 201):
        if epoch:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _, _, _, logits = model(x)
            loss = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=ids["<PAD>"])
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                _, _, _, z = model(x)
                loss = F.cross_entropy(z.transpose(1, 2), y, ignore_index=ids["<PAD>"])
                history.append({"epoch": epoch, "train_loss": float(loss),
                                "train_token_accuracy": token_accuracy(z, y, ids["<PAD>"])})

    model.eval()
    with torch.inference_mode():
        vectors, states, _, final_logits = model(x)
        probabilities = final_logits.softmax(-1)
    predictions = []
    for step in range(4):
        p = probabilities[0, step]
        top_values, top_ids = p.topk(3)
        predictions.append({"input_token": vocab[int(x[0, step])],
                            "target_token": vocab[int(y[0, step])],
                            "hidden_first4": [round(float(v), 4) for v in states[0, step, :4]],
                            "hidden_norm": round(float(states[0, step].norm()), 4),
                            "top3": [{"token": vocab[int(i)], "probability": float(v)}
                                     for v, i in zip(top_values, top_ids)]})
    checkpoint = OUT / "tiny-rnn.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = TinyLanguageModel(len(vocab))
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    reloaded.eval()
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(x)[-1], final_logits)

    result = {
        "torch_version": torch.__version__, "device": "cpu",
        "sentences": SENTENCES, "vocab": vocab, "char_vocab": cvocab,
        "unknown_sentence": "小猫 喜欢 火星",
        "demo_table": [[round(float(v), 1) for v in row] for row in demo_table],
        "demo_batch": {"sentences": [SENTENCES[0], SENTENCES[-1]],
                       "input_ids": [x[0].tolist(), x[-1].tolist()],
                       "target_ids": [y[0].tolist(), y[-1].tolist()]},
        "shapes": {"batch_ids": list(x.shape), "embedding_weight": list(model.embedding.weight.shape),
                   "vectors": list(vectors.shape), "rnn_outputs": list(states.shape),
                   "final_hidden": list(final_hidden.shape), "logits": list(final_logits.shape)},
        "valid_target_positions": int(valid.sum()), "all_target_positions": int(y.numel()),
        "initial_masked_loss": history[0]["train_loss"],
        "final_masked_loss": history[-1]["train_loss"],
        "final_training_token_accuracy": history[-1]["train_token_accuracy"],
        "history": history, "first_sentence_steps": predictions,
    }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("词表", len(vocab), "个 Token；输入", tuple(x.shape),
          "→ Embedding", tuple(vectors.shape), "→ RNN", tuple(states.shape),
          "→ logits", tuple(final_logits.shape))
    print("有效目标位置", int(valid.sum()), "/", int(y.numel()),
          "；训练损失", f"{history[0]['train_loss']:.3f}", "→", f"{history[-1]['train_loss']:.3f}")
    print("结果保存到", OUT)


if __name__ == "__main__":
    main()
