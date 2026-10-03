"""第十三节：真实小模型采样分布 + 可复核的本地 RAG 诊断。

运行：uv run python lessons/lesson-13-llm-use.py
复用第十二节 CPU 模型的权重；如缺少权重，先运行第十二节练习。
"""
from __future__ import annotations

import json
import math
import re
import runpy
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
OUT = HERE / "lesson-13-output"
PREVIOUS = HERE / "lesson-12-output"
DOCS = [
    {"id": "D1", "text": "海豚吃什么？本页讨论海豚的食性，但未给出具体食物。"},
    {"id": "D2", "text": "海豚吃的食物主要是小鱼和鱿鱼。"},
    {"id": "D3", "text": "蓝鲸是哺乳动物，不是鱼。"},
    {"id": "D4", "text": "海豚使用肺呼吸，需要浮出水面换气。"},
    {"id": "D5", "text": "企鹅生活在南半球，善于游泳。"},
]
QUESTIONS = [
    {"id": "food", "question": "海豚吃什么？", "gold_doc": "D2",
     "supported_answer": "海豚吃的食物主要是小鱼和鱿鱼。"},
    {"id": "whale", "question": "蓝鲸是鱼吗？", "gold_doc": "D3",
     "supported_answer": "蓝鲸是哺乳动物，不是鱼。"},
    {"id": "penguin", "question": "企鹅能活多少年？", "gold_doc": None,
     "supported_answer": "这些资料没有提供企鹅寿命，无法据此回答。"},
]


def chinese_bigrams(text: str) -> set[str]:
    chars = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    return {chars[i:i + 2] for i in range(len(chars) - 1)}


def retrieval_score(question: str, document: str) -> int:
    """教学用词面匹配：共同的连续两个汉字有多少种。"""
    return len(chinese_bigrams(question) & chinese_bigrams(document))


def rank_documents(question: str) -> list[dict]:
    ranked = [{"id": d["id"], "text": d["text"],
               "score": retrieval_score(question, d["text"]),
               "order": index} for index, d in enumerate(DOCS)]
    ranked.sort(key=lambda d: (-d["score"], d["order"]))
    return [{k: v for k, v in d.items() if k != "order"} for d in ranked]


def diagnose(gold_doc: str | None, selected: list[str], answer_supported: bool,
             included: list[str] | None = None) -> str:
    if gold_doc is None:
        return "资料缺失"
    if gold_doc not in selected:
        return "检索失败"
    if gold_doc not in (selected if included is None else included):
        return "上下文丢失"
    return "有证据但回答错误" if not answer_supported else "有证据且回答有据"


def pack_context(ranked: list[dict], limit: int, output_reserve: int,
                 base_chars: int = 24):
    """完整短文块顺序放入；字符槽位只用于模拟 token 预算。"""
    available = limit - output_reserve - base_chars
    if available < 0:
        return {"available_for_docs": available, "included": [],
                "omitted": [d["id"] for d in ranked]}
    included = []
    omitted = []
    for doc in ranked:
        cost = len(doc["text"])
        if cost <= available:
            included.append(doc["id"])
            available -= cost
        else:
            omitted.append(doc["id"])
    return {"available_for_docs": limit - output_reserve - base_chars,
            "included": included, "omitted": omitted}


def entropy(probabilities: torch.Tensor) -> float:
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * nonzero.log()).sum())


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    previous_json = PREVIOUS / "results.json"
    checkpoint = PREVIOUS / "tiny-transformer.pt"
    if not previous_json.exists() or not checkpoint.exists():
        raise FileNotFoundError("先运行 uv run python lessons/lesson-12-transformer.py")
    prior = json.loads(previous_json.read_text(encoding="utf-8"))
    module = runpy.run_path(str(HERE / "lesson-12-transformer.py"))
    model = module["TinyTransformerLM"](len(prior["vocab"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    vocab = prior["vocab"]
    lookup = {char: index for index, char in enumerate(vocab)}
    prefix = "春天来了，"
    ids = torch.tensor([[lookup[char] for char in prefix]], dtype=torch.long)
    with torch.inference_mode():
        logits = model(ids)[0, -1]
    temperatures = {}
    for temperature in (0.5, 1.0, 2.0):
        probabilities = torch.softmax(logits / temperature, dim=-1)
        temperatures[str(temperature)] = {
            "probabilities": probabilities.tolist(),
            "entropy_nats": entropy(probabilities),
            "top_char": vocab[int(probabilities.argmax())],
        }
    assert (temperatures["0.5"]["entropy_nats"]
            < temperatures["1.0"]["entropy_nats"]
            < temperatures["2.0"]["entropy_nats"])
    assert all(item["top_char"] == "花" for item in temperatures.values())
    top3 = torch.topk(logits, 3).indices
    top3_mask = torch.full_like(logits, -math.inf)
    top3_mask[top3] = logits[top3]
    top3_probs = torch.softmax(top3_mask, dim=-1)
    assert int((top3_probs > 0).sum()) == 3

    rankings = {q["id"]: rank_documents(q["question"]) for q in QUESTIONS}
    assert [d["id"] for d in rankings["food"][:2]] == ["D1", "D2"]
    assert rankings["whale"][0]["id"] == "D3"
    assert rankings["penguin"][0]["id"] == "D5"
    assert diagnose("D2", ["D1"], True) == "检索失败"
    assert diagnose("D2", ["D1", "D2"], True) == "有证据且回答有据"
    assert diagnose("D3", ["D3"], False) == "有证据但回答错误"
    assert diagnose(None, ["D5"], True) == "资料缺失"
    # 顺序取整块；短窗口放进误导性 D1，支持答案的 D2 进不去。
    short = pack_context(rankings["food"][:2], limit=70, output_reserve=20)
    long = pack_context(rankings["food"][:2], limit=100, output_reserve=20)
    assert short["included"] == ["D1"] and "D2" in short["omitted"]
    assert long["included"] == ["D1", "D2"]
    assert diagnose("D2", ["D1", "D2"], True, short["included"]) == "上下文丢失"

    result = {
        "torch_version": torch.__version__, "device": "cpu",
        "temperature_demo": {
            "model": "第十二节训练好的 18 字符、decoder-only 玩具模型",
            "prefix": prefix, "vocab": vocab, "logits": logits.tolist(),
            "reference_temperatures": temperatures,
            "top3_at_1": top3_probs.tolist(),
        },
        "rag_demo": {
            "retriever": "共同的连续两个汉字种数；同分按文档原顺序",
            "documents": DOCS, "questions": QUESTIONS,
            "rankings": rankings,
            "context_budget_examples": {"short": short, "long": long,
                                        "base_chars": 24},
            "scripted_wrong_answer_for_whale": "蓝鲸是鱼。",
        },
    }
    (OUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("温度 0.5/1/2 熵：", *(f"{temperatures[str(t)]['entropy_nats']:.3f}"
                                    for t in (0.5, 1.0, 2.0)))
    print("海豚问题检索前两名：", [d["id"] for d in rankings["food"][:2]])
    print("短上下文纳入：", short["included"], "；长上下文纳入：", long["included"])
    print("结果保存到", OUT)


if __name__ == "__main__":
    main()
