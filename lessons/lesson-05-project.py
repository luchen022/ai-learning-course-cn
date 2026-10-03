"""第五节：共享单车高需求分类的完整教学项目（合成数据，仅依赖 NumPy）。
运行：python lessons/lesson-05-project.py
默认比较：多数类、逻辑回归、深度 1/3/6 的树；按 3 折平均 F1 选模型。
预处理在每折训练部分拟合；确定方案后，在 120 个开发样本重训，测试 60 个保留样本。
结果与 HTML 使用相同数据、随机数、折划分及超参数。
"""
from pathlib import Path
import csv
import json
import argparse
import numpy as np


def random_stream(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state / 4294967296


def dataset():
    random = random_stream(5001)
    rows = []
    for i in range(180):
        temperature = np.floor((5 + 30 * next(random)) * 10 + .5) / 10
        rain = int(next(random) < .3)
        station = ['A', 'B', 'C'][int(next(random) * 3)]
        bias = {'A': -.8, 'B': .9, 'C': .2}[station]
        probability = 1 / (1 + np.exp(-(.22 * (temperature - 20) - 1.8 * rain + bias)))
        label = int(next(random) < probability)
        missing_temperature = next(random) < .15
        missing_station = next(random) < .1
        rows.append(dict(id=i, temperature=None if missing_temperature else float(temperature),
                         rain=rain, station=None if missing_station else station, y=label))
    return rows[:120], rows[120:]


def folds(rows):
    """每个类别分别按固定洗牌序列轮流放入三组。没有把标签作为输入特征。"""
    random = random_stream(5002)
    groups = [[] for _ in range(3)]
    for label in (0, 1):
        indices = [i for i, row in enumerate(rows) if row['y'] == label]
        for j in range(len(indices) - 1, 0, -1):
            k = int(next(random) * (j + 1))
            indices[j], indices[k] = indices[k], indices[j]
        for j, index in enumerate(indices):
            groups[j % 3].append(index)
    return [sorted(group) for group in groups]


def fit_preprocessor(rows, impute='median', scale=True, use_station=True):
    observed = np.array([r['temperature'] for r in rows if r['temperature'] is not None])
    fill = float(np.median(observed)) if impute == 'median' and len(observed) else 0.0
    filled = np.array([fill if r['temperature'] is None else r['temperature'] for r in rows])
    mean, std = float(filled.mean()), float(filled.std())
    if std == 0:
        std = 1.0
    categories = sorted({r['station'] or 'MISSING' for r in rows}) if use_station else []
    return dict(fill=fill, mean=mean, std=std, scale=scale, categories=categories)


def transform(rows, p):
    data = []
    for row in rows:
        temperature = p['fill'] if row['temperature'] is None else row['temperature']
        value = (temperature - p['mean']) / p['std'] if p['scale'] else temperature
        category = row['station'] or 'MISSING'
        data.append([value, row['rain']] + [int(category == c) for c in p['categories']])
    return np.array(data, dtype=float)


def impurity(y):
    if not len(y):
        return 0.0
    p = float(y.mean())
    return 1 - p * p - (1 - p) ** 2


def grow(x, y, depth, min_leaf=4):
    node = dict(probability=float(y.mean()), n=len(y))
    if depth == 0 or len(y) < 2 * min_leaf or np.all(y == y[0]):
        return node
    best = None
    for feature in range(x.shape[1]):
        values = np.unique(x[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2:
            mask = x[:, feature] <= threshold
            if mask.sum() < min_leaf or (~mask).sum() < min_leaf:
                continue
            score = (mask.sum() * impurity(y[mask]) + (~mask).sum() * impurity(y[~mask])) / len(y)
            if best is None or score < best[0] - 1e-12:
                best = score, feature, float(threshold), mask
    if best is not None:
        _, feature, threshold, mask = best
        node.update(feature=feature, threshold=threshold,
                    left=grow(x[mask], y[mask], depth - 1, min_leaf),
                    right=grow(x[~mask], y[~mask], depth - 1, min_leaf))
    return node


def tree_probability(tree, x):
    while 'left' in tree:
        tree = tree['left'] if x[tree['feature']] <= tree['threshold'] else tree['right']
    return tree['probability']


CANDIDATES = ['majority', 'logistic', 'tree1', 'tree3', 'tree6']
NAMES = {'majority': '多数类基线', 'logistic': '逻辑回归', 'tree1': '决策树（深度 1）',
         'tree3': '决策树（深度 3）', 'tree6': '决策树（深度 6）'}


def train_model(x, y, candidate):
    if candidate == 'majority':
        # 真正的多数类硬预测：平票为 1，与统一 p>=0.5 的决策规则一致。
        return dict(kind='majority', probability=float(y.mean() >= .5))
    if candidate.startswith('tree'):
        return dict(kind='tree', tree=grow(x, y, int(candidate[4:])))
    z = np.column_stack((np.ones(len(x)), x))
    weights = np.zeros(z.shape[1])
    for _ in range(600):
        score = np.clip(z @ weights, -40, 40)
        probability = 1 / (1 + np.exp(-score))
        gradient = z.T @ (probability - y) / len(y)
        gradient[1:] += .01 * weights[1:]  # L2 项；偏置不惩罚。
        weights -= .15 * gradient
    return dict(kind='logistic', weights=weights)


def predict_probability(model, x):
    if model['kind'] == 'majority':
        return np.full(len(x), model['probability'])
    if model['kind'] == 'tree':
        return np.array([tree_probability(model['tree'], row) for row in x])
    score = np.clip(model['weights'][0] + x @ model['weights'][1:], -40, 40)
    return 1 / (1 + np.exp(-score))


def metrics(y, probability):
    predicted = probability >= .5
    y = np.array(y, dtype=bool)
    tp = int(np.sum(predicted & y)); fp = int(np.sum(predicted & ~y))
    fn = int(np.sum(~predicted & y)); tn = int(np.sum(~predicted & ~y))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=(tp + tn) / len(y),
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if tp + fn else 0.,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)


def cross_validate(rows, impute='median', scale=True, use_station=True):
    groups = folds(rows)
    results = []
    for candidate in CANDIDATES:
        out_of_fold = np.zeros(len(rows))
        records = []
        for group in groups:
            held = set(group)
            training = [r for i, r in enumerate(rows) if i not in held]
            validation = [rows[i] for i in group]
            p = fit_preprocessor(training, impute, scale, use_station)
            x, vx = transform(training, p), transform(validation, p)
            y = np.array([r['y'] for r in training])
            model = train_model(x, y, candidate)
            train_metrics = metrics(y, predict_probability(model, x))
            probability = predict_probability(model, vx)
            val_metrics = metrics([r['y'] for r in validation], probability)
            out_of_fold[group] = probability
            records.append(dict(preprocessor=p, train=train_metrics, val=val_metrics))
        f1 = np.array([r['val']['f1'] for r in records])
        results.append(dict(candidate=candidate, folds=records, oof=out_of_fold,
                            mean=float(f1.mean()), std=float(f1.std())))
    return results


def save(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)


def main(impute="median", scale=True, use_station=True):
    output = Path(__file__).resolve().parent / 'lesson-05-output'
    output.mkdir(exist_ok=True)
    dev, test = dataset()
    save(output / 'development-data.csv', ['id', 'temperature', 'rain', 'station', 'y'],
         [(r['id'], r['temperature'], r['rain'], r['station'], r['y']) for r in dev])
    results = cross_validate(dev, impute, scale, use_station)
    records = []
    for result in results:
        print(f"{NAMES[result['candidate']]}：平均 F1={result['mean']:.6f}，折间标准差={result['std']:.6f}")
        for i, fold in enumerate(result['folds']):
            records.append((result['candidate'], i + 1, fold['train']['accuracy'],
                            fold['val']['accuracy'], fold['val']['f1'], fold['preprocessor']['fill']))
    save(output / 'cross-validation.csv', ['model', 'fold', 'train_accuracy', 'validation_accuracy', 'validation_f1', 'train_temperature_fill'], records)
    # 候选顺序固定，相同平均 F1 时保留先遇到的模型。
    selected = max(results, key=lambda r: r['mean'])
    selected_id = selected['candidate']
    pooled = metrics([r['y'] for r in dev], selected['oof'])
    error_rows = []
    for i, r in enumerate(dev):
        probability = float(selected['oof'][i]); predicted = int(probability >= .5)
        category = ('TP' if r['y'] else 'FP') if predicted else ('FN' if r['y'] else 'TN')
        error_rows.append((r['id'], r['temperature'], r['rain'], r['station'], r['y'], probability, predicted, category))
    save(output / 'out-of-fold-predictions.csv', ['id', 'temperature', 'rain', 'station', 'y', 'score', 'prediction', 'category'], error_rows)
    # 开发流程完成后再拟合完整开发集，测试仅在这里使用。
    p = fit_preprocessor(dev, impute, scale, use_station)
    model = train_model(transform(dev, p), np.array([r['y'] for r in dev]), selected_id)
    final = metrics([r['y'] for r in test], predict_probability(model, transform(test, p)))
    save(output / 'final-test.csv', ['model', 'tp', 'fp', 'fn', 'tn', 'accuracy', 'precision', 'recall', 'f1'],
         [(selected_id, *(final[k] for k in ['tp', 'fp', 'fn', 'tn', 'accuracy', 'precision', 'recall', 'f1']))])
    def jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: jsonable(v) for k, v in value.items()}
        return value
    bundle = dict(version=1, candidate=selected_id, threshold=.5,
                  config=dict(impute=impute, scale=scale, use_station=use_station),
                  preprocessor=p, model=jsonable(model))
    (output / "model-pipeline.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f'''# 第五节项目实验报告

这是可复现的合成数据教学实验，不代表真实共享单车运营效果。

## 任务与数据

预测一个站点场景是否为高需求，正类 y=1。特征为预测时已知的温度、是否下雨、站点类型；不输入样本编号与目标标签。
180 个独立合成场景：120 个开发样本、60 个保留测试样本。开发集正类 {sum(r['y'] for r in dev)}/120，温度缺失 {sum(r['temperature'] is None for r in dev)}，站点缺失 {sum(r['station'] is None for r in dev)}。
真实按日期连续采集的数据应考虑按时间划分，本教学数据没有日期与跨行依赖。

## 预处理与评估协议

配置：温度填充={impute}，标准化={scale}，使用站点={use_station}。
每折只使用该折训练部分拟合温度填充规则、填充后的均值与标准差、站点独热编码词表。缺失站点使用 MISSING 类别；未知类别编码为全零。
数值特征为处理后的温度和 0/1 下雨，启用的站点类别使用独热编码。开发集采用固定的分层 3 折交叉验证；选择指标是平均 F1，预测阈值固定为 0.5。
逻辑回归训练 600 次、学习率 0.15、梯度中 L2 系数 0.01，偏置不惩罚。树最小叶子 4 个样本。

| 模型 | 3 折平均 F1 | 折间标准差 |
|---|---:|---:|
'''
    for r in results:
        report += f"| {NAMES[r['candidate']]} | {r['mean']:.6f} | {r['std']:.6f} |\n"
    report += f'''
## 选择与错误分析

按平均 F1 选择：**{NAMES[selected_id]}**。平均 F1 为 {selected['mean']:.6f}。
合并 120 个折外预测后的混淆矩阵：TP={pooled['tp']}，FP={pooled['fp']}，FN={pooled['fn']}，TN={pooled['tn']}；合并 F1={pooled['f1']:.6f}。合并 F1 与逐折 F1 的平均不是同一种统计量。
FP 意味着实际非高需求却被预测为高需求，可能导致多备车辆；FN 意味着高需求被漏掉，可能导致备车不足。本实验用 F1 作为预先确定的教学选择指标，没有估计真实错误成本。
完整折外样本预测见 `out-of-fold-predictions.csv`。错误分析应先查标签、缺失与特征，再提出单一改动并重新验证；按子组看到的差异不能单独证明原因。

## 最终测试

锁定预处理与模型后，在完整 120 个开发样本重训；温度填充值 {p['fill']:.4f}。
60 个保留测试样本：TP={final['tp']}、FP={final['fp']}、FN={final['fn']}、TN={final['tn']}。
准确率={final['accuracy']:.6f}，精确率={final['precision']:.6f}，召回率={final['recall']:.6f}，F1={final['f1']:.6f}。
这份结果用于最终报告，不据此改选模型或阈值。阅读过本报告的测试成绩后，相同测试数据也不能再当作未知的最终考试。

## 限制与下一步

样本很少，标签按概率生成，特征不包含所有影响需求的因素。折间标准差只是本次三折分数的离散程度，不是置信区间。
下一轮研究可预先提出加入新特征或改变模型的假设，在新的开发流程中验证；如据本测试结果改进，需使用新的独立评估数据。

## 复现

`python lessons/lesson-05-project.py`（需要 NumPy）。所有随机数、折与候选顺序固定。
可选参数：`--impute zero`、`--no-scale`、`--no-station`。运行会覆盖输出目录中的当前记录。
`model-pipeline.json` 保存当前所选模型、预处理参数和阈值，可用 `json.load` 读回，结合本文件的 `transform` 和 `predict_probability` 预测新记录。
'''
    (output / 'project-report.md').write_text(report, encoding='utf-8')
    print(f"选择：{NAMES[selected_id]}；OOF F1={pooled['f1']:.6f}；最终测试 F1={final['f1']:.6f}")
    print(f'结果与报告：{output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--impute', choices=['median', 'zero'], default='median')
    parser.add_argument('--no-scale', action='store_true')
    parser.add_argument('--no-station', action='store_true')
    args = parser.parse_args()
    main(args.impute, not args.no_scale, not args.no_station)
