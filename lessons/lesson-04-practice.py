"""第四节：经典机器学习方法的教学实现。只依赖 NumPy。

运行：python lessons/lesson-04-practice.py
输出写到本文件旁的 lesson-04-output/。
使用与 HTML 相同的教学数据、随机数规则、平票规则与算法设置。
这里只处理二维数值特征，没有缺失值；并非生产库的替代品。
"""
from pathlib import Path
import csv
import numpy as np


def random_stream(seed):
    """与浏览器一致的 32 位线性同余随机序列。"""
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state / 4294967296


def gini(data):
    if len(data) == 0:
        return 0.0
    p = float(data[:, 2].mean())
    return 1 - p * p - (1 - p) ** 2


def grow(data, depth, random=None):
    """递归搜索分裂；二分类，平票预测 0；相同分数保留先遇到的分裂。"""
    ones = int(data[:, 2].sum())
    node = dict(label=int(ones > len(data) / 2), n=len(data), ones=ones)
    if depth == 0 or ones in (0, len(data)):
        return node
    features = [int(next(random) * 2)] if random is not None else [0, 1]
    best = None
    for feature in features:
        values = np.unique(data[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2:
            mask = data[:, feature] <= threshold
            left, right = data[mask], data[~mask]
            score = (len(left) * gini(left) + len(right) * gini(right)) / len(data)
            if best is None or score < best[0] - 1e-12:
                best = score, feature, float(threshold), left, right
    if best is not None:
        _, feature, threshold, left, right = best
        node.update(feature=feature, threshold=threshold,
                    left=grow(left, depth - 1, random), right=grow(right, depth - 1, random))
    return node


def predict(tree, x):
    while 'left' in tree:
        tree = tree['left'] if x[tree['feature']] <= tree['threshold'] else tree['right']
    return tree['label']


def make_forest(data, depth, count, seed):
    random = random_stream(seed)
    forest = []
    for _ in range(count):
        indices = [int(next(random) * len(data)) for _ in range(len(data))]
        forest.append(grow(data[indices], depth, random))
    return forest


def forest_predict(forest, x):
    return int(sum(predict(tree, x) for tree in forest) > len(forest) / 2)


def toy_data(noise=False):
    data = np.array([[x, y, int(x >= 3 and y >= 3)]
                     for x in (1, 2, 4, 5) for y in (1, 2, 4, 5)], dtype=float)
    if noise:
        data[(data[:, 0] == 2) & (data[:, 1] == 2), 2] = 1
    return data


def make_data(n, seed):
    random = random_stream(seed)
    data = []
    for _ in range(n):
        x, y = next(random), next(random)
        label = int((x > .5) != (y > .5))
        if next(random) < .08:
            label = 1 - label
        data.append((x, y, label))
    return np.array(data)


def accuracy(data, predictor):
    return float(np.mean([predictor(x) == x[2] for x in data]))


def logistic_regression(train, steps=2000, learning_rate=.3):
    features = np.column_stack((np.ones(len(train)), train[:, :2]))
    weights = np.zeros(3)
    for _ in range(steps):
        probability = 1 / (1 + np.exp(-(features @ weights)))
        gradient = features.T @ (probability - train[:, 2]) / len(train)
        weights -= learning_rate * gradient
    return weights


def compare_models():
    train, validation = make_data(80, 4001), make_data(120, 4002)
    weights = logistic_regression(train)
    linear = lambda x: int(1 / (1 + np.exp(-(weights[0] + weights[1:] @ x[:2]))) >= .5)
    rows = []
    for depth in (1, 2, 4, 8):
        tree = grow(train, depth)
        forest = make_forest(train, depth, 21, 4003)
        for name, fn in [('logistic', linear), ('tree', lambda x: predict(tree, x)),
                         ('forest', lambda x: forest_predict(forest, x))]:
            rows.append((depth, name, accuracy(train, fn), accuracy(validation, fn)))
    return rows


def boosting(rate=.5, rounds=8):
    x = np.arange(6, dtype=float)
    y = np.array([1., 1., 2., 4., 4., 5.])
    prediction = np.full_like(y, y.mean())
    rows = [(0, '', '', '', float(np.mean((prediction - y) ** 2)))]
    for step in range(1, rounds + 1):
        residual = y - prediction
        best = None
        for j in range(len(x) - 1):
            left, right = residual[:j + 1], residual[j + 1:]
            a, b = float(left.mean()), float(right.mean())
            error = float(np.sum((left - a) ** 2) + np.sum((right - b) ** 2))
            if best is None or error < best[0] - 1e-12:
                best = error, (x[j] + x[j + 1]) / 2, a, b
        _, threshold, a, b = best
        prediction += rate * np.where(x <= threshold, a, b)
        rows.append((step, threshold, a, b, float(np.mean((prediction - y) ** 2))))
    return prediction, rows


POINTS = np.array([[.6, .8], [1, 1.3], [1.4, .7], [.8, 1.7], [1.5, 1.5],
                   [4.5, .7], [5, 1.2], [5.5, .8], [4.7, 1.7], [5.4, 1.6],
                   [2.5, 4.5], [3, 5], [3.5, 4.7], [2.7, 5.5], [3.4, 5.6]])


def kmeans(points, centers, max_rounds=50):
    """空组保留原中心；距离相同取编号较小的组。每半步记录 SSE。"""
    centers = np.array(centers, dtype=float, copy=True)
    history = []
    for step in range(1, max_rounds + 1):
        distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = distances.argmin(axis=1)
        history.append((step, 'assign', float(distances[np.arange(len(points)), assignments].sum())))
        updated = centers.copy()
        for c in range(len(centers)):
            group = points[assignments == c]
            if len(group):
                updated[c] = group.mean(axis=0)
        movement = np.sum((updated - centers) ** 2, axis=1)
        centers = updated
        sse = float(np.sum((points - centers[assignments]) ** 2))
        history.append((step, 'update', sse))
        if np.all(movement < 1e-16):
            break
    return centers, assignments, history


PCA_POINTS = np.array([[1, 1.1], [2, 1.4], [3, 2.2], [4, 2.4],
                       [5, 3.2], [6, 3.4], [7, 4.2], [8, 4.1]])


def pca(points):
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = centered.T @ centered / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    if direction[0] < 0:  # 符号只是约定，便于与 HTML 对照。
        direction = -direction
    scores = centered @ direction[:, None]
    reconstructed = mean + scores @ direction[None, :]
    error = float(np.mean(np.sum((points - reconstructed) ** 2, axis=1)))
    ratio = float(eigenvalues[-1] / eigenvalues.sum())
    return mean, direction, scores, reconstructed, error, ratio


def save(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    output = Path(__file__).resolve().parent / 'lesson-04-output'
    output.mkdir(exist_ok=True)
    for noise in (False, True):
        data = toy_data(noise)
        print(f'\n扰动={noise}')
        for depth in (1, 2, 4):
            tree = grow(data, depth)
            print(f'深度 {depth}：训练准确率 {accuracy(data, lambda x: predict(tree, x)):.1%}；'
                  f'(4.5,1.5) 预测 {predict(tree, [4.5, 1.5])}')
    comparison = compare_models()
    save(output / 'model-comparison.csv', ['depth', 'model', 'train_accuracy', 'validation_accuracy'], comparison)
    print('\n同数据比较（训练 / 验证）')
    for depth, name, train, val in comparison:
        print(f'{depth} {name:8s}: {train:.1%} / {val:.1%}')
    _, records = boosting()
    save(output / 'boosting.csv', ['round', 'threshold', 'left_residual', 'right_residual', 'train_mse'], records)
    print(f'\n提升：初始 MSE={records[0][-1]:.6f}，8 轮后 MSE={records[-1][-1]:.6f}')
    center_rows, history_rows = [], []
    for k in (2, 3):
        for init, seeds in [('spread', [[1, 1], [5, 1], [3, 5]]),
                            ('near', [[.6, .8], [1, 1], [1.4, .7]])]:
            centers, assignments, history = kmeans(POINTS, seeds[:k])
            for c, point in enumerate(centers):
                center_rows.append((k, init, c + 1, *point, int(np.sum(assignments == c))))
            history_rows.extend((k, init, *r) for r in history)
            print(f'K={k}，初始化={init}，SSE={history[-1][-1]:.6f}，中心={centers.tolist()}')
    save(output / 'kmeans-centers.csv', ['k', 'init', 'cluster', 'x1', 'x2', 'count'], center_rows)
    save(output / 'kmeans-history.csv', ['k', 'init', 'round', 'phase', 'sse'], history_rows)
    mean, direction, scores, reconstructed, error, ratio = pca(PCA_POINTS)
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    print(f'\nPCA：均值={mean}，方向={direction}，角度={angle:.4f}°')
    print(f'形状：{PCA_POINTS.shape} → {scores.shape} → {reconstructed.shape}')
    print(f'保留方差 {ratio:.4%}，平均平方重建距离 {error:.6f}')
    save(output / 'pca-projection.csv', ['sample', 'x1', 'x2', 't', 'reconstructed_x1', 'reconstructed_x2'],
         [(i, *x, float(scores[i, 0]), *reconstructed[i]) for i, x in enumerate(PCA_POINTS)])
    print(f'\n记录已保存：{output}')


if __name__ == '__main__':
    main()
