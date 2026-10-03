"""第七节：训练更可靠的神经网络。
运行 uv run python lessons/lesson-07-training.py；CPU，无外部数据或新依赖。
三种子、单因素实验仅用于本例，不宣称某种配置普遍更好。
"""
from pathlib import Path
import copy
import csv
import json
import math
import statistics

import torch
from torch import nn

OUT = Path(__file__).with_name('lesson-07-output')
VARIANTS = ['baseline', 'sigmoid', 'zero_init', 'sgd', 'dropout', 'layernorm', 'schedule']
SEEDS = [7001, 7002, 7003]
EPOCHS = 120


def mechanism_checks():
    gradients = []
    for depth in [1, 3, 6, 10]:
        x = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        h = x
        for _ in range(depth):
            # 每层输入恰好是 0，局部导数为 1/4，线性权重为 1。
            h = torch.sigmoid(h) - .5
        h.backward()
        expected = .25 ** depth
        assert math.isclose(x.grad.item(), expected, rel_tol=1e-12)
        gradients.append({'depth': depth, 'input_grad': x.grad.item()})

    param = nn.Parameter(torch.tensor(1., dtype=torch.float64))
    optim = torch.optim.Adam([param], lr=.1, betas=(.9, .999), eps=1e-8)
    manual, m, v = 1., 0., 0.
    adam_rows = []
    for t, g in enumerate([2., -1., 3.], start=1):
        param.grad = torch.tensor(g, dtype=torch.float64)
        optim.step()
        m = .9*m + .1*g
        v = .999*v + .001*g*g
        mh, vh = m/(1-.9**t), v/(1-.999**t)
        manual -= .1*mh/(math.sqrt(vh)+1e-8)
        assert math.isclose(param.item(), manual, rel_tol=1e-12)
        adam_rows.append({'t': t, 'g': g, 'm': m, 'v': v, 'm_hat': mh,
                          'v_hat': vh, 'parameter': param.item()})

    torch.manual_seed(7100)
    x = torch.arange(1., 65.)
    drop = nn.Dropout(.5)
    drop.train()
    with torch.no_grad():
        train_output = drop(x)
    assert torch.all((train_output == 0) | (train_output == 2*x))
    assert (train_output == 0).any() and (train_output != 0).any()
    drop.eval()
    torch.testing.assert_close(drop(x), x, rtol=0, atol=0)

    x = torch.tensor([[1., 10., 100.], [3., 14., 120.]], dtype=torch.float64)
    bn = nn.BatchNorm1d(3, affine=False, momentum=.1, eps=1e-5).double()
    bn.train()
    bn_output = bn(x)
    manual_bn = (x-x.mean(0))/torch.sqrt(x.var(0, correction=0)+1e-5)
    torch.testing.assert_close(bn_output, manual_bn)
    torch.testing.assert_close(bn.running_mean, .1*x.mean(0))
    torch.testing.assert_close(bn.running_var, .9*torch.ones(3)+.1*x.var(0, correction=1))
    bn.eval()
    torch.testing.assert_close(bn(x), (x-bn.running_mean)/torch.sqrt(bn.running_var+1e-5))
    ln = nn.LayerNorm(3, elementwise_affine=False, eps=1e-5).double()
    ln_output = ln(x)
    torch.testing.assert_close(ln_output, (x-x.mean(1, keepdim=True))/
                               torch.sqrt(x.var(1, correction=0, keepdim=True)+1e-5))
    ln.eval()
    torch.testing.assert_close(ln(x), ln_output)
    result = {'sigmoid_chain': gradients, 'adam': adam_rows,
              'dropout_zero_count': int((train_output == 0).sum()),
              'batchnorm_train': bn_output.tolist(), 'layernorm': ln_output.tolist(),
              'running_mean': bn.running_mean.tolist(), 'running_var': bn.running_var.tolist()}
    print('机制核对通过：链式梯度、Adam、Dropout、BatchNorm、LayerNorm。')
    return result


def make_data():
    generator = torch.Generator().manual_seed(7070)
    raw_train = 4*torch.rand(128, 2, generator=generator)-2
    raw_val = 4*torch.rand(256, 2, generator=generator)-2
    label = lambda x: ((x[:, :1] > 0) != (x[:, 1:] > 0)).float()
    y_train, y_val = label(raw_train), label(raw_val)
    flip = torch.randperm(128, generator=generator)[:13]
    y_train[flip] = 1-y_train[flip]
    mean = raw_train.mean(0, keepdim=True)
    std = raw_train.std(0, correction=0, keepdim=True).clamp_min(1e-8)
    return (raw_train-mean)/std, y_train, (raw_val-mean)/std, y_val, mean, std


class Classifier(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.first = nn.Linear(2, 16)
        self.second = nn.Linear(16, 16)
        self.output = nn.Linear(16, 1)
        self.norm1 = nn.LayerNorm(16) if variant == 'layernorm' else nn.Identity()
        self.norm2 = nn.LayerNorm(16) if variant == 'layernorm' else nn.Identity()
        self.activation = nn.Sigmoid() if variant == 'sigmoid' else nn.ReLU()
        self.dropout = nn.Dropout(.3 if variant == 'dropout' else 0.)
        for layer in [self.first, self.second]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if variant == 'zero_init':
            for parameter in self.parameters():
                nn.init.zeros_(parameter)

    def forward(self, x):
        x = self.dropout(self.activation(self.norm1(self.first(x))))
        x = self.dropout(self.activation(self.norm2(self.second(x))))
        return self.output(x)  # 原始 logits；不要再 sigmoid 后送进 BCEWithLogitsLoss。


def train_one(variant, seed, data):
    x, y, vx, vy, mean, std = data
    torch.manual_seed(seed)
    model = Classifier(variant)
    optimizer = (torch.optim.SGD(model.parameters(), lr=.01) if variant == 'sgd'
                 else torch.optim.Adam(model.parameters(), lr=.01))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=.5) if variant == 'schedule' else None
    criterion = nn.BCEWithLogitsLoss()
    best_val, best_epoch, best_state = float('inf'), 0, None
    history = []
    for epoch in range(1, EPOCHS+1):
        lr_used = optimizer.param_groups[0]['lr']
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # 本实验全批训练：一轮一次更新，所有配置保持一致。
        assert logits.shape == y.shape == (128, 1)
        loss = criterion(logits, y)
        assert torch.isfinite(loss)
        loss.backward()
        grad_norm = math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None))
        if variant == 'zero_init':
            assert torch.count_nonzero(model.first.weight.grad) == 0
            assert torch.count_nonzero(model.second.weight.grad) == 0
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_logits, val_logits = model(x), model(vx)
            train_loss = criterion(train_logits, y).item()
            val_loss = criterion(val_logits, vy).item()
            accuracy = ((val_logits >= 0) == vy.bool()).float().mean().item()
        history.append({'variant': variant, 'seed': seed, 'epoch': epoch,
                        'lr_used': lr_used, 'train_bce_eval': train_loss,
                        'val_bce_eval': val_loss, 'val_accuracy': accuracy, 'grad_norm': grad_norm})
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if scheduler:
            scheduler.step()  # 本例在轮末调用；下轮使用更新后的学习率。
    if variant == 'schedule':
        assert history[39]['lr_used'] == .01 and history[40]['lr_used'] == .005
        assert history[79]['lr_used'] == .005 and history[80]['lr_used'] == .0025
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_accuracy = ((model(vx) >= 0) == vy.bool()).float().mean().item()
    if variant == 'baseline' and seed == SEEDS[0]:
        checkpoint = {'state_dict': best_state, 'variant': variant, 'mean': mean, 'std': std,
                      'best_epoch': best_epoch, 'input_features': 2, 'hidden_features': 16}
        torch.save(checkpoint, OUT/'best-baseline.pt')
        loaded = torch.load(OUT/'best-baseline.pt', map_location='cpu', weights_only=True)
        restored = Classifier(loaded['variant'])
        restored.load_state_dict(loaded['state_dict'])
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(vx), restored(vx), rtol=0, atol=0)
    return history, {'variant': variant, 'seed': seed, 'best_epoch': best_epoch,
                     'best_val_bce': best_val, 'accuracy_at_best_bce': best_accuracy,
                     'final_val_bce': history[-1]['val_bce_eval']}


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    mechanisms = mechanism_checks()
    data = make_data()
    histories, runs = [], []
    # 同一种子各配置的共同层使用相同初始权重；只改变命名的主要因素。
    for variant in VARIANTS:
        for seed in SEEDS:
            history, run = train_one(variant, seed, data)
            histories.extend(history)
            runs.append(run)
        subset = [r for r in runs if r['variant'] == variant]
        print(variant, '最佳验证 BCE 均值', round(statistics.mean(r['best_val_bce'] for r in subset), 4))
    summaries = []
    curves = {}
    for variant in VARIANTS:
        subset = [r for r in runs if r['variant'] == variant]
        values = [r['best_val_bce'] for r in subset]
        summaries.append({'variant': variant, 'best_val_mean': statistics.mean(values),
                          'best_val_sd': statistics.stdev(values),
                          'best_accuracy_mean': statistics.mean(r['accuracy_at_best_bce'] for r in subset),
                          'best_epochs': [r['best_epoch'] for r in subset]})
        curves[variant] = [{'epoch': epoch,
                            'train': statistics.mean(r['train_bce_eval'] for r in histories if r['variant'] == variant and r['epoch'] == epoch),
                            'val': statistics.mean(r['val_bce_eval'] for r in histories if r['variant'] == variant and r['epoch'] == epoch)}
                           for epoch in range(1, EPOCHS+1)]
    for filename, rows in [('training.csv', histories), ('runs.csv', runs), ('summary.csv', summaries)]:
        with (OUT/filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {'torch_version': torch.__version__, 'device': 'cpu', 'seeds': SEEDS,
              'epochs': EPOCHS, 'train_size': 128, 'val_size': 256, 'flipped_train_labels': 13,
              'summaries': summaries, 'curves': curves, 'mechanisms': mechanisms}
    (OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('21 次训练完成；最佳基线保存/加载验证通过。结果目录：', OUT)


if __name__ == '__main__':
    main()
