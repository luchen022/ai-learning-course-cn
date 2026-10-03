"""第八节 CNN：形状、局部计算、感受野与 CPU 合成图像分类。

运行：uv run python lessons/lesson-08-cnn.py
只依赖当前项目的 PyTorch；数据由代码生成，不下载任何数据。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


OUT = Path(__file__).with_name('lesson-08-output')


def conv_output_size(length: int, kernel: int, stride: int = 1,
                     padding: int = 0, dilation: int = 1) -> int:
    return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def mechanics() -> dict:
    image = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.],
                          [9., 10., 11., 12.], [13., 14., 15., 16.]])
    kernel = torch.tensor([[1., 0.], [0., -1.]])
    manual = torch.tensor([
        [(image[r:r+2, c:c+2] * kernel).sum().item() for c in range(3)]
        for r in range(3)
    ])
    layer = nn.Conv2d(1, 1, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(kernel[None, None])
    actual = layer(image[None, None])[0, 0]
    torch.testing.assert_close(actual, manual, rtol=0, atol=0)
    assert actual.shape == (3, 3) and actual[0, 0].item() == -5
    # 非对称卷积核：翻转核后，手算将不同。PyTorch 的 Conv2d 不翻转核。
    asym = torch.tensor([[1., 2.], [3., 4.]])
    with torch.no_grad():
        layer.weight.copy_(asym[None, None])
    assert layer(image[None, None])[0, 0, 0, 0].item() == 44
    assert (image[:2, :2] * asym.flip((0, 1))).sum().item() != 44

    x = torch.tensor([[[[1., 2.], [3., 4.]],
                       [[5., 6.], [7., 8.]],
                       [[2., 0.], [0., 2.]]]])
    multi = nn.Conv2d(3, 2, kernel_size=2)
    with torch.no_grad():
        multi.weight[0].copy_(torch.tensor([[[1., 1.], [1., 1.]],
                                                   [[-1., -1.], [-1., -1.]],
                                                   [[1., 0.], [0., 1.]]]))
        multi.bias[0] = .5
    y = multi(x)
    assert tuple(multi.weight.shape) == (2, 3, 2, 2)
    assert tuple(y.shape) == (1, 2, 1, 1)
    assert y[0, 0, 0, 0].item() == -11.5

    pool_image = torch.tensor([[[[1., 5., 2., 0.],
                                 [3., 4., 8., 6.],
                                 [7., 1., 2., 9.],
                                 [0., 3., 4., 5.]]]])
    pooled = nn.MaxPool2d(2, stride=2)(pool_image)
    torch.testing.assert_close(pooled, torch.tensor([[[[5., 8.], [7., 9.]]]]))

    for h, w, k, s, p, d in [(7, 9, 3, 2, 1, 1), (8, 8, 3, 1, 0, 2),
                              (6, 5, 2, 2, 0, 1), (16, 16, 3, 1, 1, 1)]:
        test = nn.Conv2d(3, 4, k, stride=s, padding=p, dilation=d)
        expected = (conv_output_size(h, k, s, p, d),
                    conv_output_size(w, k, s, p, d))
        assert tuple(test(torch.zeros(1, 3, h, w)).shape[2:]) == expected

    # 对 Conv3(stride=1,padding=1) → Pool2(stride=2) → Conv3,
    # 用输入的非零梯度位置检查内部一个输出的理论感受野 8×8。
    source = torch.zeros(1, 1, 20, 20, requires_grad=True)
    c1 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    c2 = nn.Conv2d(1, 1, 3, padding=0, bias=False)
    with torch.no_grad():
        c1.weight.fill_(1)
        c2.weight.fill_(1)
    # 用 AvgPool2d 使窗口内每个位置都有非零导数；池化窗口尺寸与 MaxPool 相同。
    r = c2(nn.AvgPool2d(2, stride=2)(c1(source)))
    r[0, 0, 4, 4].backward()
    ys, xs = torch.nonzero(source.grad[0, 0], as_tuple=True)
    assert (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) == (7, 14, 7, 14)

    print('卷积/池化手算、多通道与形状、感受野梯度检查均通过。')
    return {'single_input': image.tolist(), 'single_kernel': kernel.tolist(),
            'single_output': actual.tolist(), 'asymmetric_first_output': 44,
            'multichannel_first_output': y[0, 0, 0, 0].detach().item(),
            'pool_output': pooled[0, 0].tolist(),
            'receptive_field_size': [8, 8], 'receptive_field_range': [7, 14, 7, 14]}


def make_data(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    labels = torch.arange(count) % 2
    images = .15 * torch.randn(count, 1, 16, 16, generator=gen)
    for i, label in enumerate(labels.tolist()):
        line = int(torch.randint(3, 13, (1,), generator=gen))
        if label == 0:
            images[i, 0, :, line:line + 2] += 1.0  # 竖线
        else:
            images[i, 0, line:line + 2, :] += 1.0  # 横线
    return images, labels.long()


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.classifier = nn.Linear(8 * 8 * 8, 2)

    def stages(self, x):
        a = torch.relu(self.conv1(x))
        b = self.pool(a)
        c = torch.relu(self.conv2(b))
        flat = torch.flatten(c, start_dim=1)
        logits = self.classifier(flat)
        return [x, a, b, c, flat, logits]

    def forward(self, x):
        return self.stages(x)[-1]


def evaluate(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return float(criterion(logits, y)), float((logits.argmax(1) == y).float().mean())


def train() -> dict:
    torch.manual_seed(8080)
    x_train, y_train = make_data(160, 8100)
    x_val, y_val = make_data(64, 8200)
    x_test, y_test = make_data(64, 8300)
    model = SmallCNN()
    shapes = [tuple(t.shape) for t in model.stages(x_train[:2])]
    assert shapes == [(2, 1, 16, 16), (2, 4, 16, 16), (2, 4, 8, 8),
                      (2, 8, 8, 8), (2, 512), (2, 2)]
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16,
                        shuffle=True, generator=torch.Generator().manual_seed(8400))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_val = float('inf')
    best_epoch = 0
    best_state = None
    for epoch in range(1, 9):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            assert tuple(logits.shape) == (len(xb), 2)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        tr_loss, tr_acc = evaluate(model, x_train, y_train, criterion)
        va_loss, va_acc = evaluate(model, x_val, y_val, criterion)
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_accuracy': tr_acc,
                        'val_loss': va_loss, 'val_accuracy': va_acc})
        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, x_test, y_test, criterion)
    assert test_acc >= .9, f'示例没有学会简单线条：test_acc={test_acc}'
    # 测试集只评估一次，不用于选择轮数。
    checkpoint = OUT / 'best-cnn.pt'
    torch.save({'state_dict': best_state, 'best_epoch': best_epoch,
                'label_map': {0: '竖线', 1: '横线'}, 'architecture': 'SmallCNN'}, checkpoint)
    restored = SmallCNN()
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'])
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x_test), restored(x_test), rtol=0, atol=0)
    print('阶段形状：', ' → '.join(map(str, shapes)))
    print(f'最佳验证轮数 {best_epoch}；独立测试准确率 {test_acc:.1%}；预测加载一致。')
    return {'shape_trace': [list(s) for s in shapes], 'history': history,
            'best_epoch_by_val_loss': best_epoch, 'test_loss': test_loss,
            'test_accuracy': test_acc, 'train_size': len(x_train),
            'val_size': len(x_val), 'test_size': len(x_test)}


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    result = {'torch_version': torch.__version__, 'device': 'cpu',
              'mechanics': mechanics(), 'experiment': train()}
    (OUT / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with (OUT / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['experiment']['history'][0]))
        writer.writeheader()
        writer.writerows(result['experiment']['history'])


if __name__ == '__main__':
    main()
