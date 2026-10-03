"""第六节 PyTorch 基础。运行：uv run python lessons/lesson-06-pytorch.py
CPU 版即可；固定随机种子。本文件不下载任何数据。
"""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def tensor_examples():
    source = np.array([1., 2., 3.], dtype=np.float32)
    shared = torch.from_numpy(source)
    copied = torch.tensor(source)
    shared[0] = 99
    assert source[0] == 99 and copied[0].item() == 1
    print('NumPy 共享内存：', source, '；独立副本：', copied)
    x = torch.tensor([[1., 2.], [3., 4.]])
    layer = nn.Linear(2, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1., 2.], [-1., 1.], [1., 1.]]))
        layer.bias.copy_(torch.tensor([.1, .2, .3]))
    output = layer(x)
    assert layer.weight.shape == (3, 2) and output.shape == (2, 3)
    torch.testing.assert_close(output, torch.tensor([[5.1, 1.2, 3.3], [11.1, 1.2, 7.3]]))
    print('X、weight、bias、输出形状：', x.shape, layer.weight.shape, layer.bias.shape, output.shape)
    print('nn.Linear 输出：', output.detach())


def gradient_and_step():
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1)
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    x, y = torch.tensor([[2.]]), torch.tensor([[5.]])
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    torch.testing.assert_close(model.weight.grad, torch.tensor([[-12.]]))
    torch.testing.assert_close(model.bias.grad, torch.tensor([-6.]))
    assert model.weight.item() == 1  # backward 只求梯度。
    print('backward 后：w=', model.weight.item(), 'dw=', model.weight.grad.item(), 'db=', model.bias.grad.item())
    # 新建计算图，再反向一次，展示梯度默认累加。
    nn.functional.mse_loss(model(x), y).backward()
    assert model.weight.grad.item() == -24
    print('未清零，第二个新图 backward 后 dw=', model.weight.grad.item())
    optimizer.zero_grad(set_to_none=True)
    nn.functional.mse_loss(model(x), y).backward()
    optimizer.step()
    torch.testing.assert_close(model.weight, torch.tensor([[2.2]]))
    torch.testing.assert_close(model.bias, torch.tensor([.6]))
    print('step 后：w=', model.weight.item(), 'b=', model.bias.item(), '新预测=', model(x).item())


class TinyNetwork(nn.Module):
    """复现第二节：x→Linear(1,1)→tanh→Linear(1,1)。"""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 1)
        self.output = nn.Linear(1, 1)
        with torch.no_grad():
            self.hidden.weight.fill_(.5)
            self.hidden.bias.zero_()
            self.output.weight.fill_(1)
            self.output.bias.zero_()

    def forward(self, x):
        return self.output(torch.tanh(self.hidden(x)))


def network_gradients():
    model = TinyNetwork().double()
    x, y = torch.tensor([[2.]], dtype=torch.float64), torch.tensor([[1.]], dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    h = np.tanh(1.)
    d_prediction = 2 * (h - 1)
    analytic = np.array([d_prediction * (1 - h*h) * 2,
                         d_prediction * (1 - h*h), d_prediction * h, d_prediction])
    automatic = np.array([p.grad.item() for p in model.parameters()])
    assert np.allclose(automatic, analytic, atol=1e-10)
    records = []
    epsilon = 1e-5
    for (name, parameter), gradient in zip(model.named_parameters(), automatic):
        old = parameter.detach().clone()
        with torch.no_grad():
            parameter.copy_(old + epsilon)
            plus = nn.functional.mse_loss(model(x), y).item()
            parameter.copy_(old - epsilon)
            minus = nn.functional.mse_loss(model(x), y).item()
            parameter.copy_(old)
        numeric = (plus - minus) / (2 * epsilon)
        assert np.isclose(gradient, numeric, atol=1e-8)
        records.append((name, float(gradient), float(numeric)))
    old_loss = loss.item()
    optimizer.step()
    with torch.no_grad():
        new_loss = nn.functional.mse_loss(model(x), y).item()
    assert new_loss < old_loss
    print('神经网络梯度（自动 / 数值）：', records)
    print(f'神经网络单步损失：{old_loss:.10f} → {new_loss:.10f}')
    return records


def loader_examples():
    x = torch.arange(6, dtype=torch.float32).reshape(-1, 1)
    y = 2*x + 1
    dataset = TensorDataset(x, y)
    for drop in (False, True):
        loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=drop)
        sizes = [len(batch_x) for batch_x, _ in loader]
        assert sizes == ([4] if drop else [4, 2])
        print(f'batch_size=4, drop_last={drop} → 批大小 {sizes}')


def train_and_save(output, epochs=80):
    device = torch.device('cpu')  # 本课程项目安装 CPU 版。
    x_train = torch.tensor([[-2.], [-1.], [0.], [1.], [2.], [3.]])
    y_train = 2*x_train + 1
    x_val = torch.tensor([[-1.5], [-.5], [.5], [1.5], [2.5]])
    y_val = 2*x_val + 1
    # 即使本例不做缩放，仍保存输入处理约定，避免推理偷偷换单位。
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=2, shuffle=True,
                        generator=torch.Generator().manual_seed(6001), num_workers=0)
    model = nn.Linear(1, 1).to(device)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        online_sum = 0.
        seen = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            assert prediction.shape == batch_y.shape
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            online_sum += loss.item() * len(batch_x)
            seen += len(batch_x)
        model.eval()
        with torch.no_grad():
            train_mse = loss_fn(model(x_train.to(device)), y_train.to(device)).item()
            validation_mse = loss_fn(model(x_val.to(device)), y_val.to(device)).item()
        assert seen == 6
        history.append((epoch, online_sum / seen, train_mse, validation_mse,
                        model.weight.item(), model.bias.item()))
    assert history[-1][3] < 1e-6
    # 把 state_dict 中张量保存到文件；重建时仍须有相同的模型结构。
    checkpoint = {
        'model_state_dict': {name: tensor.detach().cpu().clone()
                             for name, tensor in model.state_dict().items()},
        'architecture': 'Linear(1,1)', 'input_unit': 'original numeric x',
        'epochs': epochs, 'batch_size': 2, 'learning_rate': .05,
    }
    path = output / 'linear-checkpoint.pt'
    torch.save(checkpoint, path)
    restored = nn.Linear(1, 1)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    restored.load_state_dict(loaded['model_state_dict'])
    restored.eval()
    new_x = torch.tensor([[4.], [5.]])
    with torch.no_grad():
        before = model(new_x.to(device)).cpu()
        after = restored(new_x)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    print('保存 / 加载预测一致：', after.flatten().tolist())
    print('CPU 环境：', torch.__version__, 'CUDA 构建版本：', torch.version.cuda,
          'CUDA 可用：', torch.cuda.is_available())
    return history, after.flatten().tolist(), str(path)


def save_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)


def main():
    torch.manual_seed(6001)
    output = Path(__file__).resolve().parent / 'lesson-06-output'
    output.mkdir(exist_ok=True)
    tensor_examples()
    gradient_and_step()
    gradients = network_gradients()
    loader_examples()
    history, predictions, checkpoint = train_and_save(output)
    save_csv(output/'network-gradients.csv', ['parameter', 'autograd', 'finite_difference'], gradients)
    save_csv(output/'linear-training.csv', ['epoch', 'online_batch_loss', 'end_epoch_train_mse',
                                          'validation_mse', 'w', 'b'], history)
    summary = dict(torch_version=str(torch.__version__), device='cpu',
                   cuda_build=torch.version.cuda, validation_mse=history[-1][3],
                   learned_w=history[-1][4], learned_b=history[-1][5],
                   restored_predictions=predictions, checkpoint=checkpoint)
    (output/'run-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('结果目录：', output)


if __name__ == '__main__':
    main()
