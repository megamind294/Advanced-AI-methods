"""Slide-35 XOR example: exact initial weights and one SGD update.

Run: python train_xor.py
The completed notebook additionally trains on the full XOR truth table.
"""
import torch
from torch import nn


def main():
    torch.set_num_threads(1)
    model = nn.Sequential(
        nn.Linear(2, 2, dtype=torch.float64),
        nn.Sigmoid(),
        nn.Linear(2, 1, dtype=torch.float64),
        nn.Sigmoid(),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[2., 3.], [4., 5.]], dtype=torch.float64))
        model[0].bias.copy_(torch.tensor([1., -2.], dtype=torch.float64))
        model[2].weight.copy_(torch.tensor([[2., 3.]], dtype=torch.float64))
        model[2].bias.copy_(torch.tensor([-4.], dtype=torch.float64))

    x = torch.tensor([[1., 0.]], dtype=torch.float64)
    target = torch.tensor([[1.]], dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    model.train()
    prediction = model(x)                 # Forward pass
    loss = 0.5 * (prediction - target).square().sum()
    optimizer.zero_grad()                 # Clear previous gradients
    loss.backward()                       # Backpropagate the error
    optimizer.step()                      # Update every weight and bias

    model.eval()
    with torch.inference_mode():
        updated_prediction = model(x)
        updated_loss = 0.5 * (updated_prediction - target).square().sum()

    print(f'Before: response={prediction.item():.9f}, loss={loss.item():.9f}')
    print(f'After:  response={updated_prediction.item():.9f}, loss={updated_loss.item():.9f}')
    for name, value in model.named_parameters():
        print(name, value.detach().numpy())
    assert updated_loss < loss


if __name__ == '__main__':
    main()
