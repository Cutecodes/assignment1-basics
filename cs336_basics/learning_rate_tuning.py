
import torch
from cs336_basics.optimizer import SGD

def train(lr):
    losses = []
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        losses.append(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
    
    print(f"learn_rate: {lr} loss:{losses}")

def main():
    learn_rate = [1e-1, 1e1, 1e2, 1e3]

    for lr in learn_rate:
        train(lr)

if __name__ == "__main__":
    main()