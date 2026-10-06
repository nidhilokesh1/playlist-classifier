"""1-hidden-layer neural network (sigmoid) + LogSoftmax, NLL loss, in PyTorch."""
import argparse
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from common import load_data, save_meta, save_confusion, ART, RES


def build(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Sigmoid(),
                         nn.Linear(hidden, n_out), nn.LogSoftmax(dim=1))


def main(a):
    torch.manual_seed(42)
    d = load_data(per_class=a.per_class)
    save_meta(d)
    C = len(d.classes)
    hidden = a.hidden or C            # paper used C hidden neurons
    Xtr = torch.tensor(d.Xtr, dtype=torch.float32)
    Xte = torch.tensor(d.Xte, dtype=torch.float32)
    ytr, yte = torch.tensor(d.ytr), torch.tensor(d.yte)

    model = build(Xtr.shape[1], hidden, C)
    loss_fn = nn.NLLLoss()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.l2)  # weight_decay = L2

    hist = {"loss": [], "train_acc": [], "test_acc": []}
    for ep in range(1, a.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)   # full-batch gradient descent
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            tr = (model(Xtr).argmax(1) == ytr).float().mean().item()
            te = (model(Xte).argmax(1) == yte).float().mean().item()
        hist["loss"].append(loss.item())
        hist["train_acc"].append(tr)
        hist["test_acc"].append(te)
        if ep % 25 == 0 or ep == 1:
            print(f"epoch {ep:4d}  loss {loss.item():.4f}  train {tr:.3f}  test {te:.3f}")

    with torch.no_grad():
        pred = model(Xte).argmax(1).numpy()
    rep = classification_report(d.yte, pred, target_names=d.classes)
    print(f"\nFINAL  train {hist['train_acc'][-1]:.3f}  test {hist['test_acc'][-1]:.3f}\n")
    print(rep)
    (RES / "nn_report.txt").write_text(
        f"hidden={hidden} epochs={a.epochs} lr={a.lr} l2={a.l2}\n"
        f"train acc {hist['train_acc'][-1]:.3f}, test acc {hist['test_acc'][-1]:.3f}\n\n{rep}")
    save_confusion(d.yte, pred, d.classes, RES / "nn_confusion.png", "Neural network")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["loss"])
    ax[0].set_title("Training loss (NLL)")
    ax[0].set_xlabel("epoch")
    ax[1].plot(hist["train_acc"], label="train")
    ax[1].plot(hist["test_acc"], label="test")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(RES / "nn_curves.png", dpi=150)

    torch.save({"state": model.state_dict(), "n_in": Xtr.shape[1],
                "hidden": hidden, "n_out": C}, ART / "nn.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per_class", type=int, default=150)
    p.add_argument("--hidden", type=int, default=0, help="0 = number of classes")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--l2", type=float, default=1e-4)
    main(p.parse_args())
