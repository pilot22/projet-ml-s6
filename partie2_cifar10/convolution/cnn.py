import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

class CNN(nn.Module):
    """CNN configurable (CIFAR-10, mammographies, etc.)."""

    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        input_size=32,
        filters=(64, 64, 64, 64),
        use_bn=False,
        dropout=0.0,
    ):
        super().__init__()
        f1, f2, f3, f4 = filters

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f3, f4, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(f1) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(f2) if use_bn else nn.Identity()
        self.bn3 = nn.BatchNorm2d(f3) if use_bn else nn.Identity()
        self.bn4 = nn.BatchNorm2d(f4) if use_bn else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        spatial = input_size // 4
        self.fc = nn.Linear(f4 * spatial * spatial, num_classes)

    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))  # spatial /2
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))  # spatial /4
        x = self.bn3(torch.relu(self.conv3(x)))
        x = self.bn4(torch.relu(self.conv4(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def entrainer_cnn(
    X_tr,
    y_tr,
    X_te,
    y_te,
    in_channels=3,
    num_classes=10,
    input_size=32,
    patience=5,
    augment=False,
    filters=(64, 64, 64, 64),
    use_bn=False,
    dropout=0.0,
    threshold=None, 
    epochs=20,
    batch_size=128,
    lr=0.001,
    class_weights=None,
    title="CNN",
    save_loss_path=None,
):
    """Entraîne le CNN et affiche erreurs train/test."""
    print(f"\n=== {title} (PyTorch) ===")

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )

    model = CNN(
        in_channels=in_channels,
        num_classes=num_classes,
        input_size=input_size,
        filters=filters,
        use_bn=use_bn,
        dropout=dropout,
    )
    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_start = time.perf_counter()

    best_loss = float("inf")
    patience_counter = 0
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]) if augment else None

    for epoch in range(epochs):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            if augment_transform is not None:
                X_batch = torch.stack([augment_transform(img) for img in X_batch])
                    
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            acc_tr = (model(X_tr).argmax(1) == y_tr).float().mean().item()
            acc_te = (model(X_te).argmax(1) == y_te).float().mean().item()
            test_loss = criterion(model(X_te), y_te).item()

        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - train_start
        print(
            f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f} "
            f"| Acc train: {acc_tr:.3f} | Acc test: {acc_te:.3f} "
            f"| epoch: {epoch_time:.1f}s | total: {elapsed:.1f}s"
        )

        if test_loss < best_loss - 1e-4:
            best_loss = test_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping à l'epoch {epoch + 1} (patience={patience})")
                model.load_state_dict(best_weights)
                break
    else:
        model.load_state_dict(best_weights)

    print(f"Entraînement terminé en {time.perf_counter() - train_start:.1f}s")

    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_tr).argmax(dim=1).numpy()
        if threshold is not None:
            probs_test  = torch.softmax(model(X_te), dim=1)[:, 1].numpy()
            y_pred_test = (probs_test > threshold).astype(int)
        else:
            y_pred_test = model(X_te).argmax(dim=1).numpy()
        logits = model(X_te[:20])
        probs  = torch.softmax(logits, dim=1)

    print(probs)
    print("Predictions:", probs.argmax(dim=1))
    print("Labels:", y_te[:20])

    err_train = np.mean(y_tr.numpy() != y_pred_train)
    err_test  = np.mean(y_te.numpy() != y_pred_test)
    print(f"Erreur train : {err_train:.4f}")
    print(f"Erreur test  : {err_test:.4f}")

    if save_loss_path:
        plt.figure()
        plt.plot(train_losses)
        plt.title(f"Courbe de loss — {title}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(save_loss_path, dpi=100)
        plt.close()

    return model, y_pred_train, y_pred_test, train_losses