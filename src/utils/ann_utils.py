import torch


def train_step(model, criterion, X, y, learning_rate, lmbda):

    for p in model.parameters():
        p.grad = None

    y_pred = model(X)
    y_pred = y_pred.squeeze()

    data_loss = criterion(y_pred, y)

    l2_reg = None
    for W in model.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)

    loss = data_loss + lmbda * l2_reg

    loss.backward()

    for p in model.parameters():
        p.data -= learning_rate * p.grad

    return loss.item()


def train(
    X_train,
    y_train,
    X_val,
    y_val,
    model,
    criterion,
    learning_rate,
    l2_lambda,
    device,
    batch_size=64,
    epochs=10,
    logs: bool = False,
    generator_seed=None,
):
    model.train()

    # Get number of batches
    if batch_size == -1:
        batch_size = X_train.shape[0]
    n_batches = X_train.shape[0] // batch_size

    # Keep track of records
    losses = []
    losses_test = []
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.shape[0])

        data_X = X_train[permutation]
        data_y = y_train[permutation]

        total_loss = 0
        for batch in range(n_batches):
            if batch == n_batches - 1:
                end = -1
            else:
                end = (batch + 1) * batch_size
            start = batch * batch_size
            X_batch = data_X[start:end].to(device)
            y_batch = data_y[start:end].to(device)

            loss = train_step(
                model, criterion, X_batch, y_batch, learning_rate, l2_lambda
            )
            total_loss += loss

        if logs and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Epoch {epoch + 1}/{epochs}:", total_loss / n_batches)
        losses.append(total_loss / n_batches)

        if X_val is not None:
            with torch.no_grad():
                model.eval()
                y_test_pred = model(X_val).squeeze()
                losses_test.append(criterion(y_test_pred, y_val).item())
                model.train()
    return losses, losses_test


from matplotlib import pyplot as plt
from utils.utils import classification_report


def plot(losses, losses_val, model, X_val, y_val):
    plt.figure
    plt.plot(losses, label="Train")
    plt.plot(losses_val, label="Val")
    plt.legend()
    plt.show()
    model.eval()
    y_pred = model(X_val)
    y_pred = y_pred.squeeze().cpu().detach().numpy()
    classification_report(y_val, y_pred, 1)
