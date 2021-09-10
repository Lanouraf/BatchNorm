import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt


def fit(model, train_dataloader, valid_dataloader, epochs, loss_fn=nn.CrossEntropyLoss(), lr=1e-3, plot_each=20):
    """
    Simple training loop for a given model
    :param model: Model to be trained
    :param train_dataloader: Training data
    :param valid_dataloader: Validation data
    :param epochs: Number of epochs
    :param loss_fn: Loss function
    :param lr: Learning rate
    :param plot_each: Output progress each "plot_each" epochs
    :return: training losses (per mini-batch and epoch), validation losses (per epoch), accuracy (per epoch)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr = lr)
    valid_loss = 0  # Last epoch validation loss for display
    valid_losses = []
    losses = []
    last_accuracy = 0
    accuracies = []

    progbar = tqdm(range(epochs))
    for epoch in progbar:
        # Train one epoch
        losses_batch = []
        model.train()
        for i, batch in enumerate(train_dataloader):
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            losses_batch.append(loss.item())
            if i % plot_each == 0:
                progbar.set_postfix_str(
                    f'Batch: {i}/{len(train_dataloader)}  Loss: {loss.item()} Validation: {valid_loss} Accuracy: {last_accuracy}')
        losses.append(losses_batch)

        # Validation
        model.eval()
        accuracy_batch = 0
        with torch.no_grad():
            valid_loss = 0
            for batch in valid_dataloader:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss_batch = loss_fn(pred, yb)
                valid_loss += loss_batch.item()

                accuracy_batch += accuracy(pred, yb).item()
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
        last_accuracy = accuracy_batch / len(valid_dataloader)
        accuracies.append(last_accuracy)

    return losses, valid_losses, accuracies


def accuracy(preds, y):
    """
    Computes the accuracy of the network
    :param preds: Output predictions from the model
    :param y: Categorical labels (Ground truth)
    :return: Accuracy
    """
    preds = torch.argmax(preds, dim = 1)
    return sum(preds == y) / len(preds)


def plot_compare(all_losses_a, all_losses_b, legend_a="model a", legend_b="model b", save_to="out"):
    """
    Draws plots comparing models a and b from their training and validation losses

    :param all_losses_a: Losses for model a
    :param all_losses_b: Losses for model b
    :param legend_a: Name of the model a, to be shown in the plot's legend
    :param legend_b: Name of the model b, to be shown in the plot's legend
    :param save_to: If a file name is specified, plots are saved to SVG files
    """

    losses_a = [item[0] for item in all_losses_a]
    val_losses_a = [item[1] for item in all_losses_a]
    losses_b = [item[0] for item in all_losses_b]
    val_losses_b = [item[1] for item in all_losses_b]

    f = plt.figure()
    for j, (epoch_losses_a, epoch_losses_b) in enumerate(zip(losses_a, losses_b)):
        plt_a = plt.scatter(range(len(epoch_losses_a)), [losses[-1] for losses in epoch_losses_a], marker = 'o',
                            c = 'tab:blue', alpha = 0.3)
        plt.plot(range(len(epoch_losses_a)), [losses[-1] for losses in epoch_losses_a], '-', color = 'tab:blue')
        plt_b = plt.scatter(range(len(epoch_losses_b)), [losses[-1] for losses in epoch_losses_b], marker = 'o',
                            c = 'tab:orange', alpha = 0.3)
        plt.plot(range(len(epoch_losses_a)), [losses[-1] for losses in epoch_losses_b], '-', color = 'tab:orange')
    plt.title("Training losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend((plt_a, plt_b), (legend_a, legend_b))
    if save_to is not None:
        plt.savefig(save_to + '.svg')

    f = plt.figure()
    for j, (epoch_losses_a, epoch_losses_b) in enumerate(zip(val_losses_a, val_losses_b)):
        plt_a = plt.scatter(range(len(epoch_losses_a)), epoch_losses_a, marker = 'o', c = 'tab:blue', alpha = 0.3)
        plt.plot(range(len(epoch_losses_a)), epoch_losses_a, '-', color = 'tab:blue')
        plt_b = plt.scatter(range(len(epoch_losses_b)), epoch_losses_b, marker = 'o', c = 'tab:orange', alpha = 0.3)
        plt.plot(range(len(epoch_losses_b)), epoch_losses_b, '-', color = 'tab:orange')
    plt.title("Validation losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend((plt_a, plt_b), (legend_a + '-val', legend_b + '-val'))
    if save_to is not None:
        plt.savefig(save_to + '-val.svg')
    plt.show()


def plot_compare_3(all_losses_a, all_losses_b, all_losses_c, legend_a="model a", legend_b="model b",
                   legend_c="model c", save_to="out"):
    """
    Draws a plot comparing three models (a, b and c) from their training losses

    :param all_losses_a: Losses for model a
    :param all_losses_b: Losses for model b
    :param all_losses_c:  Losses for model c
    :param legend_a: Name of the model a, to be shown in the plot's legend
    :param legend_b: Name of the model b, to be shown in the plot's legend
    :param legend_c: Name of the model c, to be shown in the plot's legend
    :param save_to: If a file name is specified, plots are saved to SVG files
    """
    losses_a = [item[0] for item in all_losses_a]
    val_losses_a = [item[1] for item in all_losses_a]
    losses_b = [item[0] for item in all_losses_b]
    val_losses_b = [item[1] for item in all_losses_b]
    losses_c = [item[0] for item in all_losses_c]
    val_losses_c = [item[1] for item in all_losses_c]

    f = plt.figure()
    for j, (epoch_losses_a, epoch_losses_b, epoch_losses_c) in enumerate(zip(losses_a, losses_b, losses_c)):
        plt_a = plt.scatter(range(len(epoch_losses_a)), [losses[-1] for losses in epoch_losses_a], marker = 'o',
                            c = 'tab:blue', alpha = 0.3)
        plt.plot(range(len(epoch_losses_a)), [losses[-1] for losses in epoch_losses_a], '-', color = 'tab:blue')
        plt_b = plt.scatter(range(len(epoch_losses_b)), [losses[-1] for losses in epoch_losses_b], marker = 'o',
                            c = 'tab:orange', alpha = 0.3)
        plt.plot(range(len(epoch_losses_b)), [losses[-1] for losses in epoch_losses_b], '-', color = 'tab:orange')
        plt_c = plt.scatter(range(len(epoch_losses_c)), [losses[-1] for losses in epoch_losses_c], marker = '.',
                            c = 'tab:green', alpha = 0.3)
        plt.plot(range(len(epoch_losses_c)), [losses[-1] for losses in epoch_losses_c], '-', color = 'tab:green')
    plt.legend((plt_a, plt_b, plt_c), (legend_a, legend_b, legend_c))
    plt.title("Training losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if save_to is not None:
        plt.savefig(save_to + '.svg')

    f = plt.figure()
    for j, (epoch_losses_a, epoch_losses_b, epoch_losses_c) in enumerate(zip(val_losses_a, val_losses_b, val_losses_c)):
        plt_a = plt.scatter(range(len(epoch_losses_a)), epoch_losses_a, marker = 'o', c = 'tab:blue', alpha = 0.3)
        plt.plot(range(len(epoch_losses_a)), epoch_losses_a, '-', color = 'tab:blue')
        plt_b = plt.scatter(range(len(epoch_losses_b)), epoch_losses_b, marker = 'o', c = 'tab:orange', alpha = 0.3)
        plt.plot(range(len(epoch_losses_b)), epoch_losses_b, '-', color = 'tab:orange')
        plt_c = plt.scatter(range(len(epoch_losses_c)), epoch_losses_c, marker = '.', c = 'tab:green', alpha = 0.3)
        plt.plot(range(len(epoch_losses_c)), epoch_losses_c, '-', color = 'tab:green')
    plt.legend((plt_a, plt_b, plt_c), (legend_a + '-val', legend_b + '-val', legend_c + '-val'))
    plt.title("Validation losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if save_to is not None:
        plt.savefig(save_to + '-val.svg')
    plt.show()


def plot_compare_accs(all_accs_a, all_accs_b, legend_a="model a", legend_b="model b", save_to='out'):
    """
    Draws a plot comparing models from their accuracies

    :param all_accs_a: Accuracies for model a
    :param all_accs_b: Accuracies for model b
    :param legend_a: Name of the model a, to be shown in the plot's legend
    :param legend_b: Name of the model b, to be shown in the plot's legend
    :param save_to: If a file name is specified, plots are saved to SVG files
    """

    f = plt.figure()
    for j, (accs_a, accs_b) in enumerate(zip(all_accs_a, all_accs_b)):
        kwargs_a = {'label':legend_a} if j == 0 else {}
        kwargs_b = {'label':legend_b} if j == 0 else {}

        plt.plot(range(len(accs_a)), accs_a, '-', color = 'tab:blue', **kwargs_a)
        plt.plot(range(len(accs_b)), accs_b, '-', color = 'tab:orange', **kwargs_b)

    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if save_to is not None:
        plt.savefig(save_to + '.svg')
    plt.show()
