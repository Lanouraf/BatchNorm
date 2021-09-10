from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import fit, plot_compare, plot_compare_3, plot_compare_accs
from models import LeNet, LeNetBN, LeNetStockBN


# Get dataset
# ---------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

train = datasets.MNIST(root = "dataset", train = True, transform = transform, download = True)
valid = datasets.MNIST(root = "dataset", train = False, transform = transform, download = True)

train_dataloader = DataLoader(train, batch_size = 256, shuffle = True)
valid_dataloader = DataLoader(valid, batch_size = 512, shuffle = False)


# Train models
# ---------------------------------------------
# Vanilla LeNet-5
all_losses_vanilla = []
all_acc_vanilla = []
for i in range(3):
    model_vanilla = LeNet()
    losses_vanilla, val_losses_vanilla, acc_vanilla = fit(model_vanilla, train_dataloader, valid_dataloader,
                                                          epochs=10, lr=1e-3)
    all_losses_vanilla.append((losses_vanilla, val_losses_vanilla))
    all_acc_vanilla.append(acc_vanilla)

# BatchNorm LeNet-5
all_losses_bn = []
all_acc_bn = []
for i in range(3):
    model_bn = LeNetBN()
    losses_bn, val_losses_bn, acc_bn = fit(model_bn, train_dataloader, valid_dataloader, epochs=10, lr=1e-3)
    all_losses_bn.append((losses_bn, val_losses_bn))
    all_acc_bn.append(acc_bn)

# BatchNorm LeNet-5 with high lr
all_losses_bn2 = []
for i in range(3):
    model_bn2 = LeNetBN()
    losses_bn2, val_losses_bn2, _ = fit(model_bn2, train_dataloader, valid_dataloader, epochs=10, lr=1e-2)
    all_losses_bn2.append((losses_bn2, val_losses_bn2))

# BatchNorm LeNet-5 with even higher lr
all_losses_bn3 = []
for i in range(3):
    model_bn3 = LeNetBN()
    losses_bn3, val_losses_bn3, _ = fit(model_bn3, train_dataloader, valid_dataloader, epochs=10, lr=5e-2)
    all_losses_bn3.append((losses_bn3, val_losses_bn3))

# Vanilla LeNet-5 with even higher lr
all_losses_vanilla2 = []
for i in range(3):
    model_vanilla2 = LeNet()
    losses_vanilla2, val_losses_vanilla2, _ = fit(model_vanilla2, train_dataloader, valid_dataloader, epochs=10, lr=5e-2)
    all_losses_vanilla2.append((losses_vanilla2, val_losses_vanilla2))

# BatchNorm LeNet-5 using Pytorch's BatchNorm
all_losses_stockbn = []
for i in range(3):
    model_stockbn = LeNetStockBN()
    losses_stockbn, val_losses_stockbn, _ = fit(model_stockbn, train_dataloader, valid_dataloader, epochs=10, lr=1e-3)
    all_losses_stockbn.append((losses_stockbn, val_losses_stockbn))


# Plot results
# ---------------------------------------------
plot_compare(all_losses_vanilla, all_losses_bn, "Vanilla", "BatchNorm", save_to="Vanilla_vs_BatchNorm")
plot_compare(all_losses_bn, all_losses_stockbn, "BatchNorm", "Pytorch BatchNorm", save_to="BatchNorm_vs_Pytorch_BatchNorm")
plot_compare_3(all_losses_vanilla, all_losses_bn, all_losses_bn2,
               "Vanilla", "Batchnorm", "BatchNorm 1e-2", save_to="Vanilla_vs_BatchNorm_vs-BatchNorm1e-2")
plot_compare(all_losses_vanilla2, all_losses_bn3, "Vanilla 5e-2", "BatchNorm 5e-2", save_to="Vanilla5e-2_vs_BatchNorm5e-2")
plot_compare_accs(all_acc_vanilla, all_acc_bn, "Vanilla", "BatchNorm", save_to="Accuracies")
