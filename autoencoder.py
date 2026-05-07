# Import de PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import de TorchVision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import des outils
import matplotlib.pyplot as plt
from torchsummary import summary
from timeit import default_timer as timer
from my_helper_functions import print_train_time, plot_training_data, train

print(f"PyTorch version : {torch.__version__}")
print(f"TorchVision version : {torchvision.__version__}")

# Constantes
RANDOM_STATE = 42

BATCH_SIZE = 32

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

HIDDEN_DIM_1 = 150
HIDDEN_DIM_2 = 50
LATENT_SPACE_DIM = 2

# Obtenir les data de Fashion-MNIST
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor(), target_transform=None)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor(), target_transform=None)

class_names = [
 'T-shirt/top',
 'Pantalon',
 'Pull',
 'Robe',
 'Manteau',
 'Sandale',
 'Chemise',
 'Sneaker',
 'Sac',
 'Bottine'
]

# Créer les mini-batchs (dataloader)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Définir le device (de préférence usage du gpu: cuda)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Définition du modèle de l'AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_HEIGHT*IMAGE_WIDTH, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, LATENT_SPACE_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_SPACE_DIM, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, IMAGE_HEIGHT*IMAGE_WIDTH),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))
        )
    
    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat


autoEncoder = AutoEncoder().to(device)
next(autoEncoder.parameters()).device
summary(autoEncoder, input_size=(1, 28, 28))

# Définition de la loss et l'optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=autoEncoder.parameters(), lr=0.001)

# Entrainement du modèle
train_start = timer()

results = train(model=autoEncoder, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=20, device=device)

train_end = timer()
train_time_autoEncoder = print_train_time(start=train_start, end=train_end, device=device)

# Affichage de l'évolution de la loss
plot_training_data("Loss", results["train_loss"], results["test_loss"])

# Visualiser la reconstruction
# Mettre le modèle en mode éval
autoEncoder.eval()

# Prendre un batch
X, _ = next(iter(test_dataloader))
X = X.to(device)

# Désactiver le gradient
with torch.inference_mode():
    x_hat = autoEncoder(X)

# Repasser sur CPU pour affichage
X = X.cpu()
x_hat = x_hat.cpu()

# Nombre d'images à afficher
n = 5

plt.figure(figsize=(10, 4))

for i in range(n):
    # Image originale
    plt.subplot(2, n, i + 1)
    plt.imshow(X[i].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Image reconstruite
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_hat[i].squeeze(), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Visualiser l'espace latent
autoEncoder.eval()

latent_points = []
labels_list = []

with torch.inference_mode():
    for X, y in test_dataloader:
        X = X.to(device)

        z = autoEncoder.encoder(X)

        latent_points.append(z.cpu())
        labels_list.append(y.cpu())

latent_points = torch.cat(latent_points, dim=0)
labels_list = torch.cat(labels_list, dim=0)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    latent_points[:, 0],
    latent_points[:, 1],
    c=labels_list,
    cmap="tab10",
    alpha=0.6
)

plt.xlabel("z1")
plt.ylabel("z2")
plt.title("Espace latent de l'autoencoder")

# Remplacement des indices par les noms
cbar = plt.colorbar(scatter, ticks=range(len(class_names)))
cbar.ax.set_yticklabels(class_names)

plt.show()

# Génération à partir de l'espace latent

# 4 coordonnées choisies dans ton espace latent
z_selected = torch.tensor([
    [-4.5, -2.0],
    [ 3.0, 2.0],
    [ 0.5, 8.0],
    [ 2.5, -5.0],
], dtype=torch.float32)

# On visualise ces coordonnées dans l'espace latent
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    latent_points[:, 0],
    latent_points[:, 1],
    c=labels_list,
    cmap="tab10",
    alpha=0.6
)

plt.scatter(
    z_selected[:, 0],
    z_selected[:, 1],
    color="black",
    marker="x",
    s=120,
    label="Points générés"
)

for i, z in enumerate(z_selected):
    plt.text(z[0] + 0.2, z[1] + 0.2, f"z{i+1}", fontsize=11)

plt.xlabel("z1")
plt.ylabel("z2")
plt.title("Coordonnées sélectionnées dans l'espace latent")

cbar = plt.colorbar(scatter, ticks=range(len(class_names)))
cbar.ax.set_yticklabels(class_names)

plt.legend()
plt.show()

# On visualise les images générées
z_selected = z_selected.to(device)

with torch.inference_mode():
    generated = autoEncoder.decoder(z_selected)

generated = generated.cpu()
z_selected_cpu = z_selected.cpu()

plt.figure(figsize=(8, 3))

for i in range(len(generated)):
    plt.subplot(1, 4, i + 1)
    plt.imshow(generated[i].squeeze(), cmap="gray")
    plt.title(f"z{i+1}\n{z_selected_cpu[i].numpy()}")
    plt.axis("off")

plt.tight_layout()
plt.show()