import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
SAE_PARAMS = {
    "layer_sizes": [800, 200],
    "activation_functions": [nn.LeakyReLU(0.3), nn.LeakyReLU(0.3)],
    "learning_rate": 1e-3,
    "batch_size": 128,
    "num_epochs": 200,
    "early_stopping_patience": 7,
    "early_stopping_delta": 1e-4,
}

MLP_PARAMS = {
    "layer_sizes": [64, 32],
    "dropout_rates": [0.5, 0.5],
    "learning_rate": 1e-4,
    "batch_size": 128,
    "num_epochs": 200,
    "early_stopping_patience": 8,
    "early_stopping_delta": 1e-4,
}

num_classes = 10
input_size = 784

# Data preparation
transform = transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)])  # Flatten images
dataset = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def get_data_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = get_data_loader(train_dataset, SAE_PARAMS["batch_size"])
val_loader = get_data_loader(val_dataset, SAE_PARAMS["batch_size"], shuffle=False)
test_loader = get_data_loader(test_dataset, SAE_PARAMS["batch_size"], shuffle=False)

# Generate targets
def generate_hypersphere_targets(num_classes, dimensions):
    random_vectors = np.random.randn(num_classes, dimensions)
    targets = random_vectors / np.linalg.norm(random_vectors, axis=1, keepdims=True)
    return torch.tensor(targets, dtype=torch.float32).to(device)

def generate_direct_targets(num_classes, dimensions):
    targets = np.eye(num_classes, dimensions)
    return torch.tensor(targets, dtype=torch.float32).to(device)

hypersphere_targets = generate_hypersphere_targets(num_classes, 120)
direct_targets = generate_direct_targets(num_classes, 10)

# SAE model
class SAE(nn.Module):
    def __init__(self, input_size, bottleneck_size, params):
        super(SAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, params["layer_sizes"][0]),
            params["activation_functions"][0],
            nn.Linear(params["layer_sizes"][0], params["layer_sizes"][1]),
            params["activation_functions"][1],
            nn.Linear(params["layer_sizes"][1], bottleneck_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, params["layer_sizes"][1]),
            params["activation_functions"][1],
            nn.Linear(params["layer_sizes"][1], params["layer_sizes"][0]),
            params["activation_functions"][0],
            nn.Linear(params["layer_sizes"][0], input_size)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed

# SAE training
def train_sae(sae_model, train_loader, val_loader, targets, params, lambda_reg):
    optimizer = optim.Adam(sae_model.parameters(), lr=params["learning_rate"])
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(params["num_epochs"]):
        # Training
        sae_model.train()
        train_loss, penalty = 0, 0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            latent, reconstructed = sae_model(batch_data)
            mse_loss = nn.MSELoss()(reconstructed, batch_data)
            reg_penalty = torch.mean(torch.norm(latent - targets[batch_labels], dim=1))
            loss = mse_loss + lambda_reg * reg_penalty
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            penalty += reg_penalty.item()

        train_loss /= len(train_loader)
        penalty /= len(train_loader)

        # Validation
        sae_model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                latent, reconstructed = sae_model(batch_data)
                mse_loss = nn.MSELoss()(reconstructed, batch_data)
                reg_penalty = torch.mean(torch.norm(latent - targets[batch_labels], dim=1))
                loss = mse_loss + lambda_reg * reg_penalty

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{params['num_epochs']} - Train Loss: {train_loss:.4f}, Penalty: {penalty:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss - params["early_stopping_delta"]:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(sae_model.state_dict(), "best_sae_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= params["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

# Extract latent representations
def extract_latent_representations(sae_model, data_loader):
    sae_model.eval()
    latent_representations, labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            latent, _ = sae_model(batch_data)
            latent_representations.append(latent.cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(latent_representations), torch.cat(labels)

class SAEClassifier(nn.Module):
    def __init__(self, encoder, input_size, encoder_output_size, classifier_hidden_sizes, num_classes, leaky_relu_negative_slope=0.01, batch_norm=True):
        super(SAEClassifier, self).__init__()
        self.encoder = encoder  # Pre-trained encoder, frozen
        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze encoder weights
        layers = []
        prev_size = encoder_output_size
        for hidden_size in classifier_hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=leaky_relu_negative_slope))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)  # Pass through the encoder to create latent space
        x = self.classifier(x)  # Pass latent space through the classifier head
        return x

def train_and_evaluate_classifier(train_loader, val_loader, test_loader, params, encoder, input_size, encoder_output_size):
    classifier = SAEClassifier(
        encoder=encoder,
        input_size=input_size,
        encoder_output_size=encoder_output_size,
        classifier_hidden_sizes=params["layer_sizes"],
        num_classes=num_classes,
        leaky_relu_negative_slope=0.01,
        batch_norm=True
    ).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(params["num_epochs"]):
        classifier.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(batch_labels).sum().item()
            train_total += batch_labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        classifier.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = classifier(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(batch_labels).sum().item()
                val_total += batch_labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{params['num_epochs']} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss - params["early_stopping_delta"]:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(classifier.state_dict(), "best_classifier_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= params["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    classifier.load_state_dict(torch.load("best_classifier_model.pth"))

    classifier.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = classifier(batch_data)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(batch_labels).sum().item()
            test_total += batch_labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(cm, display_labels=list(range(num_classes))).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


print("Training SAE with Hypersphere Targets...")
sae_model_hyper = SAE(input_size, 120, SAE_PARAMS).to(device)
train_sae(sae_model_hyper, train_loader, val_loader, hypersphere_targets, SAE_PARAMS, lambda_reg=0.06)

latent_train_hyper, labels_train_hyper = extract_latent_representations(sae_model_hyper, train_loader)
latent_val_hyper, labels_val_hyper = extract_latent_representations(sae_model_hyper, val_loader)
latent_test_hyper, labels_test_hyper = extract_latent_representations(sae_model_hyper, test_loader)

print("Training and Evaluating SAEClassifier on Hypersphere Latent Representations...")
train_and_evaluate_classifier(
    train_loader, val_loader, test_loader, MLP_PARAMS, sae_model_hyper.encoder, input_size, 120
)

print("Training SAE with Direct Targets...")
sae_model_direct = SAE(input_size, 10, SAE_PARAMS).to(device)
train_sae(sae_model_direct, train_loader, val_loader, direct_targets, SAE_PARAMS, lambda_reg=0.06)

latent_train_direct, labels_train_direct = extract_latent_representations(sae_model_direct, train_loader)
latent_val_direct, labels_val_direct = extract_latent_representations(sae_model_direct, val_loader)
latent_test_direct, labels_test_direct = extract_latent_representations(sae_model_direct, test_loader)

print("Training and Evaluating SAEClassifier on Direct Latent Representations...")
train_and_evaluate_classifier(
    train_loader, val_loader, test_loader, MLP_PARAMS, sae_model_direct.encoder, input_size, 10
)
