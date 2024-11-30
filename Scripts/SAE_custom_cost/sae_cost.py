import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Global constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
input_size = 784

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
    "layer_sizes": [100],
    "dropout_rates": [0.3],
    "learning_rate": 1e-3,
    "num_epochs": 200,
    "early_stopping_patience": 8,
    "early_stopping_delta": 1e-4,
}

# Data preparation
transform = transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)])
dataset = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)

# Target embedding generation
def generate_hypersphere_targets(num_classes, bottleneck_size):
    random_vectors = np.random.randn(num_classes, bottleneck_size)
    targets = random_vectors / np.linalg.norm(random_vectors, axis=1, keepdims=True)
    return torch.tensor(targets, dtype=torch.float32).to(device)

def generate_diagonal_targets(num_classes, bottleneck_size):
    targets = np.zeros((num_classes, bottleneck_size))
    np.fill_diagonal(targets[:, :num_classes], 1)
    return torch.tensor(targets, dtype=torch.float32).to(device)

def generate_direct_targets(num_classes, bottleneck_size):
    targets = np.eye(num_classes, bottleneck_size)
    return torch.tensor(targets, dtype=torch.float32).to(device)

TARGET_GENERATORS = {
    "diagonal": (generate_diagonal_targets, 120),
    "hypersphere": (generate_hypersphere_targets, 120),
    "direct": (generate_direct_targets, 10),
}

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
def train_sae(sae_model, train_loader, val_loader, targets, params, lambda_reg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    optimizer = optim.AdamW(sae_model.parameters(), lr=params["learning_rate"], weight_decay=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.15, patience=3, threshold=1e-4
    )
    best_val_loss = float("inf")
    early_stop_counter = 0
    train_losses, val_losses = [], []

    start_time = time.time()

    for epoch in range(params["num_epochs"]):
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
        train_losses.append(train_loss)

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
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{params['num_epochs']} - Train Loss: {train_loss:.4f}, Penalty: {penalty:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - params["early_stopping_delta"]:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(sae_model.state_dict(), os.path.join(output_dir, "best_sae_model.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= params["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    total_time = time.time() - start_time
    np.save(os.path.join(output_dir, "train_losses.npy"), train_losses)
    np.save(os.path.join(output_dir, "val_losses.npy"), val_losses)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SAE Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    print(f"Training completed in {total_time:.2f} seconds.")
    return penalty

# SAE Classifier
class SAEClassifier(nn.Module):
    def __init__(self, encoder, encoder_output_size, classifier_hidden_sizes, num_classes, freeze_encoder=True, dropout_rate=0.5):
        super(SAEClassifier, self).__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        layers = []
        prev_size = encoder_output_size
        for hidden_size in classifier_hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

# Train and evaluate classifier
def train_and_evaluate_classifier(train_loader, val_loader, test_loader, params, classifier, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr=params["learning_rate"], weight_decay=0.3)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=1e-4)

    best_val_loss = float("inf")
    early_stop_counter = 0
    train_losses, val_losses = [], []

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
        train_losses.append(train_loss)

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
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{params['num_epochs']} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss - params["early_stopping_delta"]:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(classifier.state_dict(), os.path.join(output_dir, "best_classifier_model.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= params["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    # Evaluate on test set
    classifier.load_state_dict(torch.load(os.path.join(output_dir, "best_classifier_model.pth")))
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

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(num_classes)))
    cm_display.plot(cmap="Blues")
    plt.title(
        f"Confusion Matrix\nTest Accuracy: {test_acc:.4f} ({target_type}, λ={lambda_reg}, Batch={batch_size}, Freeze={freeze_encoder})")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    return test_acc

# Experiment loop with 3D visualization
def experiment(target_type, lambda_reg, batch_size, freeze_encoder):
    SAE_PARAMS["batch_size"] = batch_size

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    target_func, bottleneck_size = TARGET_GENERATORS[target_type]
    targets = target_func(num_classes, bottleneck_size)

    output_dir = f"results/{target_type}/lambda_{lambda_reg}_batch_{batch_size}_freeze_{freeze_encoder}"
    os.makedirs(output_dir, exist_ok=True)

    sae_model = SAE(input_size, bottleneck_size, SAE_PARAMS).to(device)
    final_penalty = train_sae(sae_model, train_loader, val_loader, targets, SAE_PARAMS, lambda_reg, output_dir)

    classifier = SAEClassifier(
        encoder=sae_model.encoder,
        encoder_output_size=bottleneck_size,
        classifier_hidden_sizes=MLP_PARAMS["layer_sizes"],
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        dropout_rate=MLP_PARAMS["dropout_rates"][0]
    ).to(device)

    test_accuracy = train_and_evaluate_classifier(train_loader, val_loader, test_loader, MLP_PARAMS, classifier, output_dir)

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Lambda: {lambda_reg}\n")
        f.write(f"Penalty: {final_penalty}\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")

    return lambda_reg, final_penalty, test_accuracy

TARGET_TYPES = ["hypersphere", "diagonal", "direct"]
LAMBDA_VALUES = [0.01, 0.06, 0.3, 1.5, 5]
BATCH_SIZES = [128]
FREEZE_ENCODER_OPTIONS = [True, False]

results = []
for target_type in TARGET_TYPES:
    for lambda_reg in LAMBDA_VALUES:
        for batch_size in BATCH_SIZES:
            for freeze_encoder in FREEZE_ENCODER_OPTIONS:
                metrics = experiment(target_type, lambda_reg, batch_size, freeze_encoder)
                results.append(metrics)

lambdas, penalties, accuracies = zip(*results)

def visualize_lambda_penalty_accuracy_3d(lambdas, penalties, accuracies):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(lambdas, penalties, accuracies, c=accuracies, cmap='viridis', s=50)
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Penalty')
    ax.set_zlabel('Test Accuracy')
    ax.set_title("Lambda vs Penalty vs Test Accuracy")
    plt.show()

visualize_lambda_penalty_accuracy_3d(lambdas, penalties, accuracies)
