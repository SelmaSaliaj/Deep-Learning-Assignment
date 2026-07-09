"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer
import argparse
from pathlib import Path

def main():   

    parser = argparse.ArgumentParser(description='Train a model on medical imaging data')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--data', type=str, help='Dataset name (organs, chest, orgs, lesions, cells)')
    parser.add_argument('--model', type=str, help='Model architecture (AlexNet, VGG16, ResNet18)')
    parser.add_argument('--channels', type=int, help='Number of input channels (1 or 3)')
    parser.add_argument('--num_classes', type=int, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='../models', help='Directory to save models')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    if args.data:
        config["DATA"] = args.data
    if args.model:
        config["MODEL"] = args.model
    if args.channels:
        config["CHANNELS"] = args.channels
    if args.num_classes:
        config["NUM_CLASSES"] = args.num_classes
    if args.batch_size:
        config["BATCH_SIZE"] = args.batch_size
    if args.lr:
        config["LEARNING_RATE"] = args.lr
    if args.epochs:
        config["EPOCHS"] = args.epochs

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Models will be saved to: {save_path.absolute()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    train_loader, val_loader, _ = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

    model_class = getattr(models, config["MODEL"])
    model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"], drop_rate=0.3, activation_str="ReLu").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])

    model_filename = save_path / f"training_resuts_{config['DATA']}_{config['MODEL']}.pt"
    torch.save(model.state_dict(), model_filename)
    print(f"\nModel saved as {model_filename}")

if __name__ == "__main__":
    main()