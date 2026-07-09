import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Source dataset (chest or orgs)')
    parser.add_argument('--target', type=str, default='organs', help='Target dataset')
    parser.add_argument('--channels', type=int, default=1, help='Input channels')
    parser.add_argument('--target_classes', type=int, default=11, help='Target classes')
    parser.add_argument('--epochs', type=int, default=10, help='Fine-tune epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate (low for fine-tuning)')
    parser.add_argument('--model_dir', type=str, default='../models', help='Directory containing saved models')
    parser.add_argument('--save_dir', type=str, default='../models', help='Directory to save transfer model')
    args = parser.parse_args()

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")
    print(f"Source: {args.source} → Target: {args.target}")
    
    train_loader, val_loader, _ = get_loaders(data=args.target, data_path="../data", batch_size=16)
    
    model = models.AlexNet(in_channels=args.channels, num_classes=args.target_classes, drop_rate=0.3, activation_str="ReLu").to(device)
    
    # Load pre-trained weights from models/ folder
    source_file = Path(args.model_dir) / f"training_resuts_{args.source}_AlexNet.pt"
    print(f"Loading pre-trained weights from: {source_file}")
    
    try:
        pretrained_dict = torch.load(source_file, map_location=device)
        model_dict = model.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'classifier.6' not in k}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"\nLoaded weights from {source_file}")
    except FileNotFoundError:
        print(f"\nFile {source_file} not found! Train the source model first.")
        print("Run: python train.py --data chest --model AlexNet --channels 1 --num_classes 2 --epochs 5")
        return
    
    print("Freezing early layers...")
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
        else:
            print(f"\nTraining layer: {name}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    
    transfer_file = save_path / f"transfer_{args.source}_to_{args.target}.pt"
    torch.save(model.state_dict(), transfer_file)
    print(f"\n Transfer model saved as {transfer_file}")

if __name__ == "__main__":
    main()