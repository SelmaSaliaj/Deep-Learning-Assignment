import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from data import get_loaders
import models
from pathlib import Path

BEST_MODELS = [
    {
        "name": "Orgs + AlexNet",
        "path": "../models/training_resuts_orgs_AlexNet.pt",
        "dataset": "orgs",
        "model": "AlexNet",
        "channels": 1,
        "num_classes": 11
    },
    {
        "name": "Chest + AlexNet",
        "path": "../models/training_resuts_chest_AlexNet.pt",
        "dataset": "chest",
        "model": "AlexNet",
        "channels": 1,
        "num_classes": 2
    },
    {
        "name": "Organs (Transfer) + AlexNet",
        "path": "../models/transfer_orgs_to_organs_best.pt",
        "dataset": "organs",
        "model": "AlexNet",
        "channels": 1,
        "num_classes": 11
    },
    {
        "name": "Lesions + AlexNet",
        "path": "../models/training_resuts_lesions_AlexNet.pt",
        "dataset": "lesions",
        "model": "AlexNet",
        "channels": 3,
        "num_classes": 7
    },
    {
        "name": "Cells + AlexNet",
        "path": "../models/training_resuts_cells_AlexNet.pt",
        "dataset": "cells",
        "model": "AlexNet",
        "channels": 3,
        "num_classes": 8
    }
]

def evaluate_model(config):
    """Evaluate a single model and return metrics"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {config['name']}")
    print(f"{'='*60}")
    
    if not Path(config['path']).exists():
        print(f"\nModel file not found: {config['path']}")
        print(f"\nPlease train this model first.")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = getattr(models, config['model'])
    model = model_class(in_channels=config['channels'], num_classes=config['num_classes'], drop_rate=0.3, activation_str="ReLu").to(device)
    
    model.load_state_dict(torch.load(config['path'], map_location=device))
    model.eval()
    print(f"\nModel loaded from: {config['path']}")
    
    _, _, test_loader = get_loaders(data=config['dataset'], data_path="../data", batch_size=16)
    print(f"\nTest data loaded: {config['dataset']}")
    
    all_preds = []
    all_labels = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.flatten()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='macro') * 100
    recall = recall_score(all_labels, all_preds, average='macro') * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    avg_loss = running_loss / len(all_labels)
    
    print(f"\nResults for {config['name']}:")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print(f"  F1-Score:  {f1:.2f}%")
    print(f"  Loss:      {avg_loss:.4f}")
    
    return {
        'name': config['name'],
        'dataset': config['dataset'],
        'model': config['model'],
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': avg_loss
    }

def main():
    print("=" * 60)
    print("BENCHMARK EVALUATION - BEST MODELS")
    print("=" * 60)
    
    results = []
    
    for config in BEST_MODELS:
        result = evaluate_model(config)
        if result:
            results.append(result)
    
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Dataset':<15} {'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 78)
    
    for r in results:
        print(f"{r['dataset']:<15} {r['model']:<15} {r['accuracy']:<12.2f} {r['precision']:<12.2f} {r['recall']:<12.2f} {r['f1']:<12.2f}")
    
    print("=" * 60)
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()