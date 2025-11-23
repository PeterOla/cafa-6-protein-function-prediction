"""
Fine-tuning script for ESM-2 on GO term prediction.

This script:
1. Loads the dataset with tokenization
2. Fine-tunes the ESM-2 model (not just the classifier head)
3. Uses gradient accumulation for memory efficiency
4. Implements early stopping based on validation F1
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import json

from src.data.finetune_dataset import create_datasets
from src.models.esm_classifier import ESMForGOPrediction
from src.training.loss import AsymmetricLoss


class FineTuner:
    """Handles the fine-tuning process."""
    
    def __init__(
        self,
        model: ESMForGOPrediction,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 2e-5,
        num_epochs: int = 5,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        eval_threshold: float = 0.5,
        patience: int = 3,
        save_dir: str = "models/esm_finetuned",
        use_asymmetric_loss: bool = False,
        gamma_neg: float = 2.0,
        gamma_pos: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_threshold = eval_threshold
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        if use_asymmetric_loss:
            self.criterion = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=0.05)
            print(f"  Using AsymmetricLoss (gamma_neg={gamma_neg}, gamma_pos={gamma_pos})")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            print(f"  Using BCEWithLogitsLoss")
        
        # Tracking
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": []
        }
        
        print(f"Fine-tuner initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Total training steps: {total_steps}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc="Training")
        for i, (inputs, labels) in enumerate(pbar):
            # Move to device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Normalize loss by accumulation steps
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # Update weights after accumulation
            if (i + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on validation set with adaptive thresholding."""
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc="Evaluating")
        for inputs, labels in pbar:
            # Move to device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            labels_dev = labels.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels_dev)
            
            total_loss += loss.item()
            
            # Store probabilities (not binary predictions)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
        
        # Concatenate all batches
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        
        # Calculate metrics
        val_loss = total_loss / len(self.val_loader)
        
        # Try multiple thresholds and pick the best
        thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        best_f1 = 0.0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            preds = (all_probs > threshold).astype(int)
            f1 = f1_score(all_labels, preds, average='samples', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    "f1": f1,
                    "precision": precision_score(all_labels, preds, average='samples', zero_division=0),
                    "recall": recall_score(all_labels, preds, average='samples', zero_division=0),
                    "threshold": threshold
                }
        
        # Update the instance threshold for next evaluation
        self.eval_threshold = best_threshold
        
        return {
            "val_loss": val_loss,
            **best_metrics
        }
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("Starting Fine-Tuning")
        print("="*50)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            
            # Evaluate
            metrics = self.evaluate()
            self.history["val_loss"].append(metrics["val_loss"])
            self.history["val_f1"].append(metrics["f1"])
            self.history["val_precision"].append(metrics["precision"])
            self.history["val_recall"].append(metrics["recall"])
            
            # Print metrics
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {metrics['val_loss']:.4f}")
            print(f"  Val F1: {metrics['f1']:.4f} (threshold: {metrics['threshold']:.2f})")
            print(f"  Val Precision: {metrics['precision']:.4f}")
            print(f"  Val Recall: {metrics['recall']:.4f}")
            
            # Check for improvement
            if metrics['f1'] > self.best_f1:
                self.best_f1 = metrics['f1']
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                
                # Save best model
                save_path = self.save_dir / "best_model"
                self.model.save_pretrained(str(save_path))
                print(f"  ✅ New best F1! Model saved to {save_path}")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save training history
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "="*50)
        print("Fine-Tuning Complete!")
        print(f"Best F1: {self.best_f1:.4f} (Epoch {self.best_epoch})")
        print("="*50)


def main():
    # Hyperparameters
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION = 4  # Effective batch size = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    VOCAB_SIZE = 5000
    MIN_COUNT = 10
    MAX_SEQ_LENGTH = 512  # Shorter sequences to save memory
    PATIENCE = 3
    
    # Loss function settings
    USE_ASYMMETRIC_LOSS = True  # Use Asymmetric Loss instead of BCE
    GAMMA_NEG = 2.0  # Down-weight easy negatives (lower = more aggressive)
    GAMMA_POS = 1.0  # Focus on hard positives
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    FASTA_PATH = BASE_DIR / "Train/train_sequences.fasta"
    LABELS_PATH = BASE_DIR / "Train/train_terms.tsv"
    SAVE_DIR = BASE_DIR / "models/esm_finetuned"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset, val_dataset, vocab = create_datasets(
        FASTA_PATH,
        LABELS_PATH,
        vocab_size=VOCAB_SIZE,
        min_count=MIN_COUNT,
        val_split=0.2
    )
    
    # Save vocabulary
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVE_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary saved to {SAVE_DIR / 'vocab.json'}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nInitializing model...")
    model = ESMForGOPrediction(
        num_labels=VOCAB_SIZE,
        dropout=0.3,
        freeze_layers=0  # Fine-tune all layers
    )
    
    # Create trainer
    trainer = FineTuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        patience=PATIENCE,
        save_dir=str(SAVE_DIR),
        use_asymmetric_loss=USE_ASYMMETRIC_LOSS,
        gamma_neg=GAMMA_NEG,
        gamma_pos=GAMMA_POS
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
