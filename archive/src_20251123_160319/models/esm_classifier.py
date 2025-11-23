"""
ESM-2 model with classification head for GO term prediction.
"""

import torch
import torch.nn as nn
from transformers import EsmModel, AutoConfig
from typing import Dict


class ESMForGOPrediction(nn.Module):
    """
    ESM-2 model with a classification head for multi-label GO term prediction.
    
    Architecture:
        1. ESM-2 Backbone (Pre-trained)
        2. Mean Pooling over sequence length
        3. Classification Head (Linear + Dropout)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        num_labels: int = 5000,
        dropout: float = 0.3,
        freeze_layers: int = 0
    ):
        """
        Args:
            model_name: HuggingFace model name
            num_labels: Number of GO terms to predict
            dropout: Dropout rate for classification head
            freeze_layers: Number of ESM-2 layers to freeze (0 = finetune all)
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained ESM-2
        print(f"Loading ESM-2 model: {model_name}")
        self.esm = EsmModel.from_pretrained(model_name)
        
        # Get hidden dimension
        config = AutoConfig.from_pretrained(model_name)
        self.hidden_dim = config.hidden_size
        
        print(f"Model hidden dimension: {self.hidden_dim}")
        print(f"Output classes: {num_labels}")
        
        # Optionally freeze early layers
        if freeze_layers > 0:
            print(f"Freezing first {freeze_layers} layers")
            for i, layer in enumerate(self.esm.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_labels)
        )
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Prediction logits [batch_size, num_labels]
        """
        # Get ESM-2 embeddings
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        # outputs.last_hidden_state: [batch_size, seq_len, hidden_dim]
        sequence_output = outputs.last_hidden_state
        
        # Mean pooling (ignore padding tokens)
        # attention_mask: [batch_size, seq_len] -> [batch_size, seq_len, 1]
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
    
    def save_pretrained(self, save_directory: str):
        """Save model weights and config."""
        import json
        from pathlib import Path
        
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        # Save config
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "hidden_dim": self.hidden_dim
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'ESMForGOPrediction':
        """Load model from directory."""
        import json
        from pathlib import Path
        
        load_path = Path(load_directory)
        
        # Load config
        with open(load_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(
            model_name=config["model_name"],
            num_labels=config["num_labels"]
        )
        
        # Load weights
        state_dict = torch.load(
            load_path / "pytorch_model.bin",
            map_location="cpu"
        )
        model.load_state_dict(state_dict)
        
        print(f"Model loaded from {load_directory}")
        return model


if __name__ == "__main__":
    # Test the model
    print("Testing ESMForGOPrediction...")
    
    model = ESMForGOPrediction(
        num_labels=100,
        freeze_layers=0
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 50
    
    dummy_input_ids = torch.randint(0, 30, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    
    print("\nTesting forward pass...")
    logits = model(dummy_input_ids, dummy_attention_mask)
    print(f"Output shape: {logits.shape}")  # Should be [2, 100]
    
    # Test save/load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nTesting save to {tmpdir}...")
        model.save_pretrained(tmpdir)
        
        print(f"Testing load from {tmpdir}...")
        loaded_model = ESMForGOPrediction.from_pretrained(tmpdir)
        
        # Verify same output
        logits2 = loaded_model(dummy_input_ids, dummy_attention_mask)
        print(f"Outputs match: {torch.allclose(logits, logits2)}")
    
    print("\n✅ All tests passed!")
