import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2Apnea(nn.Module):
    """
    Wav2Vec2 model for binary sleep apnea classification from raw audio waveforms.
    """
    def __init__(self, model_name="facebook/wav2vec2-base"):
        """
        Initializes the Wav2Vec2 model with a custom classification head.
        
        Args:
            model_name (str): The HuggingFace model name to load.
        """
        super(Wav2Vec2Apnea, self).__init__()
        
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model '{model_name}': {e}")
            
        self.classification_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Apply freezing constraints
        self.freeze_feature_extractor()
        self._freeze_bottom_transformer_layers()

    def freeze_feature_extractor(self):
        """
        Freezes the feature extractor layers completely to prevent them from being updated during training.
        """
        # Freeze the CNN feature extractor
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
            
        # Freeze feature projection if it exists
        if hasattr(self.wav2vec2, 'feature_projection'):
            for param in self.wav2vec2.feature_projection.parameters():
                param.requires_grad = False

    def _freeze_bottom_transformer_layers(self):
        """
        Freezes the bottom 8 transformer layers and leaves the top 4 trainable.
        Wav2Vec2-base has 12 transformer layers (0 to 11).
        """
        # Freeze bottom 8 layers (indices 0 to 7)
        for i in range(8):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = False
        
        # Ensure top 4 layers (indices 8 to 11) are trainable
        for i in range(8, 12):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = True

    def count_trainable_params(self):
        """
        Counts the number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        """
        Forward pass for the model.
        
        Args:
            x (torch.Tensor): Raw 16kHz waveform tensor of shape (batch_size, 480000).
            
        Returns:
            torch.Tensor: Single logit for binary classification, shape (batch_size, 1).
        """
        # Wav2Vec2 expects input of shape (batch_size, sequence_length)
        outputs = self.wav2vec2(x)
        
        # Take the mean over the sequence length for the pooled output (global average pooling)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        
        logits = self.classification_head(pooled_output)
        return logits
