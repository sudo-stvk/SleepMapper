import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BahdanauAttention(nn.Module):
    """
    Bahdanau-style attention mechanism for sequences.
    Calculates attention weights over a sequence of hidden states.
    """
    def __init__(self, hidden_size):
        """
        Initialize the attention mechanism.
        
        Args:
            hidden_size (int): Size of the hidden representation.
        """
        super(BahdanauAttention, self).__init__()
        try:
            self.W = nn.Linear(hidden_size, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)
        except Exception as e:
            print(f"Error initializing BahdanauAttention: {e}")
            raise e
        
    def forward(self, hidden_states):
        """
        Forward pass for the attention mechanism.
        
        Args:
            hidden_states (torch.Tensor): Outputs from BiLSTM (batch_size, time_steps, hidden_size)
            
        Returns:
            tuple: (context_vector, attention_weights)
                - context_vector (torch.Tensor): Weighted sum of hidden states (batch_size, hidden_size)
                - attention_weights (torch.Tensor): Attention weights (batch_size, time_steps, 1)
        """
        try:
            # score: (batch_size, time_steps, hidden_size) -> (batch_size, time_steps, 1)
            score = self.v(torch.tanh(self.W(hidden_states)))
            attention_weights = F.softmax(score, dim=1)
            context_vector = attention_weights * hidden_states
            context_vector = torch.sum(context_vector, dim=1)
            return context_vector, attention_weights
        except Exception as e:
            print(f"Error in attention forward pass: {e}")
            raise e

class SleepBiLSTM(nn.Module):
    """
    2-layer BiLSTM with Bahdanau attention for sleep apnea classification from MFCCs.
    """
    def __init__(self, input_size=120, hidden_size=256, num_layers=2, dropout=0.3):
        """
        Initialize the BiLSTM model.
        
        Args:
            input_size (int): Number of input features (default: 120 for MFCC + delta + delta-delta).
            hidden_size (int): Hidden state size for LSTM.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability between LSTM layers.
        """
        super(SleepBiLSTM, self).__init__()
        try:
            self.lstm = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True, 
                bidirectional=True, 
                dropout=dropout if num_layers > 1 else 0.0
            )
            # BiLSTM output size is hidden_size * 2 due to bidirectionality
            self.attention = BahdanauAttention(hidden_size * 2)
            self.fc = nn.Linear(hidden_size * 2, 1)
        except Exception as e:
            print(f"Error initializing SleepBiLSTM: {e}")
            raise e
        
    def forward(self, x):
        """
        Forward pass for the model.
        
        Args:
            x (torch.Tensor): MFCC features (batch_size, time_steps, input_size)
            
        Returns:
            tuple: (logit, attention_weights)
                - logit (torch.Tensor): Unnormalized output (batch_size, 1)
                - attention_weights (torch.Tensor): Attention weights (batch_size, time_steps, 1)
        """
        try:
            # x: (batch, time, input_size)
            lstm_out, _ = self.lstm(x) # lstm_out: (batch, time, hidden_size * 2)
            context_vector, attention_weights = self.attention(lstm_out)
            logit = self.fc(context_vector)
            return logit, attention_weights
        except Exception as e:
            print(f"Error in SleepBiLSTM forward pass: {e}")
            raise e

    def visualize_attention(self, attention_weights, save_path=None):
        """
        Visualize the attention weights over time steps.
        
        Args:
            attention_weights (torch.Tensor or np.ndarray): Attention weights for a single sample (time_steps, 1) or (batch_size, time_steps, 1).
            save_path (str, optional): Path to save the plot. If None, plt.show() is called.
        """
        try:
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
                
            # If batched, take the first sample in the batch
            if len(attention_weights.shape) == 3:
                attention_weights = attention_weights[0]
                
            attention_weights = attention_weights.squeeze()
            
            plt.figure(figsize=(10, 4))
            plt.plot(attention_weights)
            plt.title('Bahdanau Attention Weights over Time')
            plt.xlabel('Time Steps')
            plt.ylabel('Attention Weight')
            plt.grid(True)
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        except Exception as e:
            print(f"Error visualizing attention weights: {e}")
