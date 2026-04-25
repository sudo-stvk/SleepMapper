import torch
import os

def export_to_onnx(model, save_path, input_size=(1, 1, 224, 224)):
    """
    Exports a trained model to ONNX format with dynamic batch size.
    
    Args:
        model (nn.Module): The PyTorch model to export.
        save_path (str): The path to save the ONNX model.
        input_size (tuple): The expected input shape (batch_size, channels, height, width).
                            Defaults to (1, 1, 224, 224).
    """
    model.eval()
    
    # Create dummy input with the specified size
    dummy_input = torch.randn(*input_size)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Export with dynamic axes for batch size
    torch.onnx.export(
        model, 
        dummy_input, 
        save_path, 
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model successfully exported to ONNX format at: {save_path}")

def load_checkpoint(model, path):
    """
    Loads saved .pth checkpoint into the model.
    
    Args:
        model (nn.Module): The PyTorch model.
        path (str): The path to the saved checkpoint.
        
    Returns:
        nn.Module: The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
        
    # Load on CPU by default to avoid issues when transferring between devices
    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
    
    # Handle case where checkpoint contains more than just the state_dict (e.g., optimizer state)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    print(f"Successfully loaded checkpoint from: {path}")
    return model

def count_parameters(model):
    """
    Prints and returns trainable parameter count of the model.
    
    Args:
        model (nn.Module): The PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return trainable_params
