import torch
import sys
from pathlib import Path

# Import the model class from the project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from GPTLanguageModel import GPTLanguageModel

class Model:
    """Simple model class for loading and using the GPT language model."""
    
    def __init__(self, model_path: str = None):
        """Initialize and load the model."""
        if model_path is None:
            # Default to model file in the same directory as this script
            model_path = Path(__file__).parent / "model_complete.pth"
        self.model_path = str(model_path)
        self.model = None
        self.stoi = None
        self.itos = None
        self.vocab_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model parameters
        self.vocab_size = checkpoint['vocab_size']
        self.stoi = checkpoint['stoi']
        self.itos = checkpoint['itos']
        
        # Create and load model
        self.model = GPTLanguageModel(vocab_size=self.vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, text: str) -> list:
        """Encode text to token indices."""
        return [self.stoi[char] for char in text]
    
    def decode(self, tokens: list) -> str:
        """Decode token indices to text."""
        return ''.join([self.itos[token] for token in tokens])
    
    def predict(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from a prompt."""
        # Encode prompt
        context = torch.tensor([self.encode(prompt)], dtype=torch.long, device=self.device)
        
        # Generate text
        with torch.no_grad():
            generated = self.model.generate(context, max_new_tokens=max_tokens)
        
        # Decode and return
        return self.decode(generated[0].tolist())
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def get_device(self) -> str:
        """Get device (cpu/cuda)."""
        return str(self.device)
