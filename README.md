# GPTLyricsGenerator 🎶

A character-level GPT language model trained on Drake’s lyrics dataset.  
Given a starting word or phrase, the model generates new lyric-like text in Drake’s style.  

---

## ✨ Features
- Transformer-based GPT architecture implemented **from scratch** in PyTorch  
- ~7.3M trainable parameters  
- Token + positional embeddings for sequence modeling  
- Multi-head self-attention and residual connections  
- Dropout regularization and AdamW optimizer  
- Training/validation **loss curve plots** for performance tracking  
- Generates text conditioned on user-provided prompts  

---

## 📊 Training Metrics
- **Dataset**: Drake lyrics (character-level)  
- **Final Train Loss**: ~1.0  
- **Final Validation Loss**: ~1.2  
- **Parameters**: ~7.3M  

Loss curve (Train vs Validation):

![Loss Curve](images/loss_curve.png)
