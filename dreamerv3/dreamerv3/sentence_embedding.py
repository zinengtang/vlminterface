from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class SentenceEmbedder:
    """
    Wrapper for sentence-transformers/all-MiniLM-L6-v2 that computes
    mean‑pooled, L2‑normalized sentence embeddings.
    """
    def __init__(self,
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 device: str = None):
        """
        Args:
            model_name: HuggingFace model ID.
            device: torch device (e.g. 'cuda' or 'cpu'). If None, uses CUDA if available.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """
        Perform mean pooling, taking the attention mask into account.
        """
        token_embeddings = model_output[0]                           # (batch, seq_len, dim)
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask                            # (batch, dim)

    def encode(self, sentences, max_length: int = 128):
        """
        Tokenize and encode a list of sentences into embeddings.

        Args:
            sentences: List[str], the sentences to encode.
            max_length: Maximum token length for truncation/padding.

        Returns:
            torch.Tensor of shape (len(sentences), hidden_size), L2-normalized.
        """
        # 1) Tokenize
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)

        # 2) Forward pass
        with torch.no_grad():
            model_output = self.model(**encoded)

        # 3) Mean pooling
        pooled = self._mean_pooling(model_output, encoded['attention_mask'])

        # 4) L2 normalization
        embeddings = F.normalize(pooled, p=2, dim=1)

        input_ids = encoded['input_ids']
        B, L = input_ids.shape
        if L < max_length:
            # pad on the right with tokenizer.pad_token_id
            pad_id = self.tokenizer.pad_token_id or 0
            pad_tensor = torch.full((B, max_length - L),
                                     pad_id,
                                     dtype=input_ids.dtype,
                                     device=input_ids.device)
            input_ids = torch.cat([input_ids, pad_tensor], dim=1)
        elif L > max_length:
            # truncate excess tokens on the right
            input_ids = input_ids[:, :max_length]

        return embeddings, input_ids

# Example usage:
if __name__ == "__main__":
    embedder = SentenceEmbedder()
    examples = ["This is an example sentence", "Each sentence is converted"]
    embs = embedder.encode(examples)
    print("Embeddings shape:", embs.shape)  # should be (2, hidden_dim)
    print(embs)
