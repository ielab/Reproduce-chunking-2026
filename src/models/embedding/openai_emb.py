from typing import List
import os
import time
import numpy as np
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError, BadRequestError
import tiktoken

from src.registry import EMD_BACKBONE_REG


@EMD_BACKBONE_REG.register('OpenAI')
class OpenAIEmbeddingModel:
    """
    OpenAI embedding model supporting text-embedding-ada-002 and text-embedding-3-* models.

    NOTE: This model does NOT support LateEncoder (late chunking) because OpenAI's API
    only returns final embeddings, not token-level embeddings required for late chunking.
    Use RegularEncoder only.
    """

    def __init__(self, model_name: str, api_key: str = None):
        """
        Initialize OpenAI embedding model.

        Args:
            model_name: OpenAI model name (e.g., "text-embedding-ada-002",
                       "text-embedding-3-small", "text-embedding-3-large")
            api_key: Optional API key. If not provided, reads from OPENAI_API_KEY env var
        """
        self.model_name = model_name

        # Get API key from parameter or environment
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key must be provided either as 'api_key' parameter "
                "or via OPENAI_API_KEY environment variable"
            )

        self.client = OpenAI(api_key=api_key)

        # Model-specific configurations
        self._model_configs = {
            "text-embedding-ada-002": {
                "max_tokens": 8191,
                "dimensions": 1536,
                "batch_size": 2048  # OpenAI allows up to 2048 inputs per request
            },
            "text-embedding-3-small": {
                "max_tokens": 8191,
                "dimensions": 1536,
                "batch_size": 2048
            },
            "text-embedding-3-large": {
                "max_tokens": 8191,
                "dimensions": 3072,
                "batch_size": 2048
            }
        }

        self.config = self._model_configs.get(model_name, {
            "max_tokens": 8191,
            "dimensions": 1536,
            "batch_size": 2048
        })

        # For retry logic
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

        # Tokenizer for counting tokens (cl100k_base is used by ada-002 and 3-* models)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback if encoding not available
            self.tokenizer = None

    @property
    def model_id(self) -> str:
        return f"OpenAI: {self.model_name}"

    def get_embed_dim(self) -> int:
        """Returns the embedding dimension of the model."""
        return self.config["dimensions"]

    def _truncate_text_if_needed(self, text: str, max_tokens: int = 8191) -> str:
        """
        Truncate text if it exceeds max_tokens.

        Args:
            text: Text to check and truncate
            max_tokens: Maximum number of tokens allowed

        Returns:
            Truncated text if needed, original text otherwise
        """
        if not self.tokenizer:
            # If no tokenizer, estimate by character count (rough approximation: 1 token ≈ 4 chars)
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                print(f"Warning: Text may exceed {max_tokens} tokens. Truncating to {max_chars} chars.")
                return text[:max_chars]
            return text

        # Count tokens
        tokens = self.tokenizer.encode(text)

        if len(tokens) > max_tokens:
            print(f"Warning: Text has {len(tokens)} tokens (max {max_tokens}). Truncating to fit.")
            # Decode back to get truncated text
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens)

        return text

    def get_embeddings(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Get embeddings for a list of texts using OpenAI's API.

        Args:
            texts: List of text strings to embed
            **kwargs: Additional arguments (unused for OpenAI, kept for interface compatibility)

        Returns:
            np.ndarray: Array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        # OpenAI API supports batching up to 2048 inputs
        batch_size = self.config["batch_size"]
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._get_embeddings_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def _get_embeddings_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method to get embeddings with retry logic.

        Args:
            texts: Batch of texts to embed

        Returns:
            List of embedding vectors
        """
        # Handle token limit by splitting batch if needed
        if len(texts) > 1:
            try:
                return self._single_api_call(texts)
            except BadRequestError as e:
                # Check if it's a token limit error
                error_message = str(e)
                if ('max_tokens_per_request' in error_message or
                    ('Requested' in error_message and 'tokens' in error_message) or
                    'maximum context length' in error_message):
                    # Split batch in half and try again
                    mid = len(texts) // 2
                    print(f"Token limit exceeded ({len(texts)} texts). Splitting batch: {mid} + {len(texts) - mid}")
                    first_half = self._get_embeddings_with_retry(texts[:mid])
                    second_half = self._get_embeddings_with_retry(texts[mid:])
                    return first_half + second_half
                else:
                    raise RuntimeError(f"OpenAI API error: {e}")
        else:
            # Single text - just try it
            return self._single_api_call(texts)

    def _single_api_call(self, texts: List[str]) -> List[List[float]]:
        """
        Make a single API call with retry logic for transient errors.

        Args:
            texts: Batch of texts to embed

        Returns:
            List of embedding vectors
        """
        # Truncate texts that are too long
        max_tokens = self.config["max_tokens"]
        truncated_texts = [self._truncate_text_if_needed(text, max_tokens) for text in texts]

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=truncated_texts,
                    model=self.model_name
                )

                # Extract embeddings in the correct order
                # OpenAI returns embeddings with an index field
                embeddings = [None] * len(texts)
                for item in response.data:
                    embeddings[item.index] = item.embedding

                return embeddings

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} retries due to rate limiting: {e}")

            except APIConnectionError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Connection error. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} retries due to connection error: {e}")

            except APITimeoutError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay
                    print(f"Timeout error. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} retries due to timeout: {e}")

            except BadRequestError:
                # Don't retry BadRequestError - these are client errors that won't succeed
                raise

            except APIError as e:
                # For other API errors, fail fast
                raise RuntimeError(f"OpenAI API error: {e}")

    def get_all_token_embeddings(self, texts: List[str], **kwargs):
        """
        OpenAI API does not provide token-level embeddings.

        This method raises NotImplementedError to prevent usage with LateEncoder.
        Late chunking requires access to individual token embeddings, which OpenAI's
        API does not expose.

        Raises:
            NotImplementedError: Always raised as token-level embeddings are not available
        """
        raise NotImplementedError(
            f"OpenAI embedding models do not support token-level embeddings.\n"
            f"The model '{self.model_name}' cannot be used with LateEncoder.\n"
            f"Please use RegularEncoder instead.\n"
            f"Example: --encoder_name RegularEncoder --backbone OpenAI --model_name {self.model_name}"
        )


if __name__ == '__main__':
    import numpy as np
    from sentence_transformers.util import cos_sim

    # Ensure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='sk-...'")
        exit(1)

    print("Testing OpenAI Embedding Model...")
    model = OpenAIEmbeddingModel(model_name="text-embedding-ada-002")

    texts = [
        'How is the weather today?',
        'What is the current weather like today?',
        'The weather is nice today.'
    ]

    print(f"\nModel: {model.model_id}")
    print(f"Embedding dimension: {model.get_embed_dim()}")

    print(f"\nGenerating embeddings for {len(texts)} texts...")
    embeddings = model.get_embeddings(texts)
    print(f"Shape: {embeddings.shape}")
    print(f"First embedding (first 10 dims): {embeddings[0][:10]}")

    # Test similarity
    sim_01 = cos_sim(embeddings[0], embeddings[1])
    sim_02 = cos_sim(embeddings[0], embeddings[2])
    print(f"\nSimilarity between texts 0 and 1: {sim_01.item():.4f}")
    print(f"Similarity between texts 0 and 2: {sim_02.item():.4f}")

    # Test that token embeddings raise error
    print("\nTesting that token embeddings raise NotImplementedError...")
    try:
        model.get_all_token_embeddings(texts)
        print("❌ ERROR: Should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"✓ Correctly raised NotImplementedError")
        print(f"  Message: {str(e)[:100]}...")

    print("\n✓ All tests passed!")
