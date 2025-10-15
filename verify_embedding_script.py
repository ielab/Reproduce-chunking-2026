import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from numpy.linalg import norm
import numpy as np

# --- Common Setup ---
print("Setting up model and tokenizer for jina-embeddings-v3...")
model_name = 'jinaai/jina-embeddings-v3'

# --- Device Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load the model and move it to the specified device
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

sentences = ["How is the weather today?", "The weather today is sunny with a high of 75 degrees."]
cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

print("-" * 30)

# --- Method 1: Using adapter_mask for Query vs. Passage ---
print("Running Method 1: Using adapter_mask for Query vs. Passage...")

# Move inputs to the same device as the model
encoded_input_1 = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)

task_query = 'retrieval.query'
task_passage = 'retrieval.passage'
task_id_query = model._adaptation_map[task_query]
task_id_passage = model._adaptation_map[task_passage]

# Create a mask with different IDs for each sentence and move it to the device
adapter_mask = torch.tensor([task_id_query, task_id_passage], dtype=torch.int32).to(device)

with torch.no_grad():
    model_output_1 = model(**encoded_input_1, adapter_mask=adapter_mask)

embeddings_1 = mean_pooling(model_output_1, encoded_input_1["attention_mask"])
embeddings_1 = F.normalize(embeddings_1, p=2, dim=1)
# Move embeddings to CPU for numpy conversion
embeddings_1_np = embeddings_1.cpu().numpy()

similarity_1 = cos_sim(embeddings_1_np[0], embeddings_1_np[1])
print(f"Embeddings shape: {embeddings_1_np.shape}")
print(f"Query-Passage Cosine Similarity: {similarity_1}")
print("-" * 30)


# --- Method 2: Using prompt prefix for Query vs. Passage ---
print("Running Method 2: Using prompt prefix for Query vs. Passage...")

prompts = {
    "retrieval.query": "Represent the query for retrieving evidence documents: ",
    "retrieval.passage": "Represent the document for retrieval: ",
}
prefix_query = prompts["retrieval.query"]
prefix_passage = prompts["retrieval.passage"]

sentences_with_prefix = [prefix_query + sentences[0], prefix_passage + sentences[1]]

# Move inputs to the same device as the model
encoded_input_2 = tokenizer(sentences_with_prefix, padding=True, truncation=True, return_tensors="pt").to(device)

with torch.no_grad():
    model_output_2 = model(**encoded_input_2)

embeddings_2 = mean_pooling(model_output_2, encoded_input_2["attention_mask"])
embeddings_2 = F.normalize(embeddings_2, p=2, dim=1)
# Move embeddings to CPU for numpy conversion
embeddings_2_np = embeddings_2.cpu().numpy()

similarity_2 = cos_sim(embeddings_2_np[0], embeddings_2_np[1])
print(f"Embeddings shape: {embeddings_2_np.shape}")
print(f"Query-Passage Cosine Similarity: {similarity_2}")
print("-" * 30)


# --- Comparison ---
print("Comparing results...")
similarities_are_close = np.allclose(similarity_1, similarity_2, atol=1e-6)

print(f"Are the cosine similarities from both methods numerically close? {similarities_are_close}")

if similarities_are_close:
    print("\nConclusion: Yes, the two methods produce the same query-passage similarity.")
else:
    print("\nConclusion: No, the two methods produce different query-passage similarities.")