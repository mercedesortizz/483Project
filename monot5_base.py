
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
from tqdm import tqdm
import pandas as pd

# Load the MonoT5 model and tokenizer (base version to reduce RAM usage)
model_name = "castorini/monot5-base-msmarco-10k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the NevIR test dataset
dataset = load_dataset("orionweller/NevIR", split="test")

# Function to format input for MonoT5
def format_input(query, doc):
    return f"Query: {query} Document: {doc} Relevant:"

# Function to get "true" probability score
def get_score(query, doc):
    input_text = format_input(query, doc)
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            return_dict_in_generate=True,
            output_scores=True,
            max_length=2,  # Only one token: "true" or "false"
        )

    # Get logits of the first decoding step
    logits = outputs.scores[0][0]  # shape: [vocab_size]
    probs = torch.softmax(logits, dim=-1)

    true_token_id = tokenizer.encode("true", add_special_tokens=False)[0]
    return probs[true_token_id].item()

# Evaluate using pairwise accuracy
correct = 0
total = len(dataset) * 2  # Two comparisons per example

for example in tqdm(dataset):
    q1, q2 = example["q1"], example["q2"]
    d1, d2 = example["doc1"], example["doc2"]

    # Compare relevance of d1 vs d2 for q1
    score_q1_d1 = get_score(q1, d1)
    score_q1_d2 = get_score(q1, d2)
    if score_q1_d1 > score_q1_d2:
        correct += 1

    # Compare relevance of d2 vs d1 for q2
    score_q2_d1 = get_score(q2, d1)
    score_q2_d2 = get_score(q2, d2)
    if score_q2_d2 > score_q2_d1:
        correct += 1

# Compute pairwise accuracy
accuracy = (correct / total) * 100
print(f"Pairwise Accuracy on NevIR Test Set: {accuracy:.2f}%")


def manual_analysis(instances, num_examples=10):
    correct_examples = []
    wrong_examples = []

    for inst in instances:
        q1, q2 = inst["q1"], inst["q2"]
        d1, d2 = inst["doc1"], inst["doc2"]

        # Score doc relevance for q1
        s1_d1 = get_score(q1, d1)
        s1_d2 = get_score(q1, d2)

        # Score doc relevance for q2
        s2_d1 = get_score(q2, d1)
        s2_d2 = get_score(q2, d2)

        # Evaluate q1 comparison (doc1 should be more relevant)
        if s1_d1 > s1_d2:
            correct_examples.append({
                'query': q1,
                'doc1': d1,
                'doc2': d2,
                'expected': 'doc1',
                'predicted': 'doc1',
                'score_doc1': s1_d1,
                'score_doc2': s1_d2,
            })
        else:
            wrong_examples.append({
                'query': q1,
                'doc1': d1,
                'doc2': d2,
                'expected': 'doc1',
                'predicted': 'doc2',
                'score_doc1': s1_d1,
                'score_doc2': s1_d2,
            })

        # Evaluate q2 comparison (doc2 should be more relevant)
        if s2_d2 > s2_d1:
            correct_examples.append({
                'query': q2,
                'doc1': d1,
                'doc2': d2,
                'expected': 'doc2',
                'predicted': 'doc2',
                'score_doc1': s2_d1,
                'score_doc2': s2_d2,
            })
        else:
            wrong_examples.append({
                'query': q2,
                'doc1': d1,
                'doc2': d2,
                'expected': 'doc2',
                'predicted': 'doc1',
                'score_doc1': s2_d1,
                'score_doc2': s2_d2,
            })

    # Print samples
    print("\n--- Correctly classified examples ---\n")
    for ex in correct_examples[:num_examples]:
        print(f"Query: {ex['query']}")
        print(f"Doc1 (score={ex['score_doc1']:.4f}): {ex['doc1']}")
        print(f"Doc2 (score={ex['score_doc2']:.4f}): {ex['doc2']}")
        print(f"Expected: {ex['expected']} | Predicted: {ex['predicted']}")
        print("-" * 50)

    print("\n--- Incorrectly classified examples ---\n")
    for ex in wrong_examples[:num_examples]:
        print(f"Query: {ex['query']}")
        print(f"Doc1 (score={ex['score_doc1']:.4f}): {ex['doc1']}")
        print(f"Doc2 (score={ex['score_doc2']:.4f}): {ex['doc2']}")
        print(f"Expected: {ex['expected']} | Predicted: {ex['predicted']}")
        print("-" * 50)

# Run manual analysis on a subset
manual_analysis(dataset.select(range(10)), num_examples=5)
