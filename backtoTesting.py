from datasets import load_dataset
from sentence_transformers import CrossEncoder
import numpy as np


class CrossEncoderRankingModel:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.trained = True  # Pretrained model doesn't need training

    def score(self, query, doc):
        return self.model.predict([(query, doc)])[0]

    def rank_documents(self, query, doc1, doc2):
        score1 = self.score(query, doc1)
        score2 = self.score(query, doc2)
        return score1, score2


def evaluate(instances, model):
    correct_pairs = 0
    total_pairs = len(instances)

    for inst in instances:
        q1, q2 = inst["q1"], inst["q2"]
        d1, d2 = inst["doc1"], inst["doc2"]

        s1_d1, s1_d2 = model.rank_documents(q1, d1, d2)
        s2_d1, s2_d2 = model.rank_documents(q2, d1, d2)

        if s1_d1 > s1_d2 and s2_d2 > s2_d1:
            correct_pairs += 1

    return (correct_pairs / total_pairs) * 100


def main():
    print("Loading da dataset...")
    dataset = load_dataset("orionweller/nevir")
    train_set = list(dataset["train"])
    # val_set = list(dataset["validation"])
    # test_set = list(dataset["test"])

   #  print(f"Loaded {len(train_set)} train, {len(val_set)} val, {len(test_set)} test instances.\n")

    model = CrossEncoderRankingModel()

    print("Evaluating model....be grateful for the results!")
    print(f"Train accuracy: {evaluate(train_set, model):.2f}%")
    # print(f"Validation accuracy: {evaluate(val_set, model):.2f}%")
   # print(f"Test accuracy: {evaluate(test_set, model):.2f}%")


if __name__ == "__main__":
    main()
