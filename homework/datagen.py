from .cot import CoTModel
from .data import Dataset, is_answer_valid


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    import json
    model = CoTModel()
    train_data = Dataset('train')
    questions = [train_data[i][0] for i in range(len(train_data))]
    answers = [train_data[i][1] for i in range(len(train_data))]

    # Generate multiple answers per question
    grouped_answers = model.answer(
        *questions,
        num_return_sequences=oversample,
        temperature = temperature
    )

    # Rejection sampling
    good_examples = []
    for i in range(len(questions)):
        pred_answers = grouped_answers[i]
        correct = next(
            (gen for gen in pred_answers
             if is_answer_valid(gen, answers[i])),
            None
        )
        if correct is not None:
            good_examples.append([questions[i], answers[i], correct])

    # Save dataset
    with open(output_json, "w") as f:
        json.dump(good_examples, f, indent=2)

    correctness = len(good_examples) / len(questions)
    print(f"Success Rate:{correctness:.1%}")

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
