from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        chat_template = [
            {
                "role": "system",
                "content": "You are a helpful assistant who solves problems step by step. Be concise and clearly state your final answer."
            },
            {
                "role": "user",
                "content": "Sally is given a bag of marbles with 4 red, 4 blue, and 8 yellow marbles. What is the probability that Sally draws a yellow marble?"
            },
            {
                "role": "assistant",
                "content": """
                    There are 4 + 4 + 8 = 16 total marbles.

                    The probability of drawing a yellow marble is:
                    8 / 16 = 1 / 2 = 50%

                    Answer: 50%
                """
            },
            {'role':'user','content':question}
        ]

        return self.tokenizer.apply_chat_template(
            chat_template,
            add_generation_prompt=True,
            tokenize=False
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
