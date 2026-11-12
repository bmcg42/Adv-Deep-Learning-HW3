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
                "content": 
                    "You are a unit conversion assistant. "
                    "Convert the given quantity from its original unit to the requested unit. "
                    "Return the steps in your reasoning, the equation, and then reformat the result "
                    "by placing only the numeric value inside the <answer></answer> tags. "
                    "The answer is incorrect if not finally reported inside the <answer></answer> tags"                
            },
            {
                "role": "user",
                "content": "What is the equivalent of 144 inches in feet?"
            },
            {
                "role": "assistant",
                "content": """
                    "1. Original Units: inches "
                    "2. Target Units: Feet "
                    "3. Conversion factor: 1/12 feet per inch "
                    "4. Equation: 144 inches * 1/12 feet per inch = 12 feet "
                    "5. Answer: <answer>12</answer> "
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
