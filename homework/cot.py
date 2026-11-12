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
                    "Every response must have five steps. The original units, the target units, "
                    "the conversion factor, the conversion calculation, and the numeric answer "
                    "placed inside <answer></answer> tags. "
                    "The answer is incorrect if not reported inside the <answer></answer> tags "
                    "without units in step 5."                
            },
            {
                "role": "user",
                "content": "What is the equivalent of 144 inches in feet?"
            },
            {
                "role": "assistant",
                "content":
                    "1. Original Units: Inches \n"
                    "2. Target Units: Feet \n"
                    "3. Conversion Factor: 12 inches per foot \n"
                    "4. Calculation: 144 inches / 12 inches per foot = 12 feet \n"
                    "5. Answer: <answer>12</answer> "
            },
            {
                "role": "user",
                "content": "Convert 10 hours to seconds."
            },
            {
                "role": "assistant",
                "content": (
                    "1. Original Units: Hours\n"
                    "2. Target Units: Seconds\n"
                    "3. Conversion Factor: 3600 seconds per hour\n"
                    "4. Calculation: 10 * 3600 = 36000\n"
                    "5. Answer: <answer>36000</answer>"
                )
            },
            {
                "role": "user",
                "content": "Convert 10 hours to seconds."
            },
            {
                "role": "assistant",
                "content": (
                    "1. Original Units: Hours\n"
                    "2. Target Units: Seconds\n"
                    "3. Conversion Factor: 3600 seconds per hour\n"
                    "4. Calculation: 10 * 3600 = 36000\n"
                    "5. Answer: <answer>36000</answer>"
                )
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
