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
                "content": (
                    "Convert the given quantity into the requested target unit.\n\n"
                    "Always follow this exact three-step format:\n"
                    "1. Conversion Factor: State the necessary conversion factor(s).\n"
                    "2. Calculation: Show the arithmetic, including intermediate conversions if needed.\n"
                    "3. Answer: Put only the final numeric value inside <answer></answer> tags.\n\n"
                    "Rules:\n"
                    "- Use exactly three numbered steps.\n"
                    "- Include units in steps 1 and 2.\n"
                    "- Ensure units cancel correctly.\n"
                    "- Use multiple conversion factors when necessary.\n"
                    "- Follow standard conversion conventions.\n"
                    "- Do not round intermediate calculations.\n"
                    "- Do not explain concepts or include unnecessary text.\n"
                    "- Always include the answer tags."
                )
            },
            {
                "role": "user",
                "content": "What is the equivalent of 144 inches in feet?"
            },
            {
                "role": "assistant",
                "content": (
                    "1. Conversion Factor: 1 foot = 12 inches\n"
                    "2. Calculation: 144 / 12 = 12 feet\n"
                    "3. Answer: <answer>12</answer>"
                )
            },
            {
                "role": "user",
                "content": "Convert 10 hours to seconds."
            },
            {
                "role": "assistant",
                "content": (
                    "1. Conversion Factor: 1 hour = 3600 seconds\n"
                    "2. Calculation: 10 * 3600 = 36000 seconds\n"
                    "3. Answer: <answer>36000</answer>"
                )
            },
            {
                "role": "user",
                "content": "Convert 5 kilometers to miles."
            },
            {
                "role": "assistant",
                "content": (
                    "1. Conversion Factor: 1 kilometer = 0.621371 miles\n"
                    "2. Calculation: 5 * 0.621371 = 3.106855 miles\n"
                    "3. Answer: <answer>3.106855</answer>"
                )
            },
            {
                "role": "user",
                "content": "Convert 2.5 pounds to ounces."
            },
            {
                "role": "assistant",
                "content": (
                    "1. Conversion Factor: 1 pound = 16 ounces\n"
                    "2. Calculation: 2.5 * 16 = 40 ounces\n"
                    "3. Answer: <answer>40</answer>"
                )
            },
            {
                "role": "user",
                "content": "Express 4 centuries as weeks."
            },
            {
                "role": "assistant",
                "content": (
                    "1. Conversion Factors: 1 century = 100 years; "
                    "1 year = 365 days; 1 week = 7 days\n"
                    "2. Calculation: 4 * 100 * 365 / 7 = 20857.142857 weeks\n"
                    "3. Answer: <answer>20857.142857</answer>"
                )
            },
            {
                "role": "user",
                "content": "Convert 3.5 days to hours."
            },
            {
                "role": "assistant",
                "content": (
                    "1. Conversion Factor: 1 day = 24 hours\n"
                    "2. Calculation: 3.5 * 24 = 84 hours\n"
                    "3. Answer: <answer>84</answer>"
                )
            },
            {
                "role": "user",
                "content": question
            }
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
