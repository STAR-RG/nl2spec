from openai import OpenAI
from nl2spec.core.llms.base import BaseLLM


class OpenAILLM(BaseLLM):

    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str)-> tuple[str, object]:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip(), response.usage

    def close(self):
        # não é obrigatório, mas mantemos padrão
        pass