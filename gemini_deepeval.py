from typing import Optional

from deepeval.models import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiEvaluator(DeepEvalBaseLLM):
    def __init__(self, model: str = "gemini-3-flash-preview"):
        self.model_name = model
        self.model = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
        )

    def load_model(self):
        return self.model

    def _extract_content(self, content) -> str:
        """Normalize Gemini/LangChain content into the string DeepEval expects."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif "text" in item:
                        text_parts.append(str(item["text"]))
                    else:
                        text_parts.append(str(item))
                else:
                    text_parts.append(str(item))
            return "".join(text_parts)

        return str(content)

    def generate(self, prompt: str, schema: Optional[object] = None) -> str:
        response = self.model.invoke(prompt)
        return self._extract_content(response.content)

    async def a_generate(self, prompt: str, schema: Optional[object] = None) -> str:
        response = await self.model.ainvoke(prompt)
        return self._extract_content(response.content)

    def get_model_name(self):
        return self.model_name
