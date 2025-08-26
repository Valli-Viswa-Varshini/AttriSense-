import os
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except Exception:
    genai = None

class ExplanationAgent:
    def __init__(self, llm_provider: str = "google"):
        load_dotenv()
        self.llm_provider = llm_provider
        self._configure_llm()

    def _configure_llm(self):
        if self.llm_provider == "google" and genai is not None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                except Exception:
                    pass

    def chat(self, query: str, context: dict) -> str:
        if self.llm_provider == "google" and genai is not None and os.getenv("GOOGLE_API_KEY"):
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"Answer the HR question: '{query}' based on this prediction context: {context}"
                return model.generate_content(prompt).text
            except Exception:
                pass
        return "LLM explanation is unavailable right now. Here's a simple answer: I used the prediction summary to provide HR-friendly insights."
