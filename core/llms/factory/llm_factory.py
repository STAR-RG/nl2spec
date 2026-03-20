from pathlib import Path 
from nl2spec.core.llms.llm_registry import LLMRegistry
from nl2spec.logging_utils import get_logger

log = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

registry = LLMRegistry(
    BASE_DIR /  "llms" / "config" / "information_llms.csv"
)

def create_llm(cfg: dict):

    provider = cfg["llm"]["provider"]
    model = cfg["llm"]["model"]
    # ----------------------------------------
    # MOCK PROVIDER (no registry dependency)
    # ----------------------------------------
    if provider == "mock":
        from nl2spec.core.llms.mock_llm import MockLLM
        return MockLLM()

    # ----------------------------------------
    # REAL PROVIDERS (need registry info)
    # ----------------------------------------

    info =   info = registry.get(provider, model)
    log.info("[LLM INIT] provider=%s | model=%s", provider, model)

    if provider == "gemini":
        from nl2spec.core.llms.gemini_llm import GeminiLLM
        return GeminiLLM(
            api_key=info["api_key"],
            model=info["model"],
        )

    if provider == "openAI":
        from nl2spec.core.llms.openai_llm import OpenAILLM
        return OpenAILLM(
            api_key=info["api_key"],
            model=info["model"],
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")