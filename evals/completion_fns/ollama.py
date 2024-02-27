import requests
from typing import Optional

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import CompletionPrompt
from evals.record import record_sampling

class LangChainLLMCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]

class LangChainLLMCompletionFn(CompletionFn):
    def __init__(self, llm: str = "orca-mini", llm_kwargs: Optional[dict] = None, **kwargs) -> None:
        self.llm_url = "http://127.0.0.1:11434/api/chat"
        if llm_kwargs is None:
            self.llm_kwargs = {}
        else:
            self.llm_kwargs = llm_kwargs

    def __call__(self, prompt, **kwargs) -> LangChainLLMCompletionResult:
        formatted_prompt = CompletionPrompt(prompt).to_formatted_prompt()

        request_payload = {
            "prompt": formatted_prompt,
            **self.llm_kwargs
        }
        response = requests.post(self.llm_url, json=request_payload).text
        record_sampling(prompt=formatted_prompt, sampled=response)
        return LangChainLLMCompletionResult(response)


