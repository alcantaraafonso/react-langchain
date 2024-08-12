from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

class AgentCallbackHandler(BaseCallbackHandler):

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> Any:
        """Runs when the agent is started"""
        print(f"***Prompt to LLM was: {prompts[0]}")
        print("*******")

    def on_llm_end(self, response: LLMResult, **kwargs) -> Any:
        print(f"***LLM Response: {response.generations[0][0].text}")
        print("*******")