"""
Claude API client with retry logic and error handling.
"""
import anthropic
import time
from typing import Optional
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0


class ClaudeClient:
    """Anthropic Claude API client wrapper."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-6"

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """
        Call Claude API with retry logic.
        Returns the text response.
        """
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(MAX_RETRIES):
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": messages,
                }
                if system:
                    kwargs["system"] = system

                response = self.client.messages.create(**kwargs)
                return response.content[0].text

            except anthropic.RateLimitError as e:
                wait = RETRY_DELAY * (attempt + 1) * 2
                logger.warning(f"Rate limit hit. Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}")
                time.sleep(wait)

            except anthropic.APIError as e:
                logger.error(f"Claude API error on attempt {attempt+1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise

        raise RuntimeError("Max retries exceeded calling Claude API")

    def health_check(self) -> bool:
        """Test API connectivity."""
        try:
            response = self.complete("Say 'OK' in one word.", max_tokens=10)
            return "OK" in response or len(response) > 0
        except Exception as e:
            logger.error(f"Claude health check failed: {e}")
            return False
