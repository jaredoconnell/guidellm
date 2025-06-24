from pathlib import Path
from typing import Optional, Union, Any, AsyncGenerator

from PIL import Image

from guidellm.backend import StreamingTextResponse, ResponseSummary
from guidellm.backend.backend import Backend

try:
    from vllm import LLM, SamplingParams
    VLLM_ENABLED = True
except ImportError:
    LLM = None
    SamplingParams = None
    VLLM_ENABLED = False


# TODO: Handle vLLM not being present.


class VLLMPythonBackend(Backend):
    """
    Runs directly with Python.
    """

    def __init__(
        self,
        model: Optional[str] = None,
    ):
        super().__init__(type_="vllm_python")

        self._model = model

    @property
    def model(self) -> Optional[str]:
        return self._model

    @property
    def info(self) -> dict[str, Any]:
        """
        :return: The information about the backend.
        """
        # TODO: Populate vLLM info.
        return {}

    async def check_setup(self):
        """
        Checks whether the backend is setup correctly.
        Validates that VLLM is present and that the model requested is present.
        """
        if not VLLM_ENABLED:
            raise ValueError("VLLM not available; ensure the VLLM extra is enabled.")

    async def prepare_multiprocessing(self):
        # This may not apply to this backend.
        pass

    async def available_models(self) -> list[str]:

        pass

    async def text_completions(self, prompt: Union[str, list[str]], request_id: Optional[str] = None,
                               prompt_token_count: Optional[int] = None, output_token_count: Optional[int] = None,
                               **kwargs) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        pass

    async def chat_completions(self, content: Union[
        str,
        list[Union[str, dict[str, Union[str, dict[str, str]]], Path, Image.Image]],
        Any,
    ], request_id: Optional[str] = None, prompt_token_count: Optional[int] = None,
                               output_token_count: Optional[int] = None, raw_content: bool = False, **kwargs) -> \
    AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        pass

    @property
    def target(self) -> str:
        return "n/a"