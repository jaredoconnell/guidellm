from pathlib import Path
from typing import Optional, Union, Any, AsyncGenerator

from PIL import Image

from guidellm.backend import StreamingTextResponse, ResponseSummary
from guidellm.backend.backend import Backend
from sympy.physics.units import time

try:
    from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs, PromptType, SamplingParams, RequestOutput
    VLLM_ENABLED = True
except ImportError:
    AsyncLLMEngine = None
    AsyncEngineArgs = None
    PromptType = None
    SamplingParams = None
    RequestOutput = None
    VLLM_ENABLED = False


# TODO: Handle vLLM not being present.


@Backend.register("vllm_python")
class VLLMPythonBackend(Backend):
    """
    Runs directly with Python.
    """

    llm_engine: AsyncLLMEngine

    def __init__(
        self,
        model: Optional[str] = None,
        # TODO: Generic keyword args?
    ):
        super().__init__(type_="vllm_python")

        self._model = model
        # TODO: Generic keyword args passed into AsyncEngineArgs
        # And find the proper vllm documentation for those args.
        # TODO: Start integrating settings documented here: https://docs.vllm.ai/en/latest/configuration/optimization.html
        args = AsyncEngineArgs(model=model)
        self.llm_engine = AsyncLLMEngine.from_engine_args(args)

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
        Checks whether the backend is set up correctly.
        Validates that VLLM is present and that the model requested is present.
        """
        if not VLLM_ENABLED:
            raise ValueError("VLLM not available; ensure the VLLM extra is enabled.")

    async def prepare_multiprocessing(self):
        # This may not apply to this backend.
        pass

    async def available_models(self) -> list[str]:
        # Use get_model_config
        pass

    async def text_completions(self, prompt: Union[str, list[str]], request_id: Optional[str] = None,
                               prompt_token_count: Optional[int] = None, output_token_count: Optional[int] = None,
                               **kwargs) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:

        params = SamplingParams(
            max_tokens=output_token_count,
            # Set min token count?
        )
        start_time = time.time()
        # TODO: What is prompt token count used for?
        # TODO: Determine how to make it do chat completions
        #   For chat, use a chat template. It's a jinja2 template that formats the input.
        #
        # TODO: https://docs.vllm.ai/en/v0.6.5/usage/multimodal_inputs.html
        #  (later, not first first iteration. )
        llm_generator = self.llm_engine.generate(prompt, params, request_id)
        return self._handle_llm_output(llm_generator, start_time, request_id)


    async def chat_completions(self, content: Union[
        str,
        list[Union[str, dict[str, Union[str, dict[str, str]]], Path, Image.Image]],
        Any,
    ], request_id: Optional[str] = None, prompt_token_count: Optional[int] = None,
                               output_token_count: Optional[int] = None, raw_content: bool = False, **kwargs) -> \
    AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        # TODO: Determine how to do chat completions
        # Start with text only completions.
        print("chat completions called; missing implementation")
        pass

    async def _handle_llm_output(
            self,
            generator: AsyncGenerator[RequestOutput, None],
            start_time: float,
            request_id: Optional[str],
    ) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        response_prompt_count: Optional[int] = None
        response_output_count: Optional[int] = None
        iter_count = 0
        iter_time = start_time
        first_iter_time: Optional[float] = None
        last_iter_time: Optional[float] = None

        yield StreamingTextResponse(
            type_="start",
            value="",
            start_time=start_time,
            first_iter_time=None,
            iter_count=iter_count,
            delta="",
            time=start_time,
            request_id=request_id,
        )

        async for output in generator:
            iter_count += 1
            if output.finished:
                print("Got finished output:", output)
                yield
            else:
                print("Got unfinished output:", output)
                yield StreamingTextResponse(
                    type_="iter",
                    value="TODO",  # TODO: Convert output.outputs
                    iter_count=iter_count,
                    start_time=start_time,
                    first_iter_time=first_iter_time,
                    delta="TODO",  # Does it give iterative changes?
                    time=iter_time,
                    request_id=request_id,
                )


    @property
    def target(self) -> str:
        return "n/a"
    # Consider v0 or v1 engine as return values.