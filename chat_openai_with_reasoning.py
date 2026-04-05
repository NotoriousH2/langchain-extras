"""
ChatOpenAIWithReasoning: vLLM의 reasoning 필드를 지원하는 ChatOpenAI 확장

vllm은 Qwen3/3.5 등의 thinking 모델에서 reasoning 내용을 "reasoning" 필드로
분리 반환하지만, LangChain ChatOpenAI는 이 필드를 무시합니다.

이 클래스는 ChatOpenAI를 상속하여 invoke/batch/stream 모두에서
reasoning 내용을 AIMessage.additional_kwargs["reasoning"]으로 접근 가능하게 합니다.

사용법:
    from chat_openai_with_reasoning import ChatOpenAIWithReasoning

    # Reasoning ON (기본값)
    llm = ChatOpenAIWithReasoning(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
        model="./outputs/models/Qwen3.5-9B",
        max_tokens=4096,  # reasoning + answer 합산이므로 넉넉하게
    )

    # invoke
    resp = llm.invoke("질문")
    print(resp.additional_kwargs.get("reasoning"))  # thinking 과정
    print(resp.content)                              # 최종 답변

    # stream - reasoning 먼저, 이어서 content가 스트리밍됨
    for chunk in llm.stream("질문"):
        if chunk.additional_kwargs.get("reasoning"):
            print(chunk.additional_kwargs["reasoning"], end="")
        if chunk.content:
            print(chunk.content, end="")

    # Reasoning OFF
    llm_no_think = ChatOpenAIWithReasoning(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
        model="./outputs/models/Qwen3.5-9B",
        max_tokens=2048,
        enable_thinking=False,
    )
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class ChatOpenAIWithReasoning(ChatOpenAI):
    """vLLM의 reasoning 필드를 지원하는 ChatOpenAI 확장."""

    enable_thinking: bool = True

    @property
    def _default_params(self) -> dict[str, Any]:
        params = super()._default_params
        params.setdefault("extra_body", {})
        params["extra_body"]["chat_template_kwargs"] = {
            "enable_thinking": self.enable_thinking
        }
        return params

    @property
    def _llm_type(self) -> str:
        return "chat_openai_with_reasoning"

    # ── invoke / batch 경로 ──
    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """부모 메서드 호출 후, reasoning 필드를 additional_kwargs에 주입."""
        result = super()._create_chat_result(response, generation_info)

        # response에서 reasoning 추출
        import openai as _openai

        if isinstance(response, _openai.BaseModel):
            response_dict = response.model_dump()
        elif isinstance(response, dict):
            response_dict = response
        else:
            return result

        choices = response_dict.get("choices", [])
        for i, gen in enumerate(result.generations):
            if i < len(choices):
                msg_dict = choices[i].get("message", {})
                reasoning = msg_dict.get("reasoning") or msg_dict.get(
                    "reasoning_content"
                )
                if reasoning and isinstance(gen.message, AIMessage):
                    gen.message.additional_kwargs["reasoning"] = reasoning

        return result

    # ── stream 경로 ──
    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """부모 메서드 호출 후, delta에서 reasoning 필드를 추출."""
        gen_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if gen_chunk is None:
            return None

        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning and isinstance(gen_chunk.message, AIMessageChunk):
                gen_chunk.message.additional_kwargs["reasoning"] = reasoning

        return gen_chunk
