"""
ThinkingChatHuggingFace - ChatHuggingFace에 thinking 제어 기능 추가

Usage:
    from thinking_chat import ThinkingChatHuggingFace

    # thinking ON + 사고과정 자동 제거
    llm = ThinkingChatHuggingFace(
        llm=base_llm, tokenizer=tokenizer,
        enable_thinking=True, strip_thinking=True,
    )

    # thinking ON + 사고과정 포함 (디버깅용)
    llm = ThinkingChatHuggingFace(
        llm=base_llm, tokenizer=tokenizer,
        enable_thinking=True, strip_thinking=False,
    )

    # thinking OFF
    llm = ThinkingChatHuggingFace(
        llm=base_llm, tokenizer=tokenizer,
        enable_thinking=False,
    )

Note:
    thinking 제거는 tokenizer의 chat template 내장 strip_thinking 로직을 활용하므로
    모델에 무관하게 동작합니다. (Gemma 4, Qwen3 등 thinking을 지원하는 모델)
"""

from collections.abc import Iterator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)
from langchain_huggingface import ChatHuggingFace


class ThinkingChatHuggingFace(ChatHuggingFace):
    """enable_thinking / strip_thinking을 제어할 수 있는 ChatHuggingFace."""

    enable_thinking: bool = False
    strip_thinking: bool = True

    def _to_chat_prompt(self, messages):
        """입력: enable_thinking 파라미터를 chat template에 전달."""
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")
        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")
        messages_dicts = [self._to_chatml_format(m) for m in messages]
        return self.tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def _strip_thinking_content(self, text: str) -> str:
        """
        출력에서 thinking 부분을 모델에 무관하게 제거.
        tokenizer의 chat template을 assistant 메시지로 round-trip하여
        template 내장 strip_thinking 로직을 활용.
        """
        if not (self.enable_thinking and self.strip_thinking):
            return text

        msgs = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": text},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, enable_thinking=False
            )
            token_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            cleaned = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            if "model\n" in cleaned:
                cleaned = cleaned.split("model\n", 1)[1]
            return cleaned.strip()
        except Exception:
            pass
        return text

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []
        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text),
                generation_info=g.generation_info,
            )
            chat_generations.append(chat_generation)
        return ChatResult(
            generations=chat_generations,
            llm_output=llm_result.llm_output,
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """출력 후처리: thinking 토큰 자동 제거."""
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        if self.enable_thinking and self.strip_thinking:
            for gen in result.generations:
                cleaned = self._strip_thinking_content(gen.message.content)
                gen.message = AIMessage(content=cleaned)

        return result

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        """스트리밍 출력에서 thinking 구간을 건너뛴다."""
        if not (self.enable_thinking and self.strip_thinking):
            yield from super()._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return

        # strip_thinking=True일 때: _generate로 완성된 결과를 받아서 chunk로 yield.
        # HuggingFacePipeline의 streamer가 skip_special_tokens=True로 하드코딩되어
        # thinking 구분 토큰이 제거된 채로 오기 때문에, stream chunk에서는 strip 불가.
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for gen in result.generations:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=gen.message.content),
                generation_info=gen.generation_info,
            )
