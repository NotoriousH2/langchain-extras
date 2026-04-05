# langchain-extras

LangChain에서 **thinking/reasoning 모델**(Qwen3, Gemma 4 등)을 쉽게 사용하기 위한 확장 유틸리티 모음입니다.

## 제공 클래스

| 클래스 | 백엔드 | 설명 |
|--------|--------|------|
| `ThinkingChatHuggingFace` | HuggingFace 로컬 파이프라인 | thinking 모드 on/off + 사고과정 자동 제거 |
| `ChatOpenAIWithReasoning` | vLLM OpenAI-호환 API | reasoning 필드를 `additional_kwargs`로 보존 |

---

## 설치

```bash
pip install langchain-huggingface langchain-openai
```

이 저장소의 파일을 프로젝트에 복사하거나, 서브모듈로 추가하세요.

```bash
git clone https://github.com/NotoriousH2/langchain-extras.git
```

---

## 사용법

### ThinkingChatHuggingFace

HuggingFace 로컬 모델에서 thinking 기능을 제어합니다.
tokenizer의 chat template을 활용하므로 Gemma 4, Qwen3 등 thinking을 지원하는 모델이면 모델에 무관하게 동작합니다.

```python
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoTokenizer
from thinking_chat import ThinkingChatHuggingFace

tokenizer = AutoTokenizer.from_pretrained("your-model")
base_llm = HuggingFacePipeline(pipeline=your_pipeline)

# thinking ON + 사고과정 자동 제거 (기본 권장)
llm = ThinkingChatHuggingFace(
    llm=base_llm,
    tokenizer=tokenizer,
    enable_thinking=True,
    strip_thinking=True,
)

# thinking ON + 사고과정 포함 (디버깅용)
llm = ThinkingChatHuggingFace(
    llm=base_llm,
    tokenizer=tokenizer,
    enable_thinking=True,
    strip_thinking=False,
)

# thinking OFF
llm = ThinkingChatHuggingFace(
    llm=base_llm,
    tokenizer=tokenizer,
    enable_thinking=False,
)

# 일반 호출
response = llm.invoke("대한민국의 수도는?")
print(response.content)

# 스트리밍
for chunk in llm.stream("대한민국의 수도는?"):
    print(chunk.content, end="")
```

### ChatOpenAIWithReasoning

vLLM 서버의 reasoning 필드를 LangChain에서 사용할 수 있게 합니다.
LangChain `ChatOpenAI`는 vLLM이 반환하는 `reasoning` 필드를 무시하는데, 이 클래스가 이를 `additional_kwargs["reasoning"]`으로 주입합니다.

```python
from chat_openai_with_reasoning import ChatOpenAIWithReasoning

# Reasoning ON (기본값)
llm = ChatOpenAIWithReasoning(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
    model="Qwen/Qwen3-8B",
    max_tokens=4096,  # reasoning + answer 합산이므로 넉넉하게
)

# invoke - reasoning과 content를 분리해서 확인
response = llm.invoke("1 + 1 = ?")
print("Reasoning:", response.additional_kwargs.get("reasoning"))
print("Answer:", response.content)

# stream - reasoning이 먼저, 이어서 content가 스트리밍됨
for chunk in llm.stream("1 + 1 = ?"):
    if chunk.additional_kwargs.get("reasoning"):
        print(chunk.additional_kwargs["reasoning"], end="")
    if chunk.content:
        print(chunk.content, end="")

# Reasoning OFF
llm_no_think = ChatOpenAIWithReasoning(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
    model="Qwen/Qwen3-8B",
    max_tokens=2048,
    enable_thinking=False,
)
```

---

## 비교

| | ThinkingChatHuggingFace | ChatOpenAIWithReasoning |
|---|---|---|
| 백엔드 | HuggingFace 로컬 파이프라인 | vLLM OpenAI-호환 API |
| thinking 제어 | chat template 직접 호출 | `extra_body`로 서버에 전달 |
| thinking 출력 | 제거(strip) 또는 포함 선택 | `additional_kwargs`에 분리 보존 |
| 스트리밍 | strip_thinking=True일 때 전체 생성 후 yield, 아닐 때 네이티브 스트리밍 | 네이티브 스트리밍 |

## License

MIT
