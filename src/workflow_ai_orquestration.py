"""
Continue.dev LangGraph Proxy — Workflow AI Ollama (v0.5.0)
Arquitetura Dual-Model Refinada:
- Helper LLM (`liquid/lfm2.5-1.2b` / `lfm2.5-helper`): Sumarização de Memória de Curto Prazo
  via `.invoke()` direto com prompt isolado (sem Pydantic e sem SystemMessage em runtime).
- Main LLM (`qwen/qwen3-1.7b` / `qwen3-main`): Agente de busca web via `@tool(return_direct=True)`
  + `create_agent` e geração da resposta final via `.astream()` / `.ainvoke()` livre.
"""

import os
import re
import json
import time
import uuid
from typing import List, TypedDict, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

try:
    from langchain.agents import create_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent as create_agent

# ==========================================
# 1. CONFIGURAÇÃO DE AMBIENTE E API
# ==========================================

load_dotenv()

LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL")
LOCAL_LLM_API_KEY = os.environ.get("LOCAL_LLM_API_KEY")
HELPER_MODEL_NAME = os.environ.get("HELPER_MODEL_NAME")
MAIN_MODEL_NAME = os.environ.get("MAIN_MODEL_NAME")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

app = FastAPI(title="AI Workflow Reverse Proxy", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. CONFIGURAÇÃO DOS MODELOS (SEM OVERRIDES)
# ==========================================
# Parâmetros (temperature, top_k, top_p, min_p, num_ctx) e SYSTEM prompts
# são gerenciados nativamente pelos Modelfiles no Ollama ou no LM Studio.

helper_llm = ChatOpenAI(
    base_url=LOCAL_LLM_BASE_URL,
    api_key=LOCAL_LLM_API_KEY,
    model=HELPER_MODEL_NAME,
)

main_llm = ChatOpenAI(
    base_url=LOCAL_LLM_BASE_URL,
    api_key=LOCAL_LLM_API_KEY,
    model=MAIN_MODEL_NAME,
)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

SESSION_MEMORY = {
    "summary": "",
    "summarized_count": 0,
}

# ==========================================
# 3. FUNÇÕES SINTÁTICAS E FERRAMENTA `@tool`
# ==========================================

def sanitize_text_single_newline(raw_text: str) -> str:
    """
    Remove caracteres de controle/invisíveis, normaliza espaços e colapsa
    quebras de linha duplas (\n\n+) em quebras simples (\n).
    """
    if not raw_text:
        return ""
    cleaned = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\ufeff]", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def sanitize_raw_tavily_snippet(raw_text: str, max_chars: int = 450) -> str:
    """Limpa links markdown, colchetes e marcadores [...] dos resultados brutos do Tavily."""
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw_text)
    cleaned = cleaned.replace("[...]", " ")
    cleaned = re.sub(r"[\[\]{}]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


class TavilySearchInput(BaseModel):
    """Input schema for the Tavily technical documentation search tool."""
    query: str = Field(
        description=(
            "Concise English search query (3 to 6 keywords) containing the exact library, "
            "framework, and method name (e.g., 'FastAPI lifespan async context manager docs')."
        )
    )


@tool("tavily_web_search", args_schema=TavilySearchInput, return_direct=True)
def tavily_web_search(query: str) -> str:
    """
    Search external web documentation ONLY when the user asks about newly released APIs,
    version-specific deprecations, or external libraries requiring up-to-date docs.
    Do NOT call this tool for standard Python, SQL, or classic framework tasks.
    """
    clean_query = sanitize_text_single_newline(query)
    print(f"[Main Tool Agent] Acionou tavily_web_search: '{clean_query}'")

    if tavily_client is None:
        return "No relevant external web documentation found (TAVILY_API_KEY not configured)."

    snippets: List[str] = []
    try:
        response = tavily_client.search(
            query=clean_query,
            search_depth="basic",
            max_results=2,
            include_raw_content=False,
            include_answer=False,
        )
        results = response.get("results", [])
        filtered = [r for r in results if r.get("score", 1.0) >= 0.40][:2] or results[:2]
        for idx, item in enumerate(filtered, 1):
            title = sanitize_raw_tavily_snippet(item.get("title", "Doc"), max_chars=90)
            url = item.get("url", "")
            body = sanitize_raw_tavily_snippet(item.get("content", ""), max_chars=450)
            snippets.append(f"- **Source {idx} ({title} - {url}):** {body}")
    except Exception as exc:
        print(f"[Tavily API Error]: {exc}")

    return "\n".join(snippets) if snippets else "No relevant external web documentation found."


# Agente de ferramentas vinculado ao Main LLM (Qwen3)
main_tool_agent = create_agent(
    model=main_llm,
    tools=[tavily_web_search],
)


class OpenAIMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    stream: Optional[bool] = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ==========================================
# 4. PREPARAÇÃO DE CONTEÚDO E MONTAGEM DE PROMPT
# ==========================================

def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            [item.get("text", "") for item in content if isinstance(item, dict) and "text" in item]
        )
    return str(content)


def sanitize_user_prompt(raw_text: str) -> str:
    """Preserva indentação de código, mas remove quebras de linha exageradas e ruídos."""
    if not raw_text:
        return ""
    cleaned = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\ufeff]", "", cleaned)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    normalized_lines = []
    for line in lines:
        if line and not line.startswith((" ", "\t")):
            normalized_lines.append(" ".join(line.split()))
        else:
            normalized_lines.append(line)
    return re.sub(r"\n{2,}", "\n", "\n".join(normalized_lines)).strip()


def assemble_final_prompt(
    memoria: str,
    ide_prompt: str,
    dados_web: str,
    prompt_usuario: str,
) -> List[BaseMessage]:
    """
    Monta um único HumanMessage em Markdown enxuto (sem SystemMessage em runtime e sem \n\n),
    orientando explicitamente o Main LLM sobre como tratar cada seção.
    """
    blocks: List[str] = []

    if ide_prompt:
        blocks.append(f"# IDE CONTEXT RULES\n{sanitize_text_single_newline(ide_prompt)}")

    if memoria and memoria != "Nenhum histórico.":
        blocks.append(
            "# PRIOR SESSION SUMMARY (BACKGROUND ONLY - DO NOT ANSWER PAST TURNS)\n"
            f"{sanitize_text_single_newline(memoria)}"
        )

    if dados_web and dados_web not in (
        "Nenhuma busca web solicitada.",
        "No relevant external web documentation found.",
    ):
        blocks.append(
            "# EXTERNAL TECHNICAL REFERENCE (VERIFIED DOCS - APPLY ONLY TO SUPPORT YOUR ANSWER BELOW)\n"
            f"{dados_web.strip()}"
        )

    blocks.append(
        "# CURRENT USER REQUEST (YOUR ACTIVE REQUEST TO BE ANSWERED AND FOCUSED FOR THE USER)\n"
        f"{prompt_usuario}"
    )

    return [HumanMessage(content="\n".join(blocks))]


# ==========================================
# 5. DEFINIÇÃO DO ESTADO E NÓS DO LANGGRAPH
# ==========================================

class AgentState(TypedDict):
    messages: List[BaseMessage]
    system_prompt_ide: str
    short_term_memory: str
    clean_user_prompt: str
    web_search_data: str
    final_prompt: List[BaseMessage]


def memory_and_input_prep_node(state: AgentState):
    """
    Usa o Helper LLM (`LFM2.5-1.2B`) via `.invoke()` direto com prompt isolado
    (sem Pydantic e sem SystemMessage em runtime) para comprimir o histórico incrementalmente.
    """
    messages = state.get("messages", [])
    total_msgs = len(messages)

    if total_msgs <= 2:
        SESSION_MEMORY["summary"] = ""
        SESSION_MEMORY["summarized_count"] = 0

    current_summary = SESSION_MEMORY["summary"]
    summarized_count = SESSION_MEMORY["summarized_count"]

    if total_msgs > 3:
        cutoff = total_msgs - 1
        new_to_summarize = messages[summarized_count:cutoff]
        recent_messages = messages[-1:]

        if new_to_summarize:
            history_text = "\n".join(
                [
                    f"- {'Developer' if isinstance(m, HumanMessage) else 'Assistant'}: "
                    f"{sanitize_text_single_newline(m.content)[:400]}"
                    for m in new_to_summarize
                ]
            )

            prompt_memoria = (
                "# PREVIOUS SESSION SUMMARY\n"
                f"{current_summary if current_summary else 'None'}\n"
                "# NEW DIALOGUE LOGS TO INTEGRATE IN THIRD PERSON\n"
                f"{history_text}\n"
                "# INSTRUCTION\n"
                "Return ONLY the updated third-person technical summary (max 3 bullet points) "
                "preserving all active constraints, target versions, and forbidden libraries."
            )

            try:
                res = helper_llm.invoke([HumanMessage(content=prompt_memoria)])
                current_summary = sanitize_text_single_newline(res.content)
                SESSION_MEMORY["summary"] = current_summary
                SESSION_MEMORY["summarized_count"] = cutoff
                print(f"[Memória LFM2.5-1.2B] Atualizada ({len(current_summary)} chars):\n{current_summary}")
            except Exception as exc:
                print(f"[Erro LFM2.5-1.2B]: {exc}")

        active_messages = recent_messages
    else:
        active_messages = messages

    prior_buffer = "\n".join(
        [
            f"- {'Developer' if isinstance(m, HumanMessage) else 'Assistant'}: "
            f"{sanitize_text_single_newline(m.content)[:300]}"
            for m in active_messages[:-1]
        ]
    )

    partes_memoria = list(filter(None, [current_summary, prior_buffer]))
    memoria_consolidada = "\n".join(partes_memoria) if partes_memoria else "Nenhum histórico."

    raw_user_input = active_messages[-1].content
    clean_user_input = sanitize_user_prompt(raw_user_input)

    return {
        "short_term_memory": memoria_consolidada,
        "clean_user_prompt": clean_user_input,
        "web_search_data": "Nenhuma busca web solicitada.",
    }


def tool_agent_search_node(state: AgentState):
    """
    Utiliza `create_agent` + `@tool("tavily_web_search", return_direct=True)` com o `main_llm`.
    O modelo decide autonomamente se invoca a tool de busca ou responde 'NO_SEARCH'.
    """
    clean_prompt = state["clean_user_prompt"]
    agent_input = (
        "Evaluate the developer request below. /no_think\n"
        "Call `tavily_web_search` ONLY if external up-to-date documentation is strictly required. "
        "If standard programming knowledge is sufficient, do NOT call any tool and reply only 'NO_SEARCH'.\n"
        f"# DEVELOPER REQUEST\n{clean_prompt}"
    )

    web_text = "Nenhuma busca web solicitada."
    try:
        result = main_tool_agent.invoke({"messages": [HumanMessage(content=agent_input)]})
        messages: List[BaseMessage] = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                if msg.content and "No relevant external web documentation found" not in str(msg.content):
                    web_text = str(msg.content)
                break
    except Exception as exc:
        print(f"[Aviso Tool Agent Search]: {exc}")

    return {"web_search_data": web_text}


def prompt_builder_node(state: AgentState):
    """Constrói o HumanMessage final orientado para permitir SSE Streaming no FastAPI."""
    final_messages = assemble_final_prompt(
        memoria=state["short_term_memory"],
        ide_prompt=state["system_prompt_ide"],
        dados_web=state["web_search_data"],
        prompt_usuario=state["clean_user_prompt"],
    )
    return {"final_prompt": final_messages}


workflow = StateGraph(AgentState)
workflow.add_node("memory_and_input_prep", memory_and_input_prep_node)
workflow.add_node("tool_agent_search", tool_agent_search_node)
workflow.add_node("prompt_builder", prompt_builder_node)

workflow.set_entry_point("memory_and_input_prep")
workflow.add_edge("memory_and_input_prep", "tool_agent_search")
workflow.add_edge("tool_agent_search", "prompt_builder")
workflow.add_edge("prompt_builder", END)
app_graph = workflow.compile()


# ==========================================
# 6. ENDPOINT SSE (VS CODE / CONTINUE.DEV)
# ==========================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    lc_messages: List[BaseMessage] = []
    system_prompt_ide = ""

    for m in request.messages:
        text = normalize_content(m.content)
        if m.role == "system":
            system_prompt_ide += text + "\n"
        elif m.role == "user":
            lc_messages.append(HumanMessage(content=text))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=text))

    if not lc_messages:
        raise HTTPException(status_code=400, detail="Sem mensagens do usuário.")

    initial_state: AgentState = {
        "messages": lc_messages,
        "system_prompt_ide": system_prompt_ide.strip(),
        "short_term_memory": "",
        "clean_user_prompt": "",
        "web_search_data": "",
        "final_prompt": [],
    }

    processed_state = app_graph.invoke(initial_state)
    final_prompt = processed_state["final_prompt"]

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    if request.stream:
        async def sse_generator():
            try:
                async for chunk in main_llm.astream(final_prompt):
                    if chunk.content:
                        payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk.content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"

                end_payload = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(end_payload)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as stream_err:
                print(f"[Erro de Stream]: {stream_err}")
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    response = await main_llm.ainvoke(final_prompt)
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created_time,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    print(
        f"=== Iniciando LangGraph Proxy v0.5.0 (Helper: {HELPER_MODEL_NAME} | Main: {MAIN_MODEL_NAME}) na porta 8000 ==="
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)