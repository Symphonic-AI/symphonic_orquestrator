"""
Continue.dev LangGraph Proxy — Workflow AI Ollama (v0.5.0)
Arquitetura Dual-Model com Duas Ferramentas (`tavily_web_search` + `sandbox_file_manager`):
- Helper LLM (`liquid/lfm2.5-1.2b` / `lfm2.5-helper`): Sumarização de Memória de Curto Prazo
  via `.invoke()` direto com prompt isolado (sem Pydantic e sem SystemMessage em runtime).
- Main LLM (`qwen/qwen3-1.7b` / `qwen3-main`): Agente de ferramentas via `@tool(return_direct=True)`
  + `create_agent` (`tavily_web_search` e `sandbox_file_manager`) e geração da resposta final
  via `.astream()` / `.ainvoke()` livre, com retorno integral do arquivo no ToolMessage.
"""

import os
import re
import json
import time
import uuid
from pathlib import Path
from typing import List, Literal, TypedDict, Optional, Any
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
# 1. CONFIGURAÇÃO DE AMBIENTE, SANDBOX E API
# ==========================================

load_dotenv()

SANDBOX_DIR = Path(__file__).resolve().parent / "sandbox"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL")
LOCAL_LLM_API_KEY = os.environ.get("LOCAL_LLM_API_KEY")
HELPER_MODEL_NAME = os.environ.get("HELPER_MODEL_NAME")
MAIN_MODEL_NAME = os.environ.get("MAIN_MODEL_NAME")
# TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
TAVILY_API_KEY = ""

app = FastAPI(title="Continue.dev LangGraph Proxy", version="0.5.0")

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
# 3. FUNÇÕES SINTÁTICAS E FERRAMENTAS `@tool`
# ==========================================
def sanitize_text_single_newline(raw_text: str) -> str:
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
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw_text)
    cleaned = cleaned.replace("[...]", " ")
    cleaned = re.sub(r"[\[\]{}]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def strip_outer_markdown_fences(code_text: str) -> str:
    stripped = code_text.strip()
    match = re.match(r"^```[a-zA-Z0-9_+-]*\n(.*?)\n```$", stripped, flags=re.DOTALL)
    if match:
        return match.group(1)
    return code_text


class TavilySearchInput(BaseModel):
    """Input schema for the Tavily technical documentation search tool."""
    query: str = Field(
        description=(
            "Concise English search query (3 to 6 keywords) containing the exact library, "
            "framework, and method name (e.g., 'FastAPI lifespan async context manager docs')."
        )
    )


class SandboxFileToolInput(BaseModel):
    """Input schema for reading and writing files inside the local sandbox directory."""
    operacao: Literal["read_file", "write_file"] = Field(
        description=(
            "Select the exact file system action: "
            "'read_file' to load and inspect an existing file from ./sandbox/, or "
            "'write_file' to create or overwrite a file in ./sandbox/."
        )
    )
    nome_arquivo: str = Field(
        description=(
            "Exact filename with extension to read or write inside ./sandbox/ "
            "(e.g., 'script.py', 'Main.kt', 'pipeline.sql'). Do not include folder paths."
        )
    )
    conteudo: str = Field(
        default="",
        description=(
            "MANDATORY when operacao='write_file': provide the FINAL, COMPLETE, production-ready "
            "source code to be saved on disk. Must include all required imports, strict type hints, "
            "clean modular functions/classes, error handling, and an entry point if applicable. "
            "Do NOT pass a simplified stub or placeholder here—write the exact same high-quality "
            "code that you will present to the user, without outer ``` markdown fences. "
            "Pass an empty string '' ONLY when operacao='read_file'."
        ),
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


@tool("sandbox_file_manager", args_schema=SandboxFileToolInput, return_direct=True)
def sandbox_file_manager(
    operacao: Literal["read_file", "write_file"],
    nome_arquivo: str,
    conteudo: str = "",
) -> str:
    """
    Read or write files inside the local './sandbox/' directory.
    Always call this tool when the user asks to read, inspect, create, save, or write a file
    (such as 'script.py', 'Main.kt', etc.). When operacao='write_file', pass the complete,
    production-ready code inside 'conteudo' so the file on disk matches your final answer.
    """
    safe_name = os.path.basename(nome_arquivo.strip())
    if not safe_name:
        return "FILE_TOOL_ERROR: Invalid 'nome_arquivo' (empty filename)."

    target_path = SANDBOX_DIR / safe_name
    print(
        f"[Main Tool Agent] sandbox_file_manager: operacao='{operacao}' | "
        f"nome_arquivo='{safe_name}' | len(conteudo)={len(conteudo)} chars"
    )

    if operacao == "read_file":
        if not target_path.exists() or not target_path.is_file():
            return f"FILE_TOOL_ERROR: File '{safe_name}' does not exist in ./sandbox/."
        try:
            file_text = target_path.read_text(encoding="utf-8")
            return (
                f"FILE_READ_SUCCESS: Full content of './sandbox/{safe_name}' ({len(file_text)} chars):\n"
                f"```\n{file_text}\n```"
            )
        except Exception as exc:
            return f"FILE_TOOL_ERROR: Failed to read '{safe_name}': {exc}"

    if operacao == "write_file":
        cleaned_content = strip_outer_markdown_fences(conteudo).strip()
        if not cleaned_content:
            return (
                f"FILE_WRITE_EMPTY_CONTENT_WARNING: Tool called for './sandbox/{safe_name}', "
                "but 'conteudo' was empty (0 bytes)."
            )
        try:
            target_path.write_text(cleaned_content + "\n", encoding="utf-8")
            size_bytes = len((cleaned_content + "\n").encode("utf-8"))
            line_count = len(cleaned_content.splitlines())
            return (
                f"FILE_WRITE_SUCCESS: File './sandbox/{safe_name}' has been created "
                f"({size_bytes} bytes, {line_count} lines) with the exact content below.\n"
                "Present this exact code and explain it clearly to the user:\n"
                f"```\n{cleaned_content}\n```"
            )
        except Exception as exc:
            return f"FILE_TOOL_ERROR: Failed to write '{safe_name}': {exc}"

    return f"FILE_TOOL_ERROR: Unsupported operation '{operacao}'."


main_tool_agent = create_agent(
    model=main_llm,
    tools=[tavily_web_search, sandbox_file_manager],
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
            "# TOOL EXECUTION RESULTS (VERIFIED SANDBOX FILE SYSTEM / WEB SEARCH STATE)\n"
            "IMPORTANT: If `FILE_WRITE_SUCCESS` appears below, keep your explanation and code block "
            "consistent with the saved file `./sandbox/<filename>`.\n"
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
                "preserving all active constraints, target filenames, versions, and forbidden libraries."
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
    clean_prompt = state["clean_user_prompt"]
    existing_files = [p.name for p in sorted(SANDBOX_DIR.iterdir()) if p.is_file()]
    files_str = ", ".join(existing_files) if existing_files else "None"

    agent_input = (
        "Evaluate the developer request below.\n"
        f"- **Files currently in ./sandbox/:** {files_str}\n"
        f"- **Prior Session Context:** {state['short_term_memory']}\n"
        "# TOOL EXECUTION & PARAMETER QUALITY RULES\n"
        "1. If the user asks to READ an existing file from `./sandbox/`, call `sandbox_file_manager` "
        "with `operacao='read_file'`, `nome_arquivo='<filename>'`, and `conteudo=''`.\n"
        "2. If the user asks to CREATE, SAVE, or WRITE a file (e.g. 'script.py', 'Main.kt'), "
        "call `sandbox_file_manager` with `operacao='write_file'`, `nome_arquivo='<filename>'`, and "
        "put the COMPLETE, well-structured, best-practice code inside `conteudo` (including imports, "
        "type hints, and clean functions, respecting all session constraints).\n"
        "3. If the user asks about newly released external APIs or deprecations, call `tavily_web_search`.\n"
        "4. If no file operation and no external web search are needed, do NOT call any tool and reply only 'NO_TOOL'.\n"
        f"# DEVELOPER REQUEST\n{clean_prompt}"
    )

    web_text = "Nenhuma busca web solicitada."
    try:
        result = main_tool_agent.invoke({"messages": [HumanMessage(content=agent_input)]})
        messages: List[BaseMessage] = result.get("messages", [])
        tool_outputs: List[str] = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.content:
                if "No relevant external web documentation found" not in str(msg.content):
                    tool_outputs.append(str(msg.content))
        if tool_outputs:
            web_text = "\n".join(tool_outputs)
    except Exception as exc:
        print(f"[Aviso Tool Agent]: {exc}")

    return {"web_search_data": web_text}


def prompt_builder_node(state: AgentState):
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