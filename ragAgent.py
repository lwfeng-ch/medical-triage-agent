# ragAgent.py
import logging
import logging.config
import json
import os
import re
import sys
import threading
import time
import uuid
import asyncio
from html import escape
from typing import Literal, Annotated, Optional, Tuple, List, Dict
import operator
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from utils.config import Config
from utils.logger import setup_logger
from utils.llms import get_llm
from utils.tools_config import get_rag_tools, get_medical_agent_tools
from utils.middleware import MiddlewareManager
from utils.feishu_mcp import feishu_mcp_manager
from cachetools import TTLCache

os.environ['NO_PROXY'] = 'localhost,127.0.0.1'


class RagAgentError(Exception):
    """RagAgent 基础异常 — 所有 ragAgent 模块异常的父类。"""

    def __init__(self, message: str, code: str = "UNKNOWN", details: dict = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {"error": self.message, "code": self.code, "details": self.details}


class GraphBuildError(RagAgentError):
    """图谱构建失败（数据库连接、编译错误等）。"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, code="GRAPH_BUILD_ERROR", details=details)


class ResponseExtractionError(RagAgentError):
    """响应提取失败（事件流解析异常、数据格式不匹配等）。"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, code="RESPONSE_EXTRACTION_ERROR", details=details)


class MedicalAnalysisError(RagAgentError):
    """医疗分析流程异常（分诊、安全守卫节点失败等）。"""

    def __init__(self, message: str, risk_level: str = "high", details: dict = None):
        super().__init__(message, code="MEDICAL_ANALYSIS_ERROR", details={** (details or {}), "risk_level": risk_level})


class ToolExecutionError(RagAgentError):
    """工具执行异常（超时、重试耗尽等）。"""

    def __init__(self, message: str, tool_name: str = "", details: dict = None):
        super().__init__(message, code="TOOL_EXECUTION_ERROR", details={** (details or {}), "tool_name": tool_name})


logger = setup_logger(__name__)


@dataclass
class Context:
    """用户上下文，包含用户 ID。"""
    user_id: str


class AgentState(MessagesState):
    """对话状态，包含业务字段和 Middleware 追踪字段。
    
    LangGraph 的状态是每次执行独立的，天然线程安全。
    所有 Middleware 的可变状态都存在此处，而非 Middleware 实例上。
    """
    # ===== 业务字段 =====
    relevance_score: Optional[str] = None  # 检索相关性评分
    rewrite_count: int = 0                 # 查询重写次数（防死循环）

    # ===== Middleware 追踪字段（每次执行独立，多用户安全） =====
    mw_model_call_count: Annotated[int, operator.add] = 0
    mw_model_total_time: Annotated[float, operator.add] = 0.0
    mw_tool_total_time: Annotated[float, operator.add] = 0.0
    mw_pii_detected: bool = False   # 是否检测到个人隐私信息
    mw_force_stop: bool = False     # 强制停止标志
    mw_node_timings: Optional[dict] = None  # 各节点耗时记录
    
    # 路由字段
    route_domain: Optional[Literal["general", "medical"]] = None
    route_reason: Optional[str] = None
    
    # 医疗 Agent 相关字段
    medical_analysis_result: Optional[dict] = None
    medical_context: Optional[str] = None
    retrieval_context: Optional[str] = None
    
    # 医疗建议字段
    recommended_departments: Optional[List[str]] = None  # 推荐科室列表，如：["心内科", "急诊科"]
    triage_reason: Optional[str] = None  # 分诊理由说明
    urgency_level: Optional[Literal["routine", "urgent", "emergency"]] = None  # 紧急程度三级分类
    triage_confidence: Optional[float] = None  # 分诊置信度：0.0 ~ 1.0
    
    # 风险等级字段
    risk_level: Optional[Literal["low","medium","high","critical"]] = None  # 四级风险评估
    need_clarification: bool = False  # True → 信息不足，需追问用户
    final_payload: Optional[dict] = None  # 最终输出载荷，汇总所有结果供API返回

# 定义工具配置类，用于存储和管理工具列表、名称集合和路由配置
class ToolConfig:
    """
    工具配置类 - 实施物理工具隔离。
    
    安全约束：
    - rag_tools: 仅包含检索工具，用于通用 RAG Agent
    - medical_tools: 包含完整工具集，用于医疗 Agent
    """
    
    def __init__(self, rag_tools, medical_tools):
        """
        初始化工具配置。

        Args:
            rag_tools: RAG Agent 工具列表（仅检索工具）
            medical_tools: Medical Agent 工具列表（完整工具集）
        """
        self.rag_tools = rag_tools
        self.medical_tools = medical_tools
        
        self.rag_tool_names = {tool.name for tool in rag_tools}
        self.medical_tool_names = {tool.name for tool in medical_tools}
        
        self.rag_routing_config = self._build_routing_config(rag_tools)
        self.medical_routing_config = self._build_routing_config(medical_tools)
        
        logger.info(f"ToolConfig 初始化完成:")
        logger.info(f"  - RAG Agent 工具: {self.rag_tool_names}")
        logger.info(f"  - Medical Agent 工具: {self.medical_tool_names}")

    def _build_routing_config(self, tools):
        routing_config = {}
        for tool in tools:
            tool_name = tool.name.lower()
            if "retrieve" in tool_name:
                routing_config[tool_name] = "grade_documents"
                logger.debug(f"Tool '{tool_name}' routed to 'grade_documents' (retrieval tool)")
            else:
                routing_config[tool_name] = "generate"
                logger.debug(f"Tool '{tool_name}' routed to 'generate' (non-retrieval tool)")
        if not routing_config:
            logger.warning("No tools provided or routing config is empty")
        return routing_config

    def get_rag_tools(self):
        """获取 RAG Agent 工具列表（仅检索工具）。"""
        return self.rag_tools

    def get_medical_tools(self):
        """获取 Medical Agent 工具列表（完整工具集）。"""
        return self.medical_tools

# 定义文档相关性评分模型，用于存储文档的二进制评分（'yes' 或 'no'）
class DocumentRelevanceScore(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


class ParallelToolNode:
    """
    接收工具列表和最大线程数作为参数，初始化一个工具节点对象。
    当调用并行工具节点时，它会从状态中提取消息列表，获取最后一个消息的工具调用列表，
    并行执行每个工具调用，将结果返回为工具节点，将所有工具调用的结果合并为一个列表，作为图的输出。
    """
    def __init__(self, tools, max_workers: int = None, middleware_manager: MiddlewareManager = None, timeout: int = None):
        self.tools = tools
        self.max_workers = max_workers or Config.PARALLEL_TOOL_MAX_WORKERS
        self.timeout = timeout or Config.PARALLEL_TOOL_TIMEOUT
        self.middleware_manager = middleware_manager
        self._retry_middleware = (
            middleware_manager.get_tool_retry_middleware() if middleware_manager else None
        )

    def _run_single_tool(self, tool_call: dict, tool_map: dict) -> Tuple[ToolMessage, dict]:
        """执行单个工具调用，返回 (ToolMessage, mw_updates)。
        
        Middleware 集成点:before_tool / after_tool / wrap_tool_call重试
        """
        mw_updates = {}
        try:
            tool_name = tool_call["name"]
            tool = tool_map.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")

            # ===== Middleware: before_tool =====
            if self.middleware_manager:
                before_updates, stop = self.middleware_manager.run_before_tool({}, tool_call)
                mw_updates.update(before_updates)
                if stop:
                    return ToolMessage(
                        content="工具调用被安全策略拦截",
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    ), mw_updates

            # 执行工具（支持重试）
            start_time = time.time()
            if self._retry_middleware:
                # 使用重试 Middleware 包裹工具调用
                def _invoke(tc, tm):
                    t = tm.get(tc["name"])
                    return t.invoke(tc["args"])
                result = self._retry_middleware.wrap_tool_call(_invoke, tool_call, tool_map)
            else:
                result = tool.invoke(tool_call["args"])
            elapsed = time.time() - start_time

            # ===== Middleware: after_tool =====
            if self.middleware_manager:
                after_updates = self.middleware_manager.run_after_tool({}, result, tool_name, elapsed)
                mw_updates.update(after_updates)

            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name
            ), mw_updates

        except Exception as e:
            logger.error(f"Error executing tool {tool_call.get('name', 'unknown')}: {e}")
            return ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_call["id"],
                name=tool_call.get("name", "unknown")
            ), mw_updates

    def __call__(self, state: dict) -> dict:
        """并行执行所有工具调用"""
        logger.info("ParallelToolNode processing tool calls")
        if isinstance(state, dict) and "messages" in state:
            messages = state["messages"]
        elif hasattr(state, "messages"):
            messages = state.messages
        else:
            logger.warning("No messages found in state")
            return {"messages": []}

        if not messages:
            logger.warning("Messages list is empty")
            return {"messages": []}

        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            logger.warning("No tool calls found in state")
            return {"messages": []}

        tool_map = {tool.name: tool for tool in self.tools}
        results = []
        all_mw_updates = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tool = {
                executor.submit(self._run_single_tool, tool_call, tool_map): tool_call
                for tool_call in tool_calls
            }
            for future in as_completed(future_to_tool, timeout=self.timeout):
                try:
                    tool_msg, mw_updates = future.result(timeout=self.timeout)
                    results.append(tool_msg)
                    for k, v in mw_updates.items():
                        if k in all_mw_updates and isinstance(v, (int, float)):
                            all_mw_updates[k] = all_mw_updates[k] + v
                        else:
                            all_mw_updates[k] = v
                except TimeoutError:
                    logger.error(f"Tool execution timed out after {self.timeout}s")
                    tool_call = future_to_tool[future]
                    results.append(ToolMessage(
                        content=f"Error: Tool execution timed out after {self.timeout}s",
                        tool_call_id=tool_call["id"],
                        name=tool_call.get("name", "unknown")
                    ))
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    tool_call = future_to_tool[future]
                    results.append(ToolMessage(
                        content=f"Unexpected error: {str(e)}",
                        tool_call_id=tool_call["id"],
                        name=tool_call.get("name", "unknown")
                    ))

        logger.info(f"Completed {len(results)} tool calls")
        return {"messages": results, **all_mw_updates}

# 定义获取最新用户问题的函数，用于从状态中提取用户输入的最新问题。
def get_latest_question(state: AgentState) -> Optional[str]:
    """从状态中安全地获取最新用户问题。

    Args:
        state: 当前对话状态，包含消息历史。

    Returns:
        Optional[str]: 最新问题的内容，如果无法获取则返回 None。
    """
    try:
        if not state.get("messages") or not isinstance(state["messages"], (list, tuple)) or len(state["messages"]) == 0:
            logger.warning("No valid messages found in state for getting latest question")
            return None
        
        # 
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage) and hasattr(message, "content"):
                return message.content

        logger.info("No HumanMessage found in state")
        return None

    except Exception as e:
        logger.error(f"Error getting latest question: {e}")
        return None

# 定义消息过滤函数，用于从消息列表中提取最后5条消息。
# LangChain v1 变更说明：AIMessage 现在是 ChatOpenAI invoke() 的确切返回类型
def filter_messages(messages: list) -> list:
    """
    过滤消息列表，保留 AIMessage、HumanMessage 和 ToolMessage，确保配对完整性。
    
    修复问题：
    1. ToolMessage 被过滤导致 LLM 看不到工具结果（幻觉输出）
    2. 截断时切断 AIMessage 与 ToolMessage 的配对关系（API报错风险）
    3. 状态累积层：老的 AIMessage tool_calls 污染新轮上下文
    
    评审意见参考：评估.md 问题1修复方案
    """
    filtered = []
    pending_tool_call_ids = set()
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            pending_tool_call_ids.clear()
            filtered.append(msg)
            
        elif isinstance(msg, AIMessage):
            filtered.append(msg)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id:
                        pending_tool_call_ids.add(tc_id)
                    
        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id in pending_tool_call_ids:
                filtered.append(msg)
                pending_tool_call_ids.discard(msg.tool_call_id)
    
    return _truncate_by_human_message_boundary(filtered, max_turns=3)


def _truncate_by_human_message_boundary(messages: list, max_turns: int) -> list:
    """
    以 HumanMessage 为边界截断，保证每轮消息完整。
    
    Args:
        messages: 消息列表
        max_turns: 最大保留轮次
        
    Returns:
        list: 截断后的消息列表
    """
    turn_boundaries = [i for i, m in enumerate(messages) 
                       if isinstance(m, HumanMessage)]
    if len(turn_boundaries) <= max_turns:
        return messages
    start_idx = turn_boundaries[-max_turns]
    return messages[start_idx:]


def _get_current_turn_messages(messages: list) -> list:
    """
    获取当前轮次的消息（从最后一个 HumanMessage 开始）。
    
    修复问题：
    - medical_analysis_result 跨轮累积，导致分析结果包含历史轮次的工具输出
    
    评审意见参考：评估.md 问题6修复方案
    
    Args:
        messages: 完整消息列表
        
    Returns:
        list: 当前轮次的消息列表
    """
    # 从后向前查找最后一个 HumanMessage
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    
    # 如果没有找到 HumanMessage，返回所有消息
    return messages


def collect_tool_contents(state: AgentState) -> str:
    """
    从状态中收集所有 ToolMessage 的内容并合并。

    Args:
        state: 当前对话状态。

    Returns:
        str: 合并后的工具输出内容，多个结果以换行分隔。
    """
    tool_contents = []
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_contents.append(msg.content)
        elif not isinstance(msg, ToolMessage) and tool_contents:
            break
    return "\n\n".join(reversed(tool_contents))


def store_memory(question: str, user_id: str, store: BaseStore) -> str:
    """存储用户输入中的记忆信息。

    Args:
        question (str): 用户输入的问题内容。
        user_id (str): 用户ID。
        store (BaseStore): 数据存储实例。

    Returns:
        str: 用户相关的记忆信息字符串。
    """
    namespace = ("memories", user_id)
    try:
        memories = store.search(namespace, query=str(question))
        user_info = "\n".join([d.value["data"] for d in memories])

        if "记住" in question.lower():
            memory = escape(question)
            store.put(namespace, str(uuid.uuid4()), {"data": memory})
            logger.info(f"Stored memory: {memory}")

        return user_info
    except Exception as e:
        logger.error(f"Error in store_memory: {e}")
        return ""


_prompt_cache_lock = threading.Lock()
_template_content_cache_lock = threading.Lock()

_prompt_cache = TTLCache(maxsize=Config.PROMPT_CACHE_MAX_SIZE, ttl=Config.PROMPT_CACHE_TTL)
_template_content_cache = TTLCache(maxsize=Config.PROMPT_CACHE_MAX_SIZE, ttl=Config.PROMPT_CACHE_TTL)


def _set_temperature(llm: BaseChatModel, temperature: float) -> BaseChatModel:
    """
    安全设置 LLM 温度参数，兼容不同 LangChain 版本。

    Args:
        llm: 语言模型实例。
        temperature: 目标温度值。

    Returns:
        BaseChatModel: 设置了温度的新 LLM 实例。
    """
    if hasattr(llm, "model_copy"):
        return llm.model_copy(update={"temperature": temperature})

    return llm.bind(temperature=temperature)


def create_chain(llm_chat, template_file: str, structured_output=None, max_retries: int = 3):
    """创建 LLM 处理链，加载提示模板并绑定模型，使用缓存避免重复读取文件。

    LangChain v1 变更说明：
    - PromptTemplate, ChatPromptTemplate 仍在 langchain_core.prompts 中，无需修改
    - with_structured_output() 方法保持不变
    - LCEL（LangChain Expression Language）管道操作符 | 保持不变

    Args:
        llm_chat: 语言模型实例。
        template_file: 提示模板文件路径。
        structured_output: 可选的结构化输出模型。
        max_retries: 最大重试次数（默认 3 次）。

    Returns:
        Runnable: 配置好的处理链（带重试机制）。

    Raises:
        FileNotFoundError: 如果模板文件不存在。
    """
    try:
        if template_file in _prompt_cache:
            prompt_template = _prompt_cache[template_file]
            logger.info(f"Using cached prompt template for {template_file}")
        else:
            with _prompt_cache_lock:
                if template_file not in _prompt_cache:
                    logger.info(f"Loading and caching prompt template from {template_file}")
                    _prompt_cache[template_file] = PromptTemplate.from_file(template_file, encoding="utf-8")
                prompt_template = _prompt_cache[template_file]

        prompt = ChatPromptTemplate.from_messages([("human", prompt_template.template)])
        base_chain = prompt | (llm_chat.with_structured_output(structured_output) if structured_output else llm_chat)
        
        if structured_output and max_retries > 0:
            return base_chain.with_retry(
                stop_after_attempt=max_retries,
                wait_exponential_jitter=True,
                retry_if_exception_type=(Exception,)
            )
        return base_chain
    except FileNotFoundError:
        logger.error(f"Template file {template_file} not found")
        raise

def load_prompt_template(template_file: str) -> str:
    """
    加载 prompt 模板文件内容，使用缓存避免重复读取。

    Args:
        template_file: 模板文件路径。

    Returns:
        str: 模板内容字符串。

    Raises:
        FileNotFoundError: 如果模板文件不存在。
    """
    try:
        if template_file in _template_content_cache:
            logger.info(f"Using cached prompt template for {template_file}")
            return _template_content_cache[template_file]
        
        with _template_content_cache_lock:
            if template_file not in _template_content_cache:
                logger.info(f"Loading and caching prompt template from {template_file}")
                with open(template_file, "r", encoding="utf-8") as f:
                    _template_content_cache[template_file] = f.read()
            return _template_content_cache[template_file]
    except FileNotFoundError:
        logger.error(f"Template file {template_file} not found")
        raise

# 定义意图路由节点的输出模型，用于存储意图和分类原因。
class IntentResult(BaseModel):
    intent: Literal["general", "medical"]
    reason: str = Field(default="")
    

def intent_router(
    state: AgentState,
    config: dict,
    llm_chat: BaseChatModel,
    middleware_manager: Optional[MiddlewareManager] = None
) -> dict:
    """
    意图路由节点 - 根据用户问题判断意图并设置 route_domain（正则提取 JSON 方案）。

    - prompt层面:限定LLM输出只能输出json格式的意图分类结果和原因
    - 类定义层面:定义router 类，对分类意图和分类原因进行规范
    - 验证方面：采用pydantic 进行自动化验证
    
    鲁棒性设计：
    - 支持空问题处理
    - 完整的错误日志记录

    Args:
        state: 当前对话状态。
        config: 运行时配置字典。
        llm_chat: Chat模型。
        middleware_manager: Middleware管理器。

    Returns:
        dict: 更新后的状态，包含 route_domain 字段。
    """
    logger.info("Intent router processing user query (regex extraction mode)")
    node_name = "intent_router"
    mw_updates = {}

    try:
        # ===== Middleware: before_model =====
        if middleware_manager:
            before_updates, should_stop = middleware_manager.run_before_model(state, node_name)
            mw_updates.update(before_updates)
            if should_stop:
                logger.warning(f"[{node_name}] 被 Middleware 终止")
                return {
                    "messages": [SystemMessage(content="请求已被安全策略拦截")],
                    "route_domain": "general",
                    **mw_updates
                }

        question = get_latest_question(state)
        if not question:
            return {
                "route_domain": "general",
                "route_reason": "empty_question",
                **mw_updates
            }

        # 使用普通的 LLM 调用
        intent_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_INTENT_ROUTER)
        
        start_time = time.time()
        response = intent_chain.invoke({"question": question})
        elapsed = time.time() - start_time

        # 提取文本内容
        content = ""
        if hasattr(response, 'content'):
            content = response.content.strip()
        elif isinstance(response, str):
            content = response.strip()
        else:
            content = str(response).strip()

        logger.debug(f"[{node_name}] 原始 LLM 输出: {content[:200]}...")

        # 使用正则表达式安全提取 JSON 块
        json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*?\}))*?\}', content)
        # 提取 JSON 块
        json_str = json_match.group(0)
        
        # 校验 JSON 块是否符合模型要求
        try:
            result = IntentResult.model_validate_json(json_str)

        except Exception as e:
            logger.error(f"[{node_name}] Pydantic 校验失败: {e}")
            return {
                "route_domain": "general",
                "route_reason": "invalid_schema",
                **mw_updates
            }

        intent = result.intent
        reason = result.reason
        
        logger.info(f"{node_name} 分类完成",
            extra={
                "intent": intent,
                "reason": reason,
                "elapsed_ms": elapsed * 1000,
            }
        )
        
        
        #  Middleware: after_model 
        if middleware_manager:
            after_updates = middleware_manager.run_after_model(
                state,
                result.model_dump(),
                node_name,
                elapsed
            )
            mw_updates.update(after_updates)

        return {
            "route_domain": intent,
            "route_reason": reason, 
            "final_payload": None,
            "risk_level": None,
            "recommended_departments": None,
            "urgency_level": None,
            "medical_analysis_result": None,
            **mw_updates
        }

    except Exception as e:
        logger.error(f"Intent parsing failed: {e}", exc_info=True)
        return {
            "route_domain": "general",
            "route_reason": "exception_fallback",
            **mw_updates
        }
        

# 定义代理函数，用于处理用户查询并调用工具或生成响应。
def agent(state: AgentState, config: dict, llm_chat, tool_config: ToolConfig,
          store=None, middleware_manager: MiddlewareManager = None) -> dict:
    """RAG Agent 函数 - 仅绑定检索工具（物理隔离医疗工具）。

    安全约束：此 Agent 只能访问 health_record_retriever，禁止访问医疗分析工具。

    Args:
        state: 当前对话状态。
        config: 运行时配置字典，包含 configurable 信息。
        llm_chat: Chat模型。
        tool_config: 工具配置参数。
        store: 数据存储实例（可选）。
        middleware_manager: Middleware管理器。

    Returns:
        dict: 更新后的对话状态。
    """   
    logger.info("RAG Agent processing user query")
    node_name = "rag_agent"
    mw_updates = {}

    # 从配置中获取用户ID
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    user_id = configurable.get("user_id", "unknown")
    
    try:
        question = get_latest_question(state)
        logger.info(f"rag_agent question:{question}")

        # Middleware: before_model 
        if middleware_manager:
            before_updates, should_stop = middleware_manager.run_before_model(state, node_name)
            mw_updates.update(before_updates)
            if should_stop:
                logger.warning(f"[{node_name}] 被 Middleware 终止")
                return {
                    "messages": [SystemMessage(content="请求已被安全策略拦截，请检查输入内容")],
                    **mw_updates
                }
            # 处理 PII 脱敏内容
            if before_updates.get("_mw_masked_content"):
                question = HumanMessage(content=before_updates["_mw_masked_content"])
                logger.info(f"[{node_name}] 使用脱敏后的内容")
            # 处理摘要截断请求（不直接修改 state，使用局部变量）
            if before_updates.get("_mw_should_truncate"):
                keep = before_updates.get("_mw_keep_recent", 5)
                state_messages = state["messages"][-keep:]
            else:
                state_messages = state["messages"]
        else:
            state_messages = state["messages"]

        # 处理用户记忆
        user_info = ""
        if store:
            user_info = store_memory(question, user_id, store)
        messages = filter_messages(state_messages)

        # 物理工具隔离：只绑定检索工具
        rag_tools = tool_config.get_rag_tools()
        logger.info(f"RAG Agent 绑定工具: {[t.name for t in rag_tools]}")
        llm_chat_with_tool = llm_chat.bind_tools(rag_tools)
        agent_chain = create_chain(llm_chat_with_tool, Config.PROMPT_TEMPLATE_TXT_AGENT)

        # 计时模型调用
        start_time = time.time()
        response = agent_chain.invoke({"question": question, "messages": messages, "userInfo": user_info})
        elapsed = time.time() - start_time

        # Middleware: after_model 
        if middleware_manager:
            after_updates = middleware_manager.run_after_model(state, response, node_name, elapsed)
            mw_updates.update(after_updates)

        return {"messages": [response], **mw_updates}
    except Exception as e:
        logger.error(f"Error in rag_agent processing: {e}", exc_info=True)
        return {"messages": [SystemMessage(content="处理请求时出错")], **mw_updates}
        
# 定义文档评分函数，用于评估检索到的文档内容与问题的相关性。
def grade_documents(state: AgentState, config: dict = None, llm_chat = None, middleware_manager: MiddlewareManager = None) -> dict:
    """评估检索到的文档内容与问题的相关性，并将评分结果存储在状态中。

    Args:
        state (AgentState): 当前对话状态，包含消息历史。
        config (dict): 运行配置（LangGraph 传入）。
        llm_chat: 语言模型实例。
        middleware_manager (MiddlewareManager): 中间件管理器。

    Returns:
        dict: 更新后的状态，包含评分结果。
    """
    logger.info("Grading documents for relevance")
    node_name = "grade_documents"
    mw_updates = {}

    # 检查状态是否包含消息历史
    if not state.get("messages"):
        logger.error("Messages state is empty")
        return {"relevance_score": None}

    try:
        # ===== Middleware: before_model =====
        if middleware_manager:
            before_updates, should_stop = middleware_manager.run_before_model(state, node_name)
            mw_updates.update(before_updates)
            if should_stop:
                logger.warning(f"[{node_name}] 被 Middleware 终止，默认标记文档不相关")
                return {
                    "messages": state["messages"],
                    "relevance_score": "no",
                    **mw_updates
                }
                
        question = get_latest_question(state)
        context = collect_tool_contents(state)
        if not context or str(context).strip() == "":
            logger.warning("Retrieved context is empty, auto-grading as 'no'")
            return {
                "relevance_score": "no",
                **mw_updates
            }        

        grader_llm = _set_temperature(llm_chat, 0.0)
        logger.debug(f"Grader LLM temperature set to: {getattr(grader_llm, 'temperature', 'unknown')}")
        
        start_time = time.time()
        grade_chain = create_chain(grader_llm, Config.PROMPT_TEMPLATE_TXT_GRADE, DocumentRelevanceScore)
        scored_result = grade_chain.invoke({"question": question, "context": context})
        elapsed = time.time() - start_time
        
        score = str(scored_result.binary_score).strip().lower()
         
        logger.info(f"Document relevance score: {score}")
        
        # 二次校验：确保输出仅为 yes 或 no
        if score not in ["yes", "no"]:
            logger.warning(f"Unexpected score value: {score}, defaulting to 'no'")
            score = "no"

        # ===== Middleware: after_model =====
        if middleware_manager:
            after_updates = middleware_manager.run_after_model(state, scored_result, node_name, elapsed)
            mw_updates.update(after_updates)

        return {
            "relevance_score": score,
            **mw_updates
        }
    except (IndexError, KeyError) as e:
        logger.error(f"Message access error: {e}")
        return {
            "messages": [SystemMessage(content="无法评分文档")],
            "relevance_score": None
        }
    except Exception as e:
        logger.error(f"Unexpected error in grading: {e}")
        return {
            "messages": [SystemMessage(content="评分过程中出错")],
            "relevance_score": None
        }

# 定义重写函数，用于改进用户查询。
def rewrite(state: AgentState, config: dict = None, llm_chat = None, middleware_manager: MiddlewareManager = None) -> dict:
    """重写用户查询以改进问题。

    Args:
        state: 当前对话状态。

    Returns:
        dict: 更新后的消息状态。
    """
    logger.info("Rewriting query")
    node_name = "rewrite"
    mw_updates = {}
    
    try:
        # Middleware: before_model 
        if middleware_manager:
            before_updates, should_stop = middleware_manager.run_before_model(state, node_name)
            mw_updates.update(before_updates)
            if should_stop:
                logger.warning(f"[{node_name}] 被 Middleware 终止，跳过重写")
                return {
                    "rewrite_count": state.get("rewrite_count", 0) + 1,
                    **mw_updates
                }

        question = get_latest_question(state)
        
        messages = state.get("messages",[])
        history_msgs = [msg.content for msg in messages[-5:-1] if isinstance(msg, (HumanMessage, AIMessage))]
        chat_history = "\n".join(history_msgs) if history_msgs else "无历史对话"
        
        
        rewrite_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_REWRITE)
        
        start_time = time.time()
        response = rewrite_chain.invoke({"question": question, "chat_history": chat_history} )
        elapsed = time.time() - start_time
        
        rewrite_count = state.get("rewrite_count", 0) + 1
        logger.info(f"Rewrite count: {rewrite_count}")
        
        #  Middleware: after_model 
        if middleware_manager:
            after_updates = middleware_manager.run_after_model(state, response, node_name, elapsed)
            mw_updates.update(after_updates)
        
        rewritten_text = response.content if hasattr(response, "content") else str(response) 
        return {"messages":[HumanMessage(content=rewritten_text)], "rewrite_count": rewrite_count, **mw_updates}
    except (IndexError, KeyError) as e:
        logger.error(f"Message access error in rewrite: {e}")
        return {"messages": [SystemMessage(content="无法重写查询")], **mw_updates}

# 定义生成函数，用于基于工具返回的内容生成最终回复。
def generate(state: AgentState, config: dict = None, llm_chat = None, store = None, middleware_manager: MiddlewareManager = None) -> dict:
    """基于工具返回的内容生成最终回复。

    Args:
        state: 当前对话状态。

    Returns:
        dict: 更新后的消息状态。
    """
    logger.info("Generating final response")
    node_name = "generate"
    mw_updates = {}
    
    fallback_message = AIMessage(content="抱歉，系统在整理您的档案信息时遇到了一点小波动，请稍后再试。")
    
    try:
        # Middleware: before_model 
        if middleware_manager:
            before_updates, should_stop = middleware_manager.run_before_model(state, node_name)
            mw_updates.update(before_updates)
            if should_stop:
                logger.warning(f"[{node_name}] 被 Middleware 终止")
                return {
                    "messages":[AIMessage(content="抱歉，您的提问触发了安全策略限制，无法生成相关回复。")],
                    **mw_updates
                }
        
        question = get_latest_question(state)
        context = collect_tool_contents(state)
                
        generate_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_GENERATE)
        
        start_time = time.time()
        response = generate_chain.invoke({"context": context, "question": question})
        elapsed = time.time() - start_time
        
        if hasattr(response, 'content'):
            final_content = response.content
        else:
            final_content = str(response)
        
        final_ai_msg = AIMessage(content=final_content)
        
        logger.debug(f"[generate] response type: {type(response).__name__}, has tool_calls: {hasattr(response, 'tool_calls')}, tool_calls value: {getattr(response, 'tool_calls', None)}")
        logger.debug(f"[generate] final_ai_msg has tool_calls: {hasattr(final_ai_msg, 'tool_calls')}, tool_calls value: {getattr(final_ai_msg, 'tool_calls', None)}")
        
        content_preview = str(final_content)[:200] if final_content else "EMPTY"
        logger.info(f"[generate] Response generated in {elapsed:.2f}s. Preview: {content_preview}")
        
        # Middleware: after_model 
        if middleware_manager:
            after_updates = middleware_manager.run_after_model(state, response, node_name, elapsed)
            mw_updates.update(after_updates)
            
        return {"messages": [final_ai_msg], **mw_updates}
    except (IndexError, KeyError) as e:
        logger.error(f"Message access error in generate: {e}")
        return {"messages":  [fallback_message], **mw_updates}
    except Exception as e:
        logger.error(f"Error in generate: {e}", exc_info=True)
        return {"messages": [fallback_message], **mw_updates}

def medical_agent(state: AgentState, config: dict, llm_chat, tool_config: ToolConfig, middleware_manager: MiddlewareManager = None) -> dict:
    """
    医疗分析主控节点 - 绑定完整工具集（检索 + 医疗分析工具）。

    职责：
    - 决定调用哪些医疗工具（血常规、血生化、尿常规等）
    - 管理医疗工具调用循环
    - 将工具调用结果存储到 state
    - 实现断路器机制防止死循环

    Args:
        state: 当前对话状态。
        config: 运行时配置字典。
        llm_chat: Chat模型。
        tool_config: 工具配置参数。
        middleware_manager: Middleware管理器。

    Returns:
        dict: 更新后的状态。
    """
    logger.info("Medical Agent processing user query")
    node_name = "medical_agent"
    mw_updates = {}

    try:
        #  Middleware: before_model
        if middleware_manager:
            before_updates, should_stop = middleware_manager.run_before_model(state, node_name)
            mw_updates.update(before_updates)
            if should_stop:
                logger.info("[medical_agent] Middleware 要求提前终止")
                return {"messages": state["messages"], **mw_updates}
        
        # 提取用户问题
        question = get_latest_question(state)
        if not question:
            logger.warning("No question found in medical_agent")
            return {"messages": state["messages"], **mw_updates}
        
        logger.debug(f"[medical_agent] 当前问题: {question[:100]}")
        
        # 过滤消息历史，只保留用户问题和模型回复
        messages = filter_messages(state.get("messages", []))

        # 断路器机制：统计当前用户回合每个工具的调用次数
        tool_call_counts = _count_tool_calls_in_turn(state)
        max_single_tool_calls = 3

        # 检查是否有任何单个工具被调用超过限制
        over_limit_tools = {name: count for name, count in tool_call_counts.items() if count >= max_single_tool_calls}

        #  动态工具绑定决策
        if over_limit_tools:
            # 触发断路器：有工具被调用超过限制，强制解绑工具
            over_limit_info = ", ".join([f"{name}({count}次)" for name, count in over_limit_tools.items()])
            logger.warning(
                f"[medical_agent] 触发断路器：工具 {over_limit_info} 已达到调用上限，"
                f"强制解绑工具以防死循环，总调用次数: {tool_call_counts}"
            )
        else:
            # 正常模式：加载工具列表
            use_tools = True
            prompt_key = Config.PROMPT_TEMPLATE_TXT_MEDICAL_AGENT
            medical_tools = tool_config.get_medical_tools()
            logger.info(
                f"[medical_agent] 可用工具: {[t.name for t in medical_tools]}"
            )
            
            
            medical_tools = tool_config.get_medical_tools()
            logger.info(f"Medical Agent 可用工具: {[t.name for t in medical_tools]}")
            llm_to_use = llm_chat  # 使用原始 LLM，不绑定工具
            use_tools = True
            if tool_call_counts:
                counts_info = ", ".join([f"{name}({count}次)" for name, count in tool_call_counts.items()])
                logger.info(f"[medical_agent] 本轮工具调用统计: {counts_info}")
            else:
                logger.info(f"[medical_agent] 本轮暂无工具调用，使用 JSON 格式")

        # 使用医疗专用 Prompt（从外部文件加载）
        if use_tools:
            prompt_content = load_prompt_template(Config.PROMPT_TEMPLATE_TXT_MEDICAL_AGENT)
        else:
            prompt_content = load_prompt_template(Config.PROMPT_TEMPLATE_TXT_MEDICAL_AGENT_CB)
        
        logger.debug(f"[medical_agent] Prompt content length: {len(prompt_content)}")
        logger.debug(f"[medical_agent] Prompt content preview: {prompt_content[:200]}")

        # ===== 修复：构建 prompt 时需要转换 ToolMessage =====
        # 评审意见参考：测试报告 test_retrieval_and_generation[suggestion] 失败分析
        messages_for_prompt = []
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                messages_for_prompt.append(msg)
            elif isinstance(msg, ToolMessage):
                # 将 ToolMessage 转换为 AIMessage，避免 LangChain 不支持 "tool" 类型
                messages_for_prompt.append(AIMessage(content=f"工具返回：{msg.content}"))
            else:
                # 其他类型消息转换为 HumanMessage
                messages_for_prompt.append(HumanMessage(content=str(msg.content) if hasattr(msg, "content") else str(msg)))

        logger.debug(f"[medical_agent] messages_for_prompt length: {len(messages_for_prompt)}")
        if messages_for_prompt:
            logger.debug(f"[medical_agent] Last message content preview: {str(messages_for_prompt[-1].content)[:100]}")

        medical_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_content),
            *[(msg.type if hasattr(msg, "type") else ("human" if isinstance(msg, HumanMessage) else "ai"), msg.content) for msg in messages_for_prompt],
            ("human", "{question}")
        ])

        medical_chain = medical_prompt | llm_to_use

        processed_turns = len([m for m in messages if isinstance(m, HumanMessage)])

        start_time = time.time()
        response = medical_chain.invoke({"question": question, "processed_turns": processed_turns})
        elapsed = time.time() - start_time

        # 解析 JSON 格式的工具调用
        if use_tools and hasattr(response, "content") and isinstance(response.content, str):
            tool_calls = parse_json_tool_call(response.content)
            if tool_calls:
                # 创建新的 AIMessage 包含 tool_calls
                from langchain_core.messages import AIMessage
                tool_call_message = AIMessage(
                    content=response.content,
                    tool_calls=tool_calls
                )
                response = tool_call_message

        # ===== Middleware: after_model =====
        if middleware_manager:
            after_updates = middleware_manager.run_after_model(state, response, node_name, elapsed)
            mw_updates.update(after_updates)

        return {"messages": [response], **mw_updates}

    except Exception as e:
        logger.error(f"Error in medical_agent: {e}", exc_info=True)
        return {"messages": state["messages"]}


def parse_json_tool_call(llm_output: str) -> Optional[List[Dict]]:
    """
    解析 LLM 输出的 JSON 格式工具调用。

    Args:
        llm_output: LLM 的原始输出文本。

    Returns:
        Optional[List[Dict]]: LangChain 格式的 tool_calls 列表，或 None。
    """
    import json
    import re
    
    try:
        # 提取 JSON 部分（更鲁棒的匹配）
        # 匹配从第一个 { 到最后一个 } 的内容
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if not json_match:
            return None
        
        json_str = json_match.group(0)
        # 清理多余的空格
        json_str = re.sub(r'\s+', ' ', json_str).strip()
        parsed = json.loads(json_str)
        
        # 检查格式
        if not isinstance(parsed, dict) or "tool_call" not in parsed:
            return None
        
        tool_call = parsed["tool_call"]
        if not isinstance(tool_call, dict) or "name" not in tool_call:
            return None
        
        # 转换为 LangChain 格式
        return [{
            "name": tool_call["name"],
            "args": tool_call.get("args", {}),
            "id": f"tool_{int(time.time() * 1000)}"
        }]
    except Exception as e:
        logger.error(f"Failed to parse JSON tool call: {e}")
        return None


def _count_tool_calls_in_turn(state: AgentState) -> dict:
    """
    统计当前用户回合每个工具的调用次数。

    逻辑：
    1. 倒序遍历 state["messages"] 找到最后一个 HumanMessage 的索引
    2. 统计该索引之后每个工具的调用次数

    Args:
        state: 当前对话状态。

    Returns:
        dict: 工具名称到调用次数的映射，例如 {"analyze_symptoms": 2, "health_record_retriever": 1}
    """

    messages = state.get("messages", [])
    if not messages:
        return {}

    # 倒序查找最后一个 HumanMessage 的索引
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            last_human_idx = i
            break

    if last_human_idx == -1:
        return {}

    # 统计每个工具的调用次数（仅统计 ToolMessage 实际执行结果，避免双重计数）
    tool_call_counts = {}
    for i in range(last_human_idx + 1, len(messages)):
        msg = messages[i]
        if isinstance(msg, ToolMessage) and hasattr(msg, "name"):
            tool_name = msg.name
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

    return tool_call_counts

def medical_analysis(
    state: AgentState,
    config: Optional[dict] = None,
    llm_chat: Optional[BaseChatModel] = None,
    middleware_manager: Optional[MiddlewareManager] = None
) -> dict:
    """
    医疗分析节点 - 综合分析医疗工具输出。

    职责：
    - 综合分析医疗工具输出
    - 生成结构化医疗分析报告
    - 设置 risk_level

    Args:
        state: 当前对话状态。
        llm_chat: Chat模型。
        middleware_manager: Middleware管理器。

    Returns:
        dict: 更新后的状态，包含 medical_analysis_result 和 risk_level。
    """
    logger.info("medical_analysis processing")
    node_name = "medical_analysis"
    mw_updates = {}

    try:
        if middleware_manager:
            before_updates, should_stop = middleware_manager.run_before_model(state, node_name)
            mw_updates.update(before_updates)
            if should_stop:
                return {"messages": state["messages"], **mw_updates}

        messages = state.get("messages", [])
        tool_outputs = []
        
        # ===== 修复：只收集当前轮次的工具输出，避免跨轮累积 =====
        current_turn_messages = _get_current_turn_messages(messages)
        
        for msg in current_turn_messages:
            if isinstance(msg, ToolMessage):
                tool_outputs.append(msg.content)
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                pass

        if not tool_outputs:
            logger.warning("未检测到医疗工具输出，检查 AI 是否已生成回复")
            last_ai_msg = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                    last_ai_msg = msg
                    break
            
            if last_ai_msg and last_ai_msg.content:
                logger.info("AI 已生成回复（询问更多信息），标记为 need_info")
                return {
                    "messages": state["messages"],
                    "medical_analysis_result": {
                        "status": "need_info",
                        "summary": f"AI 已请求更多信息: {last_ai_msg.content[:200]}"
                    },
                    "risk_level": "low",
                    **mw_updates
                }
            else:
                logger.warning("未检测到 AI 回复，使用默认分析结果")
                return {
                    "messages": state["messages"],
                    "medical_analysis_result": {
                        "status": "no_tools",
                        "summary": "未调用医疗分析工具"
                    },
                    "risk_level": "low",
                    **mw_updates
                }

        combined_output = "\n\n".join(tool_outputs)

        # 从外部文件加载 prompt 模板
        prompt_content = load_prompt_template(Config.PROMPT_TEMPLATE_TXT_MEDICAL_ANALYSIS)
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_content),
            ("human", "{tool_outputs}")
        ])

        analysis_chain = analysis_prompt | llm_chat

        start_time = time.time()
        response = analysis_chain.invoke({"tool_outputs": combined_output})
        elapsed = time.time() - start_time

        try:
            response_text = response.content if hasattr(response, "content") else str(response)
            
            json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*?\}))*?\}', response_text)
            
            if json_match:
                analysis_result = json.loads(json_match.group())
                # 移除 departments 字段，专注于医疗分析
                if "departments" in analysis_result:
                    del analysis_result["departments"]
            else:
                analysis_result = {
                    "summary": response_text[:500],
                    "risk_level": "low"
                }
        except Exception as e:
            logger.warning(f"解析医疗分析结果失败: {e}")
            analysis_result = {
                "summary": combined_output[:500],
                "risk_level": "low"
            }

        risk_level = analysis_result.get("risk_level", "low")
        if risk_level not in ["low", "medium", "high", "critical"]:
            risk_level = "low"

        if middleware_manager:
            after_updates = middleware_manager.run_after_model(state, response, node_name, elapsed)
            mw_updates.update(after_updates)

        logger.info(f"医疗分析完成: 风险等级={risk_level}")

        return {
            "messages": state["messages"],
            "medical_analysis_result": analysis_result,
            "risk_level": risk_level,
            **mw_updates
        }

    except Exception as e:
        logger.error(f"Error in medical_analysis: {e}", exc_info=True)
        return {
            "messages": state["messages"],
            "medical_analysis_result": {
                "status": "error",
                "summary": f"分析异常: {str(e)}"
            },
            "risk_level": "high",
            "mw_force_stop": True
        }

def department_triage(
    state: AgentState,
    config: Optional[dict] = None,
    llm_chat: Optional[BaseChatModel] = None,
    middleware_manager: Optional[MiddlewareManager] = None
) -> dict:
    """
    科室分诊节点 - 实现真实分诊逻辑。

    职责：
    - 根据医疗分析结果推荐科室
    - 判断紧急度
    - 生成分诊理由
    - 科室白名单验证

    Args:
        state: 当前对话状态。
        llm_chat: Chat模型。
        middleware_manager: Middleware管理器。

    Returns:
        dict: 更新后的状态，包含 recommended_departments、urgency_level、triage_reason。
    """
    logger.info("department_triage processing")
    node_name = "department_triage"
    mw_updates = {}

    # ===== 扩展科室白名单 + 模糊匹配映射表 =====
    # 问题：模型推荐的科室名称与白名单不完全匹配（如'心血管内科' vs '心内科'）
    # 解决：1. 扩展白名单覆盖更多常用科室 2. 使用映射表进行模糊匹配
    VALID_DEPARTMENTS = {
        # 内科系统
        "血液科", "感染科", "消化内科", "心内科", "心血管内科", "呼吸内科",
        "神经内科", "内分泌科", "肾内科", "风湿免疫科", "肿瘤内科",
        # 外科系统
        "急诊科", "普通外科", "肝胆外科", "肝胆胰外科", "胰腺外科", 
        "骨科", "泌尿外科", "神经外科", "心胸外科", "胃肠外科",
        # 妇儿专科
        "妇科", "产科", "儿科", "小儿外科", "儿童保健科",
        # 其他专科
        "皮肤科", "眼科", "耳鼻喉科", "口腔科", "精神科", "康复科", "康复医学科",
        "中医科", "中西医结合科", "营养科", "疼痛科", "体检中心", "全科医学科", "全科门诊"
    }

    # 科室名称模糊匹配映射表（模型推荐 → 白名单标准名称）
    DEPARTMENT_MAPPING = {
        # 心血管系统
        "心血管内科": "心内科",
        "心脏内科": "心内科",
        # 肝胆胰系统
        "肝胆胰外科": "肝胆外科",
        "肝胆外科": "肝胆外科",
        "胰腺外科": "肝胆外科",
        "肝病科": "肝胆外科",
        # 营养相关
        "营养科": "营养科",
        # 康复相关
        "康复医学科": "康复科",
        "康复治疗科": "康复科",
        # 其他常见映射
        "消化外科": "普通外科",
        "胃肠外科": "普通外科",
        "甲状腺外科": "普通外科",
        "乳腺外科": "普通外科",
    }

    def validate_department(dept: str) -> str:
        """
        科室白名单验证 + 模糊匹配。
        
        逻辑：
        1. 精确匹配：直接在白名单中查找
        2. 模糊匹配：通过 DEPARTMENT_MAPPING 映射表转换
        3. 关键词匹配：包含白名单科室关键词（如'内科'、'外科'等）
        4. 降级处理：无法匹配时返回'全科医学科'
        """
        # 移除括号内的补充说明（如'（兼顾腰椎术后管理）'）
        dept_clean = re.sub(r'[\（\(][^)\）]*[\）\)]', '', dept).strip()
        
        # 1. 精确匹配
        if dept_clean in VALID_DEPARTMENTS:
            return dept_clean
        
        # 2. 模糊匹配（通过映射表）
        if dept_clean in DEPARTMENT_MAPPING:
            mapped = DEPARTMENT_MAPPING[dept_clean]
            logger.info(f"[department_triage] 科室 '{dept_clean}' 映射为 '{mapped}'")
            return mapped
        
        # 3. 关键词匹配（兜底策略）
        # 检查是否包含白名单科室的关键词
        for valid_dept in sorted(VALID_DEPARTMENTS, key=len, reverse=True):
            if len(valid_dept) >= 2 and valid_dept in dept_clean:
                logger.info(f"[department_triage] 科室 '{dept_clean}' 匹配到 '{valid_dept}'（关键词匹配）")
                return valid_dept
        
        # 4. 降级处理
        logger.warning(f"科室 '{dept}' (clean: '{dept_clean}') 无法匹配，降级为全科医学科")
        return "全科医学科"

    try:
        if middleware_manager:
            before_updates, should_stop = middleware_manager.run_before_model(state, node_name)
            mw_updates.update(before_updates)
            if should_stop:
                return {"messages": state["messages"], **mw_updates}

        medical_analysis_result = state.get("medical_analysis_result", {})
        risk_level = state.get("risk_level", "low")

        analysis_error = False
        need_info_mode = False
        if not medical_analysis_result:
            logger.warning("medical_analysis_result 为空，触发悲观降级")
            analysis_error = True
        elif medical_analysis_result.get("status") == "error":
            logger.warning(f"medical_analysis_result 状态异常: {medical_analysis_result.get('status')}，触发悲观降级")
            analysis_error = True
        elif medical_analysis_result.get("status") == "skeleton":
            logger.warning("medical_analysis_result 状态为 skeleton，触发悲观降级")
            analysis_error = True
        elif medical_analysis_result.get("status") == "no_tools":
            logger.info("medical_analysis_result 状态为 no_tools，LLM 可能正在询问更多信息，使用中等风险")
            need_info_mode = True
        elif medical_analysis_result.get("status") == "need_info":
            logger.info("medical_analysis_result 状态为 need_info，AI 正在询问更多信息")
            need_info_mode = True
        elif "分析异常" in str(medical_analysis_result.get("summary", "")):
            logger.warning("medical_analysis_result 包含错误信息，触发悲观降级")
            analysis_error = True

        if analysis_error:
            return {
                "messages": state["messages"],
                "recommended_departments": ["急诊科"],
                "urgency_level": "emergency",
                "triage_reason": "系统分析异常，为安全起见建议紧急就诊",
                "triage_confidence": 0.5,
                **mw_updates
            }

        if need_info_mode:
            return {
                "messages": state["messages"],
                "recommended_departments": ["全科医学科"],
                "urgency_level": "routine",
                "triage_reason": medical_analysis_result.get("summary", "需要更多信息进行诊断"),
                "triage_confidence": 0.7,
                **mw_updates
            }

        # 从医疗分析结果中提取科室推荐，如果没有则基于症状和分析结果生成
        departments = medical_analysis_result.get("departments", [])
        
        # 如果没有科室推荐，基于分析结果生成分科室建议
        if not departments:
            # 基于分析结果和风险等级生成分科室建议
            summary = medical_analysis_result.get("summary", "")
            
            # 简单的科室推荐逻辑，基于分析结果中的关键词
            recommended_depts = []
            
            # 心血管相关
            if any(keyword in summary for keyword in ["血压", "心率", "心脏", "心血管"]):
                recommended_depts.append("心内科")
            
            # 呼吸相关
            if any(keyword in summary for keyword in ["发热", "咳嗽", "呼吸", "肺部"]):
                recommended_depts.append("呼吸内科")
            
            # 消化相关
            if any(keyword in summary for keyword in ["腹痛", "恶心", "呕吐", "消化"]):
                recommended_depts.append("消化内科")
            
            # 神经相关
            if any(keyword in summary for keyword in ["头痛", "头晕", "神经"]):
                recommended_depts.append("神经内科")
            
            # 内分泌相关
            if any(keyword in summary for keyword in ["血糖", "甲状腺", "内分泌"]):
                recommended_depts.append("内分泌科")
            
            # 肾相关
            if any(keyword in summary for keyword in ["肾功能", "尿液", "肾脏"]):
                recommended_depts.append("肾内科")
            
            # 如果没有推荐科室，默认全科医学科
            if not recommended_depts:
                recommended_depts = ["全科医学科"]
            
            departments = recommended_depts

        validated_departments = [validate_department(d) for d in departments]
        validated_departments = list(dict.fromkeys(validated_departments))

        # ===== 修复：统一紧急度判定逻辑 =====
        # 评审意见参考：评估.md 问题4修复方案
        
        # 1. 中英文枚举统一映射
        URGENCY_CN_TO_EN = {
            "低": "routine",
            "中": "urgent", 
            "高": "urgent",
            "极高": "emergency"
        }
        
        # 2. 从工具输出中提取 urgency_level
        tool_urgency = None
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                try:
                    result = json.loads(msg.content)
                    if "urgency_level" in result:
                        tool_urgency_cn = result["urgency_level"]
                        # 统一转换为英文枚举
                        tool_urgency = URGENCY_CN_TO_EN.get(tool_urgency_cn, "routine")
                        logger.info(f"[department_triage] 工具紧急度 '{tool_urgency_cn}' 转换为 '{tool_urgency}'")
                        break
                except (json.JSONDecodeError, AttributeError):
                    pass
        
        # 3. 根据 risk_level 计算规则紧急度
        URGENCY_RULES = {
            "critical": "emergency",
            "high": "urgent",
            "medium": "urgent",
            "low": "routine"
        }
        rule_urgency = URGENCY_RULES.get(risk_level, "routine")  # 默认改为 routine
        
        # 4. 冲突解决：规则优先级高于工具输出
        urgency_priority = {"emergency": 3, "urgent": 2, "routine": 1}
        
        if tool_urgency:
            # 如果规则紧急度更高，使用规则
            if urgency_priority.get(rule_urgency, 0) > urgency_priority.get(tool_urgency, 0):
                urgency_level = rule_urgency
                logger.info(f"[department_triage] 规则紧急度 {rule_urgency} 覆盖工具紧急度 {tool_urgency}")
            else:
                urgency_level = tool_urgency
                logger.info(f"[department_triage] 使用工具紧急度: {tool_urgency}")
        else:
            urgency_level = rule_urgency
            logger.info(f"[department_triage] 使用规则紧急度: {rule_urgency}")

        triage_reason = medical_analysis_result.get("summary", "根据医疗分析结果推荐科室")

        if middleware_manager:
            after_updates = middleware_manager.run_after_model(state, None, node_name, 0.0)
            mw_updates.update(after_updates)

        logger.info(f"分诊结果: 科室={validated_departments}, 紧急度={urgency_level}, 风险={risk_level}")

        return {
            "messages": state["messages"],
            "recommended_departments": validated_departments,
            "urgency_level": urgency_level,
            "triage_reason": triage_reason,
            "triage_confidence": 0.85,
            **mw_updates
        }

    except Exception as e:
        logger.error(f"Error in department_triage: {e}", exc_info=True)
        # 悲观降级：异常时默认高风险 + 紧急就诊 + 急诊科
        return {
            "messages": state["messages"],
            "recommended_departments": ["急诊科"],
            "urgency_level": "emergency",
            "triage_reason": f"分诊异常: {str(e)}，为安全起见建议紧急就诊",
            "triage_confidence": 0.3
        }

def medical_safety_guard(
    state: AgentState,
    config: Optional[dict] = None,
    middleware_manager: Optional[MiddlewareManager] = None
) -> dict:
    """
    医疗安全守门节点 - 实现真实安全防线。

    职责：
    - 风险识别与拦截
    - 强制添加免责声明
    - 高风险强制覆盖
    - 生成 final_payload

    Args:
        state: 当前对话状态。
        middleware_manager: Middleware管理器。

    Returns:
        dict: 更新后的状态，包含 final_payload。
    """
    logger.info("medical_safety_guard processing")
    node_name = "medical_safety_guard"
    mw_updates = {}

    try:
        if middleware_manager:
            before_updates, should_stop = middleware_manager.run_before_model(state, node_name)
            mw_updates.update(before_updates)
            if should_stop:
                return {"messages": state["messages"], **mw_updates}

        risk_level = state.get("risk_level", "low")
        recommended_departments = state.get("recommended_departments", ["全科医学科"])
        urgency_level = state.get("urgency_level", "routine")
        triage_reason = state.get("triage_reason", "")
        medical_analysis_result = state.get("medical_analysis_result", {})

        analysis_error = False
        need_info_mode = False
        if not medical_analysis_result:
            logger.warning("[medical_safety_guard] medical_analysis_result 为空，触发悲观降级")
            analysis_error = True
        elif medical_analysis_result.get("status") == "error":
            logger.warning(f"[medical_safety_guard] medical_analysis_result 状态异常: {medical_analysis_result.get('status')}")
            analysis_error = True
        elif medical_analysis_result.get("status") == "skeleton":
            logger.warning("[medical_safety_guard] medical_analysis_result 状态为 skeleton，触发悲观降级")
            analysis_error = True
        elif medical_analysis_result.get("status") == "no_tools":
            logger.info("[medical_safety_guard] medical_analysis_result 状态为 no_tools，LLM 可能正在询问更多信息")
            need_info_mode = True
        elif medical_analysis_result.get("status") == "need_info":
            logger.info("[medical_safety_guard] medical_analysis_result 状态为 need_info，AI 正在询问更多信息")
            need_info_mode = True
        elif "分析异常" in str(medical_analysis_result.get("summary", "")):
            logger.warning("[medical_safety_guard] medical_analysis_result 包含错误信息")
            analysis_error = True

        if analysis_error:
            risk_level = "high"
            recommended_departments = ["急诊科"]
            urgency_level = "emergency"
            triage_reason = "系统分析异常，为安全起见建议紧急就诊"

        if need_info_mode:
            risk_level = "low"
            recommended_departments = ["全科医学科"]
            urgency_level = "routine"
            triage_reason = medical_analysis_result.get("summary", "需要更多信息进行诊断")

        messages = state.get("messages", [])
        answer = ""
        answer_source = "unknown"
        
        # ===== 修复：从多个来源提取 answer =====
        # 评审意见参考：评估.md 问题3修复方案
        
        # 1. 尝试从最近的 AIMessage（非 tool_calls）获取
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                # 如果有 tool_calls 但没有 content，跳过
                if hasattr(msg, "tool_calls") and msg.tool_calls and not msg.content:
                    continue
                if msg.content:
                    answer = msg.content
                    answer_source = "ai_message"
                    logger.info(f"[medical_safety_guard] 从 AIMessage 获取 answer: {answer[:100]}...")
                    break
        
        # 2. 如果 answer 仍为空，从 medical_analysis_result 生成
        if not answer and medical_analysis_result:
            summary = medical_analysis_result.get("summary", "")
            
            answer_parts = []
            if summary:
                answer_parts.append(summary)
            
            answer = "\n".join(answer_parts)
            answer_source = "fallback_from_analysis_result"
            
            # ===== 新增：监控告警 =====
            logger.warning(
                "[medical_safety_guard] ⚠️ answer 使用兜底逻辑生成，"
                "说明 medical_agent 未能正常生成自然语言回复，"
                "请检查 filter_messages 和工具调用链路"
            )
            if middleware_manager:
                middleware_manager.record_metric("answer_fallback_triggered", 1)
        
        # 3. 最后兜底：使用工具输出
        if not answer:
            tool_outputs = []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tool_outputs.append(msg.content)
            if tool_outputs:
                answer = "根据分析结果：" + tool_outputs[0][:500]
                answer_source = "fallback_from_tool_message"
                logger.info(f"[medical_safety_guard] 从 ToolMessage 获取 answer")
        
        logger.info(f"[medical_safety_guard] answer 来源: {answer_source}")

        risk_warning = ""
        if risk_level in ["high", "critical"]:
            risk_warning = "⚠️ 警告：检测到高危指标，请立即前往急诊科或拨打120就诊！"
        elif analysis_error:
            risk_warning = "⚠️ 系统分析异常，为安全起见建议紧急就诊！"

        disclaimer = "\n\n⚠️ 以上分析仅供参考，请以医生面诊为准。"

        final_answer = answer
        if risk_warning:
            final_answer = f"{risk_warning}\n\n{answer}"
        final_answer += disclaimer

        final_payload = {
            "route": "medical",
            "answer": final_answer,
            "structured_data": {
                "analysis": medical_analysis_result,
                "triage": {
                    "recommended_departments": recommended_departments,
                    "urgency_level": urgency_level,
                    "triage_reason": triage_reason,
                    "triage_confidence": state.get("triage_confidence", 0.8)
                }
            },
            "risk_warning": risk_warning if risk_warning else "无高危风险",
            "risk_level": risk_level,
            "disclaimer": "以上分析仅供参考，请以医生面诊为准。"
        }

        if middleware_manager:
            after_updates = middleware_manager.run_after_model(state, None, node_name, 0.0)
            mw_updates.update(after_updates)

        logger.info(f"安全守卫完成: 风险等级={risk_level}, 紧急度={urgency_level}")

        # 当风险等级为critical时，异步调用飞书MCP存储风险信息
        if risk_level == "critical":
            # 检查飞书 MCP 是否已初始化
            if not feishu_mcp_manager or not feishu_mcp_manager.is_initialized():
                logger.warning("飞书 MCP 未初始化，跳过 critical 风险记录")
            else:
                async def save_to_feishu():
                    """异步保存风险信息到飞书多维表格"""
                    try:
                        # 获取用户 ID
                        user_id = state.get("user_id", "unknown")
                        # 构建风险原因
                        risk_reason = f"紧急度：{urgency_level}\n推荐科室：{', '.join(recommended_departments)}\n原因：{triage_reason}"
                        
                        # 调用飞书 MCP 管理器
                        success = feishu_mcp_manager.add_critical_risk_record(
                            user_id=user_id,
                            risk_reason=risk_reason
                        )
                        if success:
                            logger.info(f"成功将 critical 风险记录保存到飞书多维表格")
                        else:
                            logger.warning("保存 critical 风险记录到飞书多维表格失败")
                    except Exception as e:
                        logger.error(f"保存风险记录到飞书失败：{str(e)}")
                
                def sync_save_to_feishu():
                    """在线程中运行异步飞书保存操作"""
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(save_to_feishu())
                    finally:
                        loop.close()

                executor = ThreadPoolExecutor(max_workers=1)
                executor.submit(sync_save_to_feishu)
                logger.info("已启动线程池任务，将 critical 风险记录保存到飞书多维表格")

        token_slim_updates = {
            "medical_context": "",
            "retrieval_context": "",
        }
        mw_updates.update(token_slim_updates)
        logger.debug("[medical_safety_guard] Token 瘦身: 已清理 medical_context 和 retrieval_context")

        return {
            "final_payload": final_payload,
            "risk_level": risk_level,
            "recommended_departments": recommended_departments,
            "urgency_level": urgency_level,
            **mw_updates
        }

    except Exception as e:
        logger.error(f"Error in medical_safety_guard: {e}", exc_info=True)
        # 悲观降级：异常时默认高风险 + 紧急就诊 + 急诊科
        return {
            "final_payload": {
                "route": "medical",
                "answer": "处理医疗分析时发生错误，为安全起见建议紧急就诊。",
                "risk_warning": "⚠️ 系统异常，建议紧急就诊！",
                "risk_level": "high",
                "recommended_departments": ["急诊科"],
                "urgency_level": "emergency",
                "disclaimer": "以上分析仅供参考，请以医生面诊为准。"
            }
        }

# 定义路由函数，用于根据工具调用的结果动态决定下一步路由。
# 它会根据状态中的消息历史和工具调用结果，动态路由到生成节点或文档评分节点。
def original_route_after_tools(state: AgentState, tool_config: ToolConfig) -> Literal["generate", "grade_documents"]:
    """根据工具调用的结果动态决定下一步路由，使用配置字典支持多工具并包含容错处理。

    Args:
        state: 当前对话状态，包含消息历史和可能的工具调用结果。
        tool_config: 工具配置参数。

    Returns:
        Literal["generate", "grade_documents"]: 下一步的目标节点。
    """
    if not state.get("messages") or not isinstance(state["messages"], list):
        logger.error("Messages state is empty or invalid, defaulting to generate")
        return "generate"

    try:
        last_message = state["messages"][-1]
        msg_type = type(last_message).__name__
        msg_name = getattr(last_message, "name", None)
        
        logger.info(f"original_route_after_tools: last_message type={msg_type}, name={msg_name}")

        if not hasattr(last_message, "name"):
            logger.info("Last message has no name attribute, routing to generate")
            return "generate"
        
        if last_message.name is None:
            logger.warning("ToolMessage name is None, checking if it's a ToolMessage")
            from langchain_core.messages import ToolMessage
            if isinstance(last_message, ToolMessage):
                logger.info("Confirmed ToolMessage with None name, routing to grade_documents (RAG pathway)")
                return "grade_documents"
            logger.info("Non-ToolMessage with None name, routing to generate")
            return "generate"

        tool_name = last_message.name
        domain = state.get("route_domain", "general")
        if domain == "general":
            tool_names = tool_config.rag_tool_names
            routing_config = tool_config.rag_routing_config
        else:
            tool_names = tool_config.medical_tool_names
            routing_config = tool_config.medical_routing_config

        logger.info(f"tool_name={tool_name}, domain={domain}, rag_tool_names={tool_names}")

        if tool_name not in tool_names:
            logger.info(f"Unknown tool {tool_name}, routing to generate")
            return "generate"

        target = routing_config.get(tool_name, "generate")
        logger.info(f"Tool {tool_name} routed to {target} based on config")
        return target

    except IndexError:
        logger.error("No messages available in state, defaulting to generate")
        return "generate"
    except AttributeError:
        logger.error("Invalid message object, defaulting to generate")
        return "generate"
    except Exception as e:
        logger.error(f"Unexpected error in route_after_tools: {e}, defaulting to generate")
        return "generate"

def global_route_after_tools(state: AgentState, tool_config: ToolConfig) -> Literal["generate", "grade_documents", "medical_agent"]:
    """
    全局工具路由函数 - 根据 route_domain 决定下一步路由。

    Args:
        state: 当前对话状态。
        tool_config: 工具_config: 工具配置参数。

    Returns:
        Literal["generate", "grade_documents", "medical_agent"]: 下一步的目标节点。
    """
    domain = state.get("route_domain", "general")
    
    logger.info(f"global_route_after_tools: route_domain={domain}, messages count={len(state.get('messages', []))}")
    
    if domain == "medical":
        logger.info("Medical pathway: returning to medical_agent for continued reasoning")
        return "medical_agent"
    else:
        logger.info("General pathway: using original routing logic")
        result = original_route_after_tools(state, tool_config)
        logger.info(f"original_route_after_tools returned: {result}")
        return result

def route_after_intent(state: AgentState) -> Literal["rag_agent", "medical_agent"]:
    """
    根据 route_domain 决定下一步路由。

    Args:
        state: 当前对话状态。

    Returns:
        Literal["rag_agent", "medical_agent"]: 下一步的目标节点。
    """
    route_domain = state.get("route_domain")
    logger.info(f"Routing after intent_router: {route_domain}")
    
    if route_domain == "medical":
        return "medical_agent"
    return "rag_agent"

def route_after_medical_agent(state: AgentState) -> Literal["call_tools", "medical_analysis"]:
    """
    根据 medical_agent 的输出决定下一步路由。

    增强功能：
    - 工具调用去重：如果本次想调用的工具和参数与上一次完全一致，强制阻断

    Args:
        state: 当前对话状态。

    Returns:
        Literal["call_tools", "medical_analysis"]: 下一步的目标节点。
    """
    if not state.get("messages"):
        logger.warning("No messages in state, routing to medical_analysis")
        return "medical_analysis"

    last_message = state["messages"][-1]
    
    # 如果有 tool_calls，检查是否重复
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # ===== 去重拦截机制 =====
        current_tool_calls = last_message.tool_calls
        
        # 查找上一次的 tool_calls（倒数第二条 AIMessage）
        for i in range(len(state["messages"]) - 2, -1, -1):
            prev_msg = state["messages"][i]
            if hasattr(prev_msg, "tool_calls") and prev_msg.tool_calls:
                prev_tool_calls = prev_msg.tool_calls
                
                # 检查是否完全相同
                if _are_tool_calls_identical(current_tool_calls, prev_tool_calls):
                    logger.warning(
                        f"[route_after_medical_agent] 检测到重复工具调用: "
                        f"{[tc.get('name', tc) for tc in current_tool_calls]}，强制跳转到 medical_analysis"
                    )
                    return "medical_analysis"
                break
        
        logger.info("Medical agent has tool calls, routing to call_tools")
        return "call_tools"
    
    # 否则进入医疗分析阶段
    logger.info("Medical agent has no tool calls, routing to medical_analysis")
    return "medical_analysis"


def _are_tool_calls_identical(tc1: list, tc2: list) -> bool:
    """
    比较两组 tool_calls 是否完全相同（忽略 tool_call_id）。
    
    修复问题：
    - tool_call_id 每次生成都不同，导致去重失败
    
    评审意见参考：评估.md 问题2修复方案

    Args:
        tc1: 第一组 tool_calls。
        tc2: 第二组 tool_calls。

    Returns:
        bool: 是否完全相同（name 和 args 相同）。
    """
    if len(tc1) != len(tc2):
        return False
    
    for call1, call2 in zip(tc1, tc2):
        name1 = call1.get("name", "") if isinstance(call1, dict) else getattr(call1, "name", "")
        name2 = call2.get("name", "") if isinstance(call2, dict) else getattr(call2, "name", "")
        
        if name1 != name2:
            return False
        
        args1 = call1.get("args", {}) if isinstance(call1, dict) else getattr(call1, "args", {})
        args2 = call2.get("args", {}) if isinstance(call2, dict) else getattr(call2, "args", {})
        
        if json.dumps(args1, sort_keys=True) != json.dumps(args2, sort_keys=True):
            return False
    
    return True


def route_after_grade(state: AgentState) -> Literal["generate", "rewrite"]:
    """根据状态中的评分结果决定下一步路由，包含增强的状态校验和容错处理。

    Args:
        state: 当前对话状态，预期包含 messages 和 relevance_score 字段。

    Returns:
        Literal["generate", "rewrite"]: 下一步的目标节点。
    """
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to rewrite")
        return "rewrite"

    # 检查状态是否包含 messages 字段，若缺失则记录错误并默认路由到 rewrite
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to rewrite")
        return "rewrite"

    # 检查 messages 是否为空，若为空则记录警告并默认路由到 rewrite
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to rewrite")
        return "rewrite"

    # 获取状态中的 relevance_score，若不存在则返回 None
    relevance_score = state.get("relevance_score")
    # 获取状态中的 rewrite_count
    rewrite_count = state.get("rewrite_count", 0)
    logger.info(f"Routing based on relevance_score: {relevance_score}, rewrite_count: {rewrite_count}")

    # 如果重写次数超过 3 次，强制路由到 generate
    if rewrite_count >= 3:
        logger.info("Max rewrite limit reached, proceeding to generate")
        return "generate"

    try:
        # 检查 relevance_score 是否为有效字符串，若不是则视为无效评分
        if not isinstance(relevance_score, str):
            logger.warning(f"Invalid relevance_score type: {type(relevance_score)}, defaulting to rewrite")
            return "rewrite"

        # 如果评分结果为 "yes"，表示文档相关，路由到 generate 节点
        if relevance_score.lower() == "yes":
            logger.info("Documents are relevant, proceeding to generate")
            return "generate"

        # 如果评分结果为 "no" 或其他值（包括空字符串），路由到 rewrite 节点
        logger.info("Documents are not relevant or scoring failed, proceeding to rewrite")
        return "rewrite"

    except AttributeError:
        # 捕获 relevance_score 不支持 lower() 方法的异常（例如 None），默认路由到 rewrite
        logger.error("relevance_score is not a string or is None, defaulting to rewrite")
        return "rewrite"
    except Exception as e:
        # 捕获其他未预期的异常，记录详细错误信息并默认路由到 rewrite
        logger.error(f"Unexpected error in route_after_grade: {e}, defaulting to rewrite")
        return "rewrite"


from utils.middleware import (
    LoggingMiddleware,
    PIIDetectionMiddleware,
    SummarizationMiddleware,
    ToolRetryMiddleware,
)

def _build_workflow_graph(llm_chat, llm_embedding, tool_config: ToolConfig, middleware_manager: MiddlewareManager, store: BaseStore) -> StateGraph:
    """构建工作流图的核心逻辑。

    Args:
        llm_chat: Chat模型。
        llm_embedding: Embedding模型。
        tool_config: 工具配置参数。
        middleware_manager: Middleware管理器。
        store: 存储实例（PostgresStore 或 InMemoryStore）。

    Returns:
        StateGraph: 配置好的工作流图（未编译）。
        
        构建工作流图核心逻辑。

    拓扑结构：
        START → intent_router
            ├─→ rag_agent → call_tools → grade_documents → generate → END
            │                                           └─→ rewrite ↩
            └─→ medical_agent → medical_analysis → department_triage
                                                → medical_safety_guard → END
    """
    from langgraph.prebuilt import tools_condition
    
    workflow = StateGraph(AgentState, context_schema=Context)
    
    #  节点工厂：统一依赖注入，异常处理
    def make_node(fn, **kwargs):
        """
        包装节点函数，注入固定依赖并捕获异常。
        
        Args:
            fn: 原始节点函数
            **kwargs: 需要注入的依赖
        """
        def node(state: AgentState, config=None):
            try:
                return fn(state, config, **kwargs) if config is not None else fn(state, **kwargs)
            except RagAgentError:
                raise   # 业务异常直接上抛
            except Exception as e:
                logger.error(f"节点 '{fn.__name__}' 执行异常: {e}", exc_info=True)
                raise GraphBuildError(
                    f"节点执行失败: {fn.__name__}",
                    details={"error": str(e)}
                )
        node.__name__ = fn.__name__  # 保留函数名，便于日志追踪
        return node
    
    #  注册节点
    workflow.add_node("intent_router", make_node(
        intent_router,
        llm_chat=llm_chat,
        middleware_manager=middleware_manager,
    ))
    workflow.add_node("rag_agent", make_node(
        agent,
        llm_chat=llm_chat,
        tool_config=tool_config,
        store=store,
        middleware_manager=middleware_manager,
    ))
    workflow.add_node("medical_agent", make_node(
        medical_agent,
        llm_chat=llm_chat,
        tool_config=tool_config,
        store=store,
        middleware_manager=middleware_manager,
    ))
    workflow.add_node("medical_analysis", make_node(
        medical_analysis,
        llm_chat=llm_chat,
        middleware_manager=middleware_manager,
    ))
    workflow.add_node("department_triage", make_node(
        department_triage,
        llm_chat=llm_chat,
        middleware_manager=middleware_manager,
    ))
    workflow.add_node("medical_safety_guard", make_node(
        medical_safety_guard,
        middleware_manager=middleware_manager,
    ))
    workflow.add_node("call_tools", ParallelToolNode(
        tool_config.get_medical_tools(),  
        middleware_manager=middleware_manager,
    ))
    workflow.add_node("rewrite", make_node(
        rewrite,
        llm_chat=llm_chat,
        middleware_manager=middleware_manager,
    ))
    workflow.add_node("generate", make_node(
        generate,
        llm_chat=llm_chat,
        store=store,
        middleware_manager=middleware_manager,
    ))
    workflow.add_node("grade_documents", make_node(
        grade_documents,
        llm_chat=llm_chat,
        middleware_manager=middleware_manager,
    ))
    
    
    # ===== 新增拓扑连接 =====
    # START → intent_router
    workflow.add_edge(START, "intent_router")
    
    # intent_router → 条件路由（general / medical）
    workflow.add_conditional_edges(
        source="intent_router",
        path=route_after_intent,
        path_map={"rag_agent":     "rag_agent", 
                  "medical_agent": "medical_agent"}
    )
    
    # 通用rag线路
    workflow.add_conditional_edges(
        source="rag_agent", 
        path=tools_condition, 
        path_map={"tools": "call_tools", END: END})
    
    workflow.add_conditional_edges(
        source="call_tools", 
        path=lambda state: global_route_after_tools(state, tool_config), 
        path_map={"generate": "generate", 
                  "grade_documents": "grade_documents", 
                  "medical_agent": "medical_agent"}) # 医疗线路如果判定为医疗咨询，则跳转到专门的医疗智能体处理（体现了多智能体协作的架构）。
    
    workflow.add_conditional_edges(
        source="grade_documents", 
        path=route_after_grade, path_map={"generate": "generate", 
                                          "rewrite": "rewrite"})
    
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "rag_agent")
    
    # 医疗线路
    # medical_agent → call_tools（医疗工具调用）
    workflow.add_conditional_edges(source="medical_agent",
                                   path=route_after_medical_agent,
                                   path_map={"call_tools": "call_tools",
                                             "medical_analysis": "medical_analysis"})
    
    # medical_analysis → department_triage → medical_safety_guard → END
    workflow.add_edge("medical_analysis", "department_triage")
    workflow.add_edge("department_triage", "medical_safety_guard")
    workflow.add_edge("medical_safety_guard", END)
    
    return workflow

# ===== 全局连接池（避免重复创建）=====
def get_connection_pool(db_uri: str):
    from psycopg_pool import ConnectionPool
    logger.info("Initializing PostgreSQL connection pool...")
    return ConnectionPool(db_uri, min_size=1, max_size=10)


# ===== Middleware 初始化 =====
def init_middleware():
    return MiddlewareManager([
        LoggingMiddleware(),
        PIIDetectionMiddleware(mode=Config.MW_PII_MODE),
        SummarizationMiddleware(
            max_messages=Config.MW_SUMMARIZATION_THRESHOLD,
            keep_recent=Config.MW_SUMMARIZATION_KEEP_RECENT
        ),
        ToolRetryMiddleware(
            max_retries=Config.MW_TOOL_MAX_RETRIES,
            backoff_factor=Config.MW_TOOL_BACKOFF_FACTOR
        ),
    ])


# ===== PostgreSQL 存储初始化 =====
def init_postgres_store(db_uri: str, llm_embedding) -> Tuple:
    try:
        import psycopg2
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore

        # --- 连接测试（带超时）---
        psycopg2.connect(db_uri, connect_timeout=3).close()
        logger.info("✓ PostgreSQL connection successful")

        # --- 获取连接池（单例）---
        pool = get_connection_pool(db_uri)

        # --- 自动获取 embedding 维度 ---
        embedding_dims = getattr(llm_embedding, "dimension", None)
        if embedding_dims is None:
            logger.warning("Embedding dimension not found, fallback to Config")
            embedding_dims = Config.EMBEDDING_DIMS

        # --- 初始化 ---
        checkpointer = PostgresSaver(pool)
        store = PostgresStore(
            pool,
            index={
                "dims": embedding_dims,
                "embed": llm_embedding
            }
        )

        checkpointer.setup()
        store.setup()

        logger.info("PostgresSaver and PostgresStore initialized successfully")

        return checkpointer, store

    except ImportError as e:
        logger.error(f"Missing PostgreSQL dependencies: {e}")
        raise

    except Exception as e:
        logger.error(f"PostgreSQL initialization failed: {e}")
        raise
    

# ===== Memory 存储初始化（fallback）=====
def init_memory_store():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore

    logger.warning("Using in-memory storage (fallback mode)")

    return MemorySaver(), InMemoryStore()


def create_graph(llm_chat, llm_embedding, tool_config: ToolConfig) -> StateGraph:
    """创建并配置状态图。
    
    - agent 节点(state, config)，通过 config 传递上下文
    
    Args:
        llm_chat: Chat模型。
        llm_embedding: Embedding模型。
        tool_config: 工具配置参数。

    Returns:
        StateGraph: 编译后的状态图。

    Raises:
        Exception: 如果初始化失败。
    """
    DB_URI = Config.DB_URI
    
    # ===== 初始化 Middleware 管理器（全局唯一，但无可变状态） =====
    middleware_manager = MiddlewareManager([
        LoggingMiddleware(),
        PIIDetectionMiddleware(mode=Config.MW_PII_MODE),
        SummarizationMiddleware(
            max_messages=Config.MW_SUMMARIZATION_THRESHOLD,
            keep_recent=Config.MW_SUMMARIZATION_KEEP_RECENT
        ),
        ToolRetryMiddleware(
            max_retries=Config.MW_TOOL_MAX_RETRIES,
            backoff_factor=Config.MW_TOOL_BACKOFF_FACTOR
        ),
    ])
    
    try:
        # 导入 PostgresSaver 和 PostgresStore
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore
        import psycopg2
        
        conn_psycopg2 = psycopg2.connect(DB_URI)
        print("✓ 数据库连接成功!")
        conn_psycopg2.close()
        
        # 使用 psycopg-pool 创建连接池
        from psycopg_pool import ConnectionPool
        # 创建连接池
        pool = ConnectionPool(DB_URI, min_size=1, max_size=10)
        
        # 创建 PostgresSaver 和 PostgresStore
        checkpointer = PostgresSaver(pool)
        store = PostgresStore(pool, index={"dims": 1536, "embed": llm_embedding})
        checkpointer.setup()
        store.setup()
        logger.info("PostgresSaver and PostgresStore initialized successfully")

        # 构建工作流图并编译
        workflow = _build_workflow_graph(llm_chat, llm_embedding, tool_config, middleware_manager, store)
        return workflow.compile(checkpointer=checkpointer, store=store)
        
    except Exception as e:
        logger.error(f"Failed to setup PostgresSaver or PostgresStore: {e}")
        logger.warning("Falling back to in-memory storage due to PostgreSQL compatibility issues")
        
        # 降级到使用内存存储
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore
        
        checkpointer = MemorySaver()
        store = InMemoryStore()
        
        # 构建工作流图并编译
        workflow = _build_workflow_graph(llm_chat, llm_embedding, tool_config, middleware_manager, store)
        return workflow.compile(checkpointer=checkpointer, store=store)


def extract_graph_response(events) -> Tuple[str, Optional[dict]]:
    """
    从 graph.stream() 事件流中提取结构化响应 — 共享 API。

    供 main.py（HTTP 服务）和 graph_response（CLI）共同使用，
    消除两处重复的事件解析逻辑。

    Args:
        events: graph.stream() 返回的事件迭代器

    Returns:
        Tuple[str, Optional[dict]]:
            - content_text: 最终 AI 回复文本
            - final_payload: 医疗安全守卫生成的结构化载荷（仅 medical 路由时非 None）

    Raises:
        ResponseExtractionError: 事件流为空或格式异常时抛出
    """
    last_final_payload = None
    last_route_domain = None
    last_ai_message = None
    event_count = 0
    last_node_name = None

    for event in events:
        event_count += 1
        try:
            for node_name, value in event.items():
                last_node_name = node_name
                logger.info(f"[extract_graph_response] 事件 {event_count}: 节点={node_name}, keys={list(value.keys())}")
                
                if "messages" not in value or not isinstance(value.get("messages"), list):
                    continue

                if "route_domain" in value:
                    last_route_domain = value["route_domain"]
                    logger.info(f"[extract_graph_response] 更新 route_domain={last_route_domain}")

                if "final_payload" in value and value["final_payload"]:
                    last_final_payload = value["final_payload"]
                    logger.info(
                        f"检测到 final_payload，route={last_final_payload.get('route')}",
                        extra={"payload_route": last_final_payload.get("route")}
                    )

                last_message = value["messages"][-1]
                logger.info(f"[extract_graph_response] last_message type={type(last_message).__name__}, has_content={hasattr(last_message, 'content')}, has_tool_calls={hasattr(last_message, 'tool_calls')}")
                
                if hasattr(last_message, "tool_calls"):
                    tool_calls_value = last_message.tool_calls
                    tool_calls_type = type(tool_calls_value).__name__
                    tool_calls_count = len(tool_calls_value) if isinstance(tool_calls_value, (list, tuple)) else "N/A"
                    logger.info(f"[extract_graph_response] tool_calls type={tool_calls_type}, value={tool_calls_value}, count={tool_calls_count}")
                    
                    if tool_calls_value and isinstance(tool_calls_value, (list, tuple)) and len(tool_calls_value) > 0:
                        logger.info(f"[extract_graph_response] 跳过有 tool_calls 的消息 (tool_calls count: {len(tool_calls_value)})")
                        continue

                if hasattr(last_message, "content"):
                    content = last_message.content
                    if content:
                        if hasattr(last_message, "name") and last_message.name:
                            logger.info(f"Tool Output [{last_message.name}]: {content[:200]}")
                        else:
                            last_ai_message = last_message
                            logger.info(f"[extract_graph_response] 设置 last_ai_message, content length={len(content)}")
                    else:
                        logger.warning(f"[extract_graph_response] 消息 content 为空: type={type(last_message).__name__}")

        except Exception as e:
            logger.warning(f"事件流解析异常（跳过该事件）: {e}")
            continue

    if event_count == 0:
        raise ResponseExtractionError(
            "事件流为空，图谱未产生任何输出",
            details={"possible_causes": ["图谱未编译", "输入被 Middleware 拦截", "数据库连接失败"]}
        )

    content_text = last_ai_message.content if last_ai_message else ""

    if not content_text and not last_final_payload:
        logger.error(f"[extract_graph_response] 提取失败: event_count={event_count}, last_node={last_node_name}, route_domain={last_route_domain}, has_ai_message={last_ai_message is not None}")
        raise ResponseExtractionError(
            "未能从事件流中提取有效回复文本",
            details={
                "event_count": event_count,
                "last_node": last_node_name,
                "has_final_payload": last_final_payload is not None,
                "route_domain": last_route_domain,
                "has_ai_message": last_ai_message is not None,
            }
        )

    if last_final_payload and last_final_payload.get("answer"):
        content_text = last_final_payload["answer"]

    return content_text, last_final_payload


def graph_response(graph: StateGraph, user_input: str, config: dict, tool_config: ToolConfig, context: Context) -> None:
    """处理用户输入并输出响应（美化版）。

    输出逻辑：
    - 如果结果包含 final_payload → 以 Markdown 风格格式化输出
    - 否则 → 直接打印普通 AI 回复

    Args:
        graph: 状态图实例。
        user_input: 用户输入。
        config: 运行时配置。
        tool_config: 工具配置参数。
        context: 运行时上下文。
    """
    try:
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            context=context
        )

        content_text, last_final_payload = extract_graph_response(events)

        if last_final_payload:
            _print_medical_output(last_final_payload)
        elif content_text:
            print(f"\nAssistant: {content_text}\n")

    except ResponseExtractionError as ree:
        logger.error(f"响应提取失败: {ree.to_dict()}")
        print(f"Assistant: 响应提取失败 — [{ree.code}] {ree.message}")
    except ValueError as ve:
        logger.error(f"Value error in response processing: {ve}")
        print("Assistant: 处理响应时发生值错误")
    except RagAgentError as rae:
        logger.error(f"RagAgent 异常 [{rae.code}]: {rae.message}")
        print(f"Assistant: 处理请求时发生错误 — [{rae.code}] {rae.message}")
    except Exception as e:
        logger.error(f"Error processing response: {e}", exc_info=True)
        print("Assistant: 处理响应时发生未知错误")


def _print_medical_output(payload: dict) -> None:
    """以 Markdown 风格打印医疗分析结果。

    Args:
        payload: final_payload 字典，包含医疗分析完整数据。
    """
    print("\n" + "=" * 60)
    
    # 🏥 医疗分析
    print("\n🏥 【医疗分析】")
    print("-" * 40)
    answer = payload.get("answer", "")
    if answer:
        print(answer)
    
    # 🩺 分诊建议与紧急度
    triage = payload.get("structured_data", {}).get("triage", {})
    if triage:
        print("\n🩺 【分诊建议与紧急度】")
        print("-" * 40)
        departments = triage.get("recommended_departments", [])
        urgency = triage.get("urgency_level", "routine")
        reason = triage.get("triage_reason", "")
        confidence = triage.get("triage_confidence", 0.8)
        
        print(f"推荐科室: {', '.join(departments)}")
        print(f"紧急程度: {_get_urgency_display(urgency)}")
        print(f"分诊理由: {reason}")
        print(f"置信度: {confidence:.0%}")
    
    # ⚠️ 安全警告
    risk_warning = payload.get("risk_warning", "")
    risk_level = payload.get("risk_level", "low")
    if risk_warning and risk_warning != "无高危风险":
        print("\n⚠️ 【安全警告】")
        print("-" * 40)
        print(risk_warning)
    
    print("\n" + "=" * 60 + "\n")


def _get_urgency_display(urgency: str) -> str:
    """将紧急程度转换为可读显示文本。

    Args:
        urgency: 紧急程度代码 (routine/urgent/emergency)

    Returns:
        str: 可读的紧急程度描述。
    """
    urgency_map = {
        "routine": "🟢 常规就诊",
        "urgent": "🟡 尽快就医",
        "emergency": "🔴 紧急就诊"
    }
    return urgency_map.get(urgency, f"📋 {urgency}")


def main():
    """主函数，初始化并运行聊天机器人。"""
    try:
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
        
        # ===== 物理工具隔离：分别初始化 RAG 和 Medical 工具 =====
        rag_tools = get_rag_tools(llm_embedding)
        medical_tools = get_medical_agent_tools(llm_embedding)
        tool_config = ToolConfig(rag_tools=rag_tools, medical_tools=medical_tools)

        graph = create_graph(llm_chat, llm_embedding, tool_config)

        print("聊天机器人准备就绪！输入 'quit'、'exit' 或 'q' 结束对话。")
        config = {"configurable": {"thread_id": "1"}}
        context = Context(user_id="1")

        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("拜拜!")
                break
            if not user_input:
                print("请输入聊天内容！")
                continue
            graph_response(graph, user_input, config, tool_config, context)

    except RuntimeError as e:
        logger.error(f"Initialization error: {e}")
        print(f"错误: 初始化失败 - {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n被用户打断。再见！")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"错误: 发生未知错误 - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()