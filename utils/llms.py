# utils/llms.py
import os
import logging
from http import HTTPStatus
from typing import Sequence, Optional

import dashscope
from pydantic import Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from langchain_core.documents.compressor import BaseDocumentCompressor


# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 模型配置字典
MODEL_CONFIGS = {
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "chat_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small"
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "chat_model": "qwen-plus",
        "rerank_model": "qwen3-rerank",
        "embedding_model": "text-embedding-v1"
    },
    "oneapi": {
        "base_url": "http://139.224.72.218:3000/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "chat_model": "qwen-max",
        "embedding_model": "text-embedding-v1"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "chat_model": "qwen2.5:32b",
        "embedding_model": "bge-m3:latest"
    }
}


# 默认配置
DEFAULT_LLM_TYPE = "qwen"
DEFAULT_TEMPERATURE = 0.5


# ──────────────────────────────────────────────────────────────
# DashScope Rerank 自定义组件
# ──────────────────────────────────────────────────────────────
class DashScopeReranker(BaseDocumentCompressor):
    """
    基于阿里百炼 DashScope 封装的文档重排压缩器。
    该类实现了 LangChain 的 BaseDocumentCompressor 接口。
    """
    model: str = Field(default="qwen3-rerank")
    api_key: str = Field(default="")
    top_n: int = Field(default=3)
    instruct: str = Field(
        default="Given a web search query, retrieve relevant passages that answer the query."
    )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """核心重排逻辑：调用 DashScope API 重新对 Document 排序并截断"""

        if not self.api_key:
            logger.error("Reranker 缺少 API_KEY，跳过重排。")
            return documents[:self.top_n]

        # 动态设置百炼全局密钥
        dashscope.api_key = self.api_key

        if not documents:
            return []

        docs_texts = [doc.page_content for doc in documents]

        try:
            logger.info(f"调用 Rerank 模型 [{self.model}], 待重排文档数: {len(docs_texts)}")

            resp = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=docs_texts,
                top_n=self.top_n,
                return_documents=True,
                instruct=self.instruct
            )

            if resp.status_code == HTTPStatus.OK:
                reranked_docs = []
                for item in resp.output.results:
                    # 【修复点】直接获取 item.index，并强制转换类型确保安全
                    original_index = int(item.index)
                    relevance_score = float(item.relevance_score)
                    
                    doc = documents[original_index]
                    doc.metadata["rerank_score"] = relevance_score 
                    reranked_docs.append(doc)
                return reranked_docs
            else:
                logger.error(f"Rerank 失败: code={resp.code}, msg={resp.message}")
                return documents[:self.top_n]

        except Exception as e:
            logger.error(f"调用 Rerank 发生异常: {e}")
            return documents[:self.top_n]


class LLMInitializationError(Exception):
    """自定义异常类用于LLM初始化错误"""
    pass


def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    """
    初始化LLM实例

    Args:
        llm_type (str): LLM类型，可选值为 'openai', 'oneapi', 'qwen', 'ollama'

    Returns:
        tuple[ChatOpenAI, OpenAIEmbeddings]: LLM实例和Embedding实例

    Raises:
        LLMInitializationError: 当LLM初始化失败时抛出
    """
    try:
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}. 可用的类型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        if llm_type == "ollama":
            os.environ["OPENAI_API_KEY"] = "NA"

        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=30,
            max_retries=2
        )

        llm_embedding = OpenAIEmbeddings(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["embedding_model"],
            deployment=config["embedding_model"],
            check_embedding_ctx_length=False
        )

        logger.info(f"成功初始化 {llm_type} LLM")
        return llm_chat, llm_embedding

    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")


def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    """
    获取LLM实例的封装函数，提供默认值和错误处理
    """
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warning(f"使用默认配置重试: {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        raise


def get_reranker(llm_type: str = DEFAULT_LLM_TYPE, top_n: int = 3) -> BaseDocumentCompressor:
    """
    获取重排器实例的封装函数。
    由 tools_config.py 在组装检索器时直接调用，做到完全解耦。
    """
    try:
        config = MODEL_CONFIGS.get(llm_type, MODEL_CONFIGS[DEFAULT_LLM_TYPE])
        model_name = config.get("rerank_model", "qwen3-rerank")
        api_key = config.get("rerank_api_key", os.getenv("DASHSCOPE_API_KEY"))

        if not api_key:
            logger.warning(f"缺少重排 API Key ({llm_type})，请检查环境变量 DASHSCOPE_API_KEY")

        reranker = DashScopeReranker(
            model=model_name,
            api_key=api_key,
            top_n=top_n
        )
        logger.info(f"成功初始化 {llm_type} 的 Reranker: {model_name} (top_n={top_n})")
        return reranker

    except Exception as e:
        logger.error(f"初始化 Reranker 失败: {str(e)}")
        raise LLMInitializationError(f"初始化 Reranker 失败: {str(e)}")


# 示例使用
if __name__ == "__main__":
    # 提醒：如果是本地直接运行此文件测试，请确保环境变量已配置
    if not os.getenv("DASHSCOPE_API_KEY"):
        logger.warning("未检测到 DASHSCOPE_API_KEY 环境变量，接下来的 API 调用测试可能会失败！")

    test_llm_type = "qwen"
    logger.info(f"========== 开始测试 [{test_llm_type}] 模型链路 ==========")

    try:
        # ---------------------------------------------------------
        # 测试 1: 初始化与验证 Chat / Embedding 模型
        # ---------------------------------------------------------
        logger.info(">>> 正在加载 Chat 与 Embedding 模型...")
        llm_chat, llm_embedding = get_llm(test_llm_type)

        # 1.1 测试 Chat 模型对话能力
        logger.info(">>> 测试 Chat 模型推理能力...")
        chat_res = llm_chat.invoke("你好，请只回复'Chat模型连接成功'这8个字。")
        logger.info(f"[Chat 响应]: {chat_res.content}")

        # 1.2 测试 Embedding 模型向量化能力
        logger.info(">>> 测试 Embedding 模型向量化能力...")
        embed_res = llm_embedding.embed_query("这是一个用于测试文本向量化的句子。")
        logger.info(f"[Embedding 响应]: 成功生成向量，维度大小为 {len(embed_res)}")

        # ---------------------------------------------------------
        # 测试 2: 初始化与验证 Reranker (重排) 模型
        # ---------------------------------------------------------
        logger.info(">>> 正在加载 Reranker 模型...")
        # 设定截取前 2 名
        reranker = get_reranker(test_llm_type, top_n=2)

        # 构造混合了医学和非医学领域的测试文档
        test_docs = [
            Document(page_content="苹果公司的最新款手机即将发布。"),
            Document(page_content="高血压患者应注意低盐饮食，多吃新鲜蔬菜和水果，并定期监测血压。"),
            Document(page_content="量子力学是物理学的一个重要分支，研究微观粒子的运动规律。"),
            Document(page_content="长期熬夜和不规律的作息会导致心血管疾病发病率上升。")
        ]
        test_query = "有什么预防高血压的饮食建议？"

        logger.info(f">>> 测试 Rerank 排序能力 (Query: '{test_query}')...")
        reranked_docs = reranker.compress_documents(documents=test_docs, query=test_query)

        logger.info(f"[Rerank 响应]: 重排完成，截取 Top {len(reranked_docs)} 文档：")
        for i, doc in enumerate(reranked_docs):
            score = doc.metadata.get('rerank_score', '无分数')
            logger.info(f"   Top {i+1} [相关性得分: {score}]: {doc.page_content}")

        logger.info("========== 所有模型链路测试通过！==========")

    except LLMInitializationError as e:
        logger.error(f"模型配置或初始化失败，程序终止: {str(e)}")
    except Exception as e:
        logger.error(f"API 请求过程中发生异常（请检查网络或余额）: {str(e)}")
        
 