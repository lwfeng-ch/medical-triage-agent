# vectorSave.py
"""
向量存储引擎 v2 - 增强版
完整知识库构建链路: 任意格式文件 → MinerU高保真Markdown → 两阶段语义切分 → 带元数据的向量存储
"""
# vectorSave.py
"""
向量存储引擎 v2 - 增强版
完整知识库构建链路: 任意格式文件 → MinerU高保真Markdown → 两阶段语义切分 → 带元数据的向量存储
"""
import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Callable

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.embeddings import Embeddings

from utils.config import Config
from mineru_client import MinerUClient
from utils.markdown_splitter import MarkdownSplitter
from utils.logger import setup_logger

logger = setup_logger(__name__)

os.environ['NO_PROXY'] = 'localhost,127.0.0.1'


class CustomEmbeddings(Embeddings):
    """
    将自定义 embedding 函数包装为 LangChain Embeddings 标准接口。
    QdrantVectorStore 要求 embedding 参数必须实现此接口。
    """

    def __init__(self, embedding_fn: Callable[[List[str]], List[List[float]]]):
        self.embedding_fn = embedding_fn

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_fn(texts)

    def embed_query(self, text: str) -> List[float]:
        result = self.embedding_fn([text])
        return result[0] if result else []


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    根据 LLM_TYPE 配置调用对应的 Embedding 模型计算向量。
    """
    llm_type = Config.LLM_TYPE

    client_config = {
        "openai": {
            "base_url": Config.OPENAI_API_BASE,
            "api_key": Config.OPENAI_API_KEY,
            "model": Config.OPENAI_EMBEDDING_MODEL,
        },
        "qwen": {
            "base_url": Config.QWEN_API_BASE,
            "api_key": Config.QWEN_API_KEY,
            "model": Config.QWEN_EMBEDDING_MODEL,
        },
        "ollama": {
            "base_url": Config.OLLAMA_API_BASE,
            "api_key": Config.OLLAMA_API_KEY,
            "model": Config.OLLAMA_EMBEDDING_MODEL,
        },
        "oneapi": {
            "base_url": Config.ONEAPI_API_BASE,
            "api_key": Config.ONEAPI_API_KEY,
            "model": Config.ONEAPI_EMBEDDING_MODEL,
        },
    }

    config = client_config.get(llm_type)
    if not config:
        logger.error(f"不支持的 LLM_TYPE: {llm_type}")
        return []

    try:
        client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"]
        )
        data = client.embeddings.create(
            input=texts,
            model=config["model"]
        ).data
        return [item.embedding for item in data]
    except Exception as e:
        logger.error(f"生成向量时出错: {e}")
        return []


def generate_vectors(data: List[str], max_batch_size: int = None) -> List[List[float]]:
    """
    对文本按批次进行向量计算。
    """
    max_batch_size = max_batch_size or Config.EMBEDDING_BATCH_SIZE
    results = []

    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        response = get_embeddings(batch)
        results.extend(response)

    return results


class VectorStoreV2:
    """
    增强版向量数据库连接器（Qdrant）- 支持混合检索
    """

    def __init__(
        self,
        collection_name: str = None,
        embedding_fn: Callable = None,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        qdrant_local_path: str = None,
        use_hybrid: bool = True
    ):
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        self.embedding_fn = embedding_fn or generate_vectors
        self._collection_initialized = False
        self.use_hybrid = use_hybrid

        # ✅ 预先创建 LangChain 兼容的包装对象（复用，避免重复创建）
        self.lc_embeddings = CustomEmbeddings(self.embedding_fn)
        # vectorstore 引用（混合检索时使用）
        self.vectorstore: Optional[QdrantVectorStore] = None

        url = qdrant_url or Config.QDRANT_URL
        api_key = qdrant_api_key or Config.QDRANT_API_KEY
        local_path = qdrant_local_path or Config.QDRANT_LOCAL_PATH

        try:
            if url and url != ":memory:":
                self.client = QdrantClient(
                    url=url,
                    api_key=api_key,
                    prefer_grpc=False,
                )
                logger.info(f"连接到 Qdrant 服务器：{url}")
            else:
                self.client = QdrantClient(
                    path=local_path,
                    prefer_grpc=False,
                )
                logger.info(f"使用 Qdrant 本地持久化模式：{local_path}")
        except Exception as e:
            logger.error(f"连接 Qdrant 失败：{e}")
            raise

    def _ensure_collection(self, vector_size: int):
        """
        确保集合存在，如不存在则创建（支持混合检索）。
        集合已存在时，同样需要初始化 vectorstore 引用。
        """
        if self._collection_initialized:
            return

        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                # ── 集合不存在，新建 ──────────────────────────────────
                if self.use_hybrid:
                    os.environ["FASTEMBED_CACHE_PATH"] = os.path.abspath(
                        "model/model/sparsemodel"
                    )
                    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            "text-dense": qdrant_models.VectorParams(
                                size=vector_size,
                                distance=qdrant_models.Distance.COSINE,
                            )
                        },
                        sparse_vectors_config={
                            "text-sparse": qdrant_models.SparseVectorParams(
                                index=qdrant_models.SparseIndexParams(
                                    on_disk=False,
                                )
                            )
                        },
                    )
                    logger.info(
                        f"创建支持混合检索的 Qdrant 集合："
                        f"{self.collection_name}, 维度：{vector_size}"
                    )

                    self.vectorstore = QdrantVectorStore(
                        client=self.client,
                        embedding=self.lc_embeddings,
                        sparse_embedding=sparse_embeddings,
                        collection_name=self.collection_name,
                        retrieval_mode=RetrievalMode.HYBRID,
                        vector_name="text-dense",
                        sparse_vector_name="text-sparse",
                    )
                    logger.info("QdrantVectorStore 混合检索初始化成功")

                else:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=qdrant_models.VectorParams(
                            size=vector_size,
                            distance=qdrant_models.Distance.COSINE
                        )
                    )
                    logger.info(
                        f"创建 Qdrant 集合：{self.collection_name}, 维度：{vector_size}"
                    )

            else:
                # ── 集合已存在，恢复 vectorstore 引用 ──────────────────
                logger.info(f"Qdrant 集合已存在：{self.collection_name}")

                if self.use_hybrid:
                    os.environ["FASTEMBED_CACHE_PATH"] = os.path.abspath(
                        "model/model/sparsemodel"
                    )
                    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

                    self.vectorstore = QdrantVectorStore(
                        client=self.client,
                        embedding=self.lc_embeddings,
                        sparse_embedding=sparse_embeddings,
                        collection_name=self.collection_name,
                        retrieval_mode=RetrievalMode.HYBRID,
                        vector_name="text-dense",
                        sparse_vector_name="text-sparse",
                    )
                    logger.info("QdrantVectorStore 混合检索（已有集合）初始化成功")

            self._collection_initialized = True

        except Exception as e:
            logger.error(f"确保集合存在时出错：{e}", exc_info=True)
            raise

    def add_documents(self, documents: List[str]):
        """添加纯文本文档（兼容原接口）"""
        embeddings = self.embedding_fn(documents)
        if not embeddings:
            logger.error("向量计算结果为空，无法添加文档")
            return

        self._ensure_collection(len(embeddings[0]))

        points = []
        for doc, embedding in zip(documents, embeddings):
            point = qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"text-dense": embedding} if self.use_hybrid else embedding,
                payload={
                    "document": doc,
                    "page_content": doc,
                    "source": "unknown",
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"成功添加 {len(points)} 个文档到集合 '{self.collection_name}'")

    def upsert_with_metadata(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str] = None,
        use_context_prefix: bool = True
    ) -> List[str]:
        """
        带完整元数据的文档插入（核心方法）。
        混合检索模式下通过 QdrantVectorStore.add_documents() 自动计算稀疏向量。
        """
        if len(texts) != len(metadatas):
            raise ValueError(
                f"texts({len(texts)}) 与 metadatas({len(metadatas)}) 数量不匹配"
            )

        # 标题前置拼接，提升检索精度
        embed_texts = texts
        if use_context_prefix:
            splitter = MarkdownSplitter()
            embed_texts = [
                splitter.build_context_string({"content": t, "metadata": m})
                for t, m in zip(texts, metadatas)
            ]

        embeddings = self.embedding_fn(embed_texts)
        if not embeddings:
            logger.error("向量计算结果为空")
            return []

        self._ensure_collection(len(embeddings[0]))

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # ── 混合检索模式：通过 QdrantVectorStore 写入（自动计算稀疏向量）──
        if self.use_hybrid and self.vectorstore is not None:
            from langchain_core.documents import Document
            lc_docs = [
                Document(
                    page_content=text,
                    metadata={
                        "document": text,
                        "page_content": text,
                        **metadata
                    }
                )
                for text, metadata in zip(texts, metadatas)
            ]
            self.vectorstore.add_documents(lc_docs, ids=ids)
            logger.info(
                f"[混合检索] 成功写入 {len(lc_docs)} 条数据到 '{self.collection_name}'"
            )
            return ids

        # ── 纯向量模式：直接 upsert ─────────────────────────────────────
        points = []
        for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            payload = {"document": text, "page_content": text, **metadata}
            point = qdrant_models.PointStruct(
                id=id_,
                vector=embedding,
                payload=payload
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(
            f"成功写入 {len(points)} 条数据（含元数据）到 '{self.collection_name}'"
        )
        return ids

    def search(
        self,
        query: str,
        top_n: int = 5,
        query_filter: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        相似性搜索。
        混合检索集合必须指定 using="text-dense"，否则 Qdrant 报 400。
        """
        try:
            query_embedding = self.embedding_fn([query])
            if not query_embedding:
                logger.error("查询向量计算失败")
                return {"documents": [[]], "distances": [[]]}

            search_kwargs = {
                "collection_name": self.collection_name,
                "query": query_embedding[0],
                "limit": top_n,
                "with_payload": True,
            }

            # ✅ 混合检索集合必须指定向量名，否则 Qdrant 不知道用哪个向量
            if self.use_hybrid:
                search_kwargs["using"] = "text-dense"

            if query_filter:
                search_kwargs["query_filter"] = query_filter

            search_results = self.client.query_points(**search_kwargs)

            documents, distances = [], []
            for point in search_results.points:
                doc_text = point.payload.get(
                    "document",
                    point.payload.get("page_content", "")
                )
                documents.append(doc_text)
                distances.append(point.score)

            return {"documents": [documents], "distances": [distances]}

        except Exception as e:
            logger.error(f"检索向量数据库时出错: {e}")
            return {"documents": [[]], "distances": [[]]}

    def clear_collection(self, clear: bool = False):
        """删除当前集合（重建时清空脏数据）"""
        if not clear:
            return

        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name in collection_names:
                self.client.delete_collection(collection_name=self.collection_name)
                logger.info(f"已删除旧集合: {self.collection_name}，脏数据已清空")
                # ✅ 重置状态，下次 _ensure_collection 会重新创建
                self._collection_initialized = False
                self.vectorstore = None
            else:
                logger.info(f"集合 {self.collection_name} 不存在，无需清理")
        except Exception as e:
            logger.error(f"清理集合时出错: {e}")


class KnowledgeBaseBuilder:
    """
    知识库构建器 - 整合完整链路的顶层入口
    链路：文件 → MinerU 转换 → Markdown 切分 → 向量化 → 存储
    """

    def __init__(
        self,
        collection_name: str = None,
        mineru_api_url: str = None,
        clear_existing: bool = False,
        use_hybrid: bool = True
    ):
        self.mineru_client = MinerUClient(api_url=mineru_api_url)
        self.splitter = MarkdownSplitter()
        self.vector_store = VectorStoreV2(
            collection_name=collection_name,
            use_hybrid=use_hybrid
        )
        # ✅ 只在第一次调用前清空一次，后续文件追加写入
        self.clear_existing = clear_existing
        self._cleared = False  # 标记是否已执行过清空

    def build_from_file(
        self,
        file_path: str,
        parse_method: str = None
    ) -> Dict[str, Any]:
        """
        从单个文件构建知识库。
        注意：clear_existing 仅在第一次调用时生效，避免多文件场景互相覆盖。
        """
        logger.info(f"开始构建知识库，源文件: {file_path}")

        result = self.mineru_client.convert_file(file_path, parse_method=parse_method)
        if not result["success"] or not result["markdown"]:
            logger.error(f"MinerU 转换失败: {result.get('error')}")
            return {"success": False, "error": result.get("error"), "chunks_count": 0}

        markdown_text = result["markdown"]
        chunks = self.splitter.split_text(markdown_text)

        # ✅ 仅首次清空，后续文件追加，不再重复删库
        if self.clear_existing and not self._cleared:
            self.vector_store.clear_collection(clear=True)
            self._cleared = True

        texts = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "filename": result.get("filename", ""),
                "source": file_path,
                **chunk.get("metadata", {})
            }
            for chunk in chunks
        ]

        self.vector_store.upsert_with_metadata(texts, metadatas)

        return {
            "success": True,
            "filename": result.get("filename"),
            "chunks_count": len(chunks),
            "markdown_length": len(markdown_text),
        }

    def build_from_directory(
        self,
        input_dir: str = None,
        output_dir: str = None,
        parse_method: str = None
    ) -> Dict[str, Any]:
        """从目录批量构建知识库"""
        input_dir = input_dir or Config.INPUT_DIR
        md_results = self.mineru_client.convert_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            parse_method=parse_method
        )

        all_chunks = []
        for filename, markdown_text in md_results.items():
            if not markdown_text:
                continue
            chunks = self.splitter.split_text(markdown_text)
            for chunk in chunks:
                chunk["source_filename"] = filename
                all_chunks.append(chunk)

        # 目录模式：整体清空一次即可
        if self.clear_existing:
            self.vector_store.clear_collection(clear=True)

        texts = [chunk["content"] for chunk in all_chunks]
        metadatas = [
            {
                "filename": chunk.get("source_filename", ""),
                "source": chunk.get("source_filename", ""),
                **chunk.get("metadata", {})
            }
            for chunk in all_chunks
        ]

        self.vector_store.upsert_with_metadata(texts, metadatas)

        total_files = sum(1 for v in md_results.values() if v)
        return {
            "success": True,
            "files_processed": total_files,
            "total_files": len(md_results),
            "chunks_count": len(all_chunks),
        }

    def search(self, query: str, top_n: int = 5) -> Dict[str, Any]:
        """搜索知识库"""
        return self.vector_store.search(query, top_n=top_n)


def vectorStoreSave():
    """
    从 input 文件夹批量导入所有支持的文件到知识库。
    """
    input_dir = "input"

    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在：{input_dir}")
        return

    supported_extensions = ['.pdf', '.docx', '.pptx', '.html', '.htm', '.txt', '.md']
    files_to_process = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1].lower() in supported_extensions
    ]

    if not files_to_process:
        logger.warning(f"在 {input_dir} 目录中未找到支持的文件")
        logger.info(f"支持的文件格式：{', '.join(supported_extensions)}")
        return

    logger.info(
        f"发现 {len(files_to_process)} 个待处理文件："
        f"{[os.path.basename(f) for f in files_to_process]}"
    )

    builder = KnowledgeBaseBuilder(
        collection_name=Config.QDRANT_COLLECTION_NAME,
        clear_existing=True   # ✅ 只清空一次，后续文件均追加
    )

    total_chunks = 0
    successful_files = 0

    for file_path in files_to_process:
        try:
            result = builder.build_from_file(file_path)
            if result.get("success"):
                successful_files += 1
                total_chunks += result.get("chunks_count", 0)
                logger.info(
                    f"✓ {os.path.basename(file_path)}: {result.get('chunks_count')} 个片段"
                )
            else:
                logger.error(f"✗ {os.path.basename(file_path)}: {result.get('error')}")
        except Exception as e:
            logger.error(f"✗ {os.path.basename(file_path)}: 处理失败 - {e}")

    logger.info(f"\n知识库构建完成:")
    logger.info(f"  - 成功处理：{successful_files}/{len(files_to_process)} 个文件")
    logger.info(f"  - 总片段数：{total_chunks} 个")

    if total_chunks > 0:
        user_query = "张三九的健康状况是什么，医院门诊的上班时间"
        search_results = builder.search(user_query, top_n=3)
        logger.info(f"\n测试查询：'{user_query}'")
        logger.info(f"检索结果：{search_results}")


if __name__ == "__main__":
    vectorStoreSave()