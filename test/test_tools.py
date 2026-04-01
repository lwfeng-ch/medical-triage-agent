import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.llms import get_llm
import logging
import shutil
import tempfile
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

# 导入你项目中的核心模块
from utils.llms import get_llm
from utils.tools_config import get_tools
from utils.config import Config

# 设置基础日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 【关键点】：强制开启 tools_config 的 DEBUG 日志，以便观察粗排(k=5)和精排(top_n=3)的打印信息
logging.getLogger("utils.tools_config").setLevel(logging.DEBUG)
logging.getLogger("utils.llms").setLevel(logging.INFO)

def run_test():
    # 1. 创建一个临时的本地目录供 Qdrant 使用，保证测试环境隔离
    temp_qdrant_dir = tempfile.mkdtemp()
    logger.info(f"创建临时向量库目录: {temp_qdrant_dir}")

    # 临时覆盖 Config 里的数据库配置，指向临时目录
    Config.QDRANT_URL = ""
    Config.QDRANT_LOCAL_PATH = temp_qdrant_dir
    Config.QDRANT_COLLECTION_NAME = "test_hybrid_collection"

    try:
        # 2. 初始化底层模型
        logger.info(">>> 正在初始化 LLM 与 Embedding 模型...")
        _, llm_embedding = get_llm("qwen")

        # 3. 初始化稀疏向量模型 (BM25)
        logger.info(">>> 正在初始化 FastEmbed BM25 稀疏模型...")
        os.environ["FASTEMBED_CACHE_PATH"] = os.path.abspath("model/model/sparsemodel")
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # 4. 准备测试用的医疗假数据 (故意混入干扰项，验证 Rerank 威力)
        mock_docs = [
            Document(page_content="苹果公司今天发布了最新的 iPhone 16，具备强大的 AI 功能。"), # 彻底的干扰项
            Document(page_content="量子力学是研究微观粒子运动规律的物理学分支。"), # 彻底的干扰项
            Document(page_content="患者张三，男，45岁，既往有高血压病史5年，目前服用降压药控制良好。"), # 相关病史
            Document(page_content="李四的体检报告显示：空腹血糖偏高，建议低糖饮食，增加运动。"), # 弱相关
            Document(page_content="高血压患者日常饮食应以低盐低脂为主，多吃新鲜蔬菜，避免熬夜。"), # 强相关目标
            Document(page_content="王五，女，体检心电图提示窦性心律不齐，建议心内科随诊。") # 其他医疗数据
        ]

        # 5. 模拟真实入库过程 (同时生成 Dense 稠密向量 和 Sparse 稀疏向量)
        logger.info(">>> 正在将测试数据写入临时混合检索库 (双向量生成中)...")
        QdrantVectorStore.from_documents(
            documents=mock_docs,
            embedding=llm_embedding,
            sparse_embedding=sparse_embeddings,
            path=Config.QDRANT_LOCAL_PATH,
            collection_name=Config.QDRANT_COLLECTION_NAME,
            retrieval_mode=RetrievalMode.HYBRID
        )
        logger.info("测试数据写入完成！\n")

        # ==========================================
        # 核心测试：调用 get_tools 验证工具模块
        # ==========================================
        logger.info("========== 开始测试 Tool 模块 ==========")
        tools = get_tools(llm_embedding, llm_type="qwen")

        # 提取两个工具
        retriever_tool = next((t for t in tools if t.name == "health_record_retriever"), None)
        multiply_tool = next((t for t in tools if t.name == "multiply"), None)

        assert retriever_tool is not None, "未找到 health_record_retriever 工具"
        assert multiply_tool is not None, "未找到 multiply 工具"

        # 测点 1: 普通计算工具
        logger.info(">>> 测试 1: multiply 工具调用")
        mul_result = multiply_tool.invoke({"a": 3.0, "b": 4.0})
        logger.info(f"[Multiply 结果]: 3.0 * 4.0 = {mul_result}\n")

        # 测点 2: 核心两阶段检索工具
        test_query = "预防高血压的饮食建议有哪些？"
        logger.info(f">>> 测试 2: health_record_retriever 工具调用")
        logger.info(f"[搜索 Query]: {test_query}")
        
        # 触发工具执行 (注意看控制台的 DEBUG 打印，会展示两阶段过程)
        retriever_result = retriever_tool.invoke({"query": test_query})

        logger.info("\n========== 检索工具最终返回给 Agent 的内容 ==========")
        print(retriever_result)
        logger.info("=======================================================")

    finally:
        # 清理临时生成的数据库文件
        logger.info(f"测试结束，清理临时向量库目录: {temp_qdrant_dir}")
        shutil.rmtree(temp_qdrant_dir, ignore_errors=True)

if __name__ == "__main__":
    run_test()