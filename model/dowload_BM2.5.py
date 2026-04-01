# ─── 步骤1：有网机器上预下载 ───
from fastembed import SparseTextEmbedding

sparse_model_path = "model/sparsemodel"
sparse = SparseTextEmbedding(
    model_name="Qdrant/bm25",
    cache_dir=sparse_model_path       # 下载到指定目录
)
# 运行后，./sparsemodel/ 下会有 Qdrant/bm25 的文件

# ─── 步骤2：把 ./sparsemodel/ 整个拷贝到离线服务器 ───

# ─── 步骤3：离线服务器上使用 ───
# 方法 A：通过 fastembed 原生加载（离线可用）
bm25_model = SparseTextEmbedding(
    model_name="Qdrant/bm25",
    specific_model_path=sparse_model_path  # 直接指向本地目录
)
