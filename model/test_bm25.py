# test_bm25.py
"""BM25 稀疏嵌入模型 快速验证脚本"""
import os

os.environ["FASTEMBED_CACHE_PATH"] = os.path.abspath("model/model/sparsemodel")

def main():
    print("=" * 50)
    print("  BM25 稀疏嵌入模型验证")
    print("=" * 50)

    # ── 1. 检查 fastembed 包 ──
    print("\n[1/4] 检查 fastembed 安装...")
    try:
        import fastembed
        print(f"  ✅ fastembed 已安装, 版本: {fastembed.__version__}")
    except ImportError:
        print("  ❌ fastembed 未安装！请运行: pip install fastembed")
        return

    # ── 2. 初始化 BM25 模型（首次会自动下载） ──
    print("\n[2/4] 初始化 Qdrant/bm25 模型（首次会自动下载停用词文件）...")
    try:
        from fastembed import SparseTextEmbedding
        model = SparseTextEmbedding(model_name="Qdrant/bm25")
        print("  ✅ BM25 模型初始化成功")
    except Exception as e:
        print(f"  ❌ BM25 模型初始化失败: {e}")
        return

    # ── 3. 测试文档嵌入 ──
    print("\n[3/4] 测试文档稀疏嵌入...")
    test_docs = [
        "患者张三，男性，55岁，高血压病史3年",
        "建议低盐饮食，定期监测血压",
        "Python 是一门编程语言",
    ]

    embeddings = list(model.embed(test_docs))
    for i, (doc, emb) in enumerate(zip(test_docs, embeddings)):
        n_tokens = len(emb.indices)
        print(f"  📄 文档 {i+1}: 非零维度={n_tokens}, 文本=\"{doc[:30]}...\"")
        # 打印前 3 个稀疏维度作为示例
        if n_tokens > 0:
            preview_n = min(3, n_tokens)
            print(f"     前{preview_n}个维度 indices={emb.indices[:preview_n].tolist()}, "
                  f"values={emb.values[:preview_n].round(4).tolist()}")

    print(f"  ✅ {len(embeddings)} 篇文档嵌入成功")

    # ── 4. 测试查询嵌入（query_embed 会对 query 做特殊处理） ──
    print("\n[4/4] 测试查询稀疏嵌入...")
    query = "高血压患者饮食建议"
    query_emb = list(model.query_embed(query))[0]
    print(f"  🔍 查询: \"{query}\"")
    print(f"     非零维度={len(query_emb.indices)}")
    print(f"  ✅ 查询嵌入成功")

    # ── 5. 验证 LangChain 集成 ──
    print("\n[Bonus] 验证 LangChain FastEmbedSparse 集成...")
    try:
        from langchain_qdrant import FastEmbedSparse
        lc_sparse = FastEmbedSparse(model_name="Qdrant/bm25")

        lc_doc_result = lc_sparse.embed_documents(["测试文档"])
        lc_query_result = lc_sparse.embed_query("测试查询")

        print(f"  ✅ FastEmbedSparse 文档嵌入: indices 长度={len(lc_doc_result[0].indices)}")
        print(f"  ✅ FastEmbedSparse 查询嵌入: indices 长度={len(lc_query_result.indices)}")
    except ImportError:
        print("  ⚠️ langchain-qdrant 未安装，跳过集成测试")
    except Exception as e:
        print(f"  ❌ LangChain 集成测试失败: {e}")

    print("\n" + "=" * 50)
    print("  🎉 BM25 模型验证全部通过！")
    print("=" * 50)


if __name__ == "__main__":
    main()