# 智能医疗分诊系统 - 项目规则

## 项目概述

基于 LangGraph 的双路由智能医疗分诊系统，融合 RAG 技术与医疗专业知识图谱。

**核心特性:**
- 双路由架构: General RAG Agent + Medical Agent
- 混合检索: BM25 + 向量检索
- 知识库构建: MinerU GPU 解析 → 语义切分 → 向量化存储
- 医疗安全守卫: 风险评估 + 分诊建议 + 免责声明

## 技术栈

**核心框架:**
- LangChain 1.0+ / LangGraph 1.0+
- FastAPI 0.100+ / Gradio 4.0+
- Qdrant 1.12+ (向量数据库)
- PostgreSQL (持久化存储)

**关键依赖:**
- langchain-qdrant (混合检索)
- langgraph-checkpoint-postgres (状态持久化)
- openai (Embedding API)
- pydantic (数据验证)

## 代码风格规范

### 基本原则

1. **简洁优先**: 不过度设计，适当使用设计模式
2. **单一职责**: 函数不超过 50 行，圈复杂度 ≤ 10
3. **DRY 原则**: 提取公共逻辑，不写重复代码
4. **类型注解**: 所有公共函数必须有类型注解
5. **不硬编码**: 配置走环境变量或 Config 类

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 函数/方法 | snake_case | `get_user_by_id()` |
| 类 | PascalCase | `KnowledgeBaseBuilder` |
| 常量 | UPPER_SNAKE_CASE | `MAX_RETRY_COUNT` |
| 私有方法 | _leading_underscore | `_validate_input()` |
| 变量 | snake_case | `user_input` |
| 布尔变量 | is_/has_/can_ 前缀 | `is_valid`, `has_permission` |

### 注释模板（强制格式）

```python
def function_name(param1: Type, param2: Type) -> ReturnType:
    """
    一句话描述功能。

    Args:
        param1 (Type): 参数描述。
        param2 (Type): 参数描述。

    Returns:
        ReturnType: 返回值描述。

    Raises:
        ExceptionType: 异常条件描述。
    """
    pass


class ClassName:
    """
    类功能描述。

    Attributes:
        attr1 (Type): 属性描述。
        attr2 (Type): 属性描述。
    """
    
    def __init__(self, param: Type):
        """
        初始化方法。

        Args:
            param (Type): 参数描述。
        """
        pass
```

### 模块顶部注释

```python
"""
模块功能描述。

依赖说明:
- langchain: 用于 LLM 调用
- langgraph: 用于状态机编排

使用示例:
    from module import ClassName
    instance = ClassName()
    result = instance.method()
"""
```

## 模块划分原则

### 分层架构

```
表现层 (FastAPI Routes / Gradio UI)
    ↓
应用层 (LangGraph Runtime / Middleware Manager)
    ↓
领域层 (Agent Nodes / Tool Nodes / Medical Analysis)
    ↓
基础设施层 (Vector Store / LLM Client / Database)
```

**依赖原则:**
- 上层可依赖下层，下层不可依赖上层
- 同层模块间通过接口通信
- 基础设施层通过依赖注入提供实现

### 目录结构

```
L1-Project-2/
├── main.py                 # FastAPI 服务入口
├── ragAgent.py             # Agent 核心逻辑
├── vectorSave.py           # 向量存储引擎
├── gradio_ui.py            # Gradio 前端
├── config.py               # 统一配置
├── prompts/                # 提示模板目录
├── utils/                  # 工具模块
│   ├── config/             # 配置子模块
│   ├── medical_analysis/   # 医疗分析子模块
│   ├── llms.py             # LLM 客户端
│   ├── tools_config.py     # 工具配置
│   ├── retriever.py        # 检索器
│   ├── middleware.py       # Middleware
│   ├── logger.py           # 日志工具
│   └── auth.py             # 认证工具
├── test/                   # 测试目录
├── input/                  # 输入文件目录
└── output/                 # 输出文件目录
```

## 开发流程规范

### Bug 修复流程

1. **Memory 检索历史** → **Analyzer 阅读相关代码**
2. **Sequential Thinking 分析** 结合 **debugger** 找到根源错误（复述问题 + 至少 2 种可能原因）
3. **涉及第三方库时 Context7 查文档确认**
4. **制定方案 + Mermaid 流程图** → **Feedback 确认**
5. **TDD 先写测试复现** → **执行修复** → **自检副作用**
6. **Mermaid 画修复前后对比** → **Feedback 验收** → **Memory 保存**

### 新功能流程

1. **Memory 检索** → **Analyzer 分析结构** → **Context7 查文档**
2. **Sequential Thinking 设计** + **Mermaid 架构图**
3. **Feedback 确认方案**
4. **TDD 先写测试** → **编码**（遵循注释模板标明 Args/Returns/Raises）
5. **Feedback 验收** → **Memory 保存决策**

### 重构流程

1. **Analyzer 全面分析** → **Mermaid 画前后对比图**
2. **Feedback 确认方案** → **分模块执行**
3. **每模块完成后 Feedback 确认再继续**
4. **全量测试** → **Feedback 最终验收** → **Memory 保存**

## 接口设计标准

### RESTful API 规范

**URL 设计:**
- 使用名词复数形式: `/v1/documents`
- 使用连字符分隔: `/v1/medical-reports`
- 避免深层嵌套: `/v1/users/{id}/documents` (最多 2 层)

**HTTP 方法映射:**

| 方法 | 用途 | 示例 |
|------|------|------|
| GET | 查询资源 | `GET /v1/documents` |
| POST | 创建资源 | `POST /v1/documents/upload` |
| PUT | 全量更新 | `PUT /v1/documents/{id}` |
| PATCH | 部分更新 | `PATCH /v1/documents/{id}` |
| DELETE | 删除资源 | `DELETE /v1/documents/{id}` |

**响应格式:**

```json
// 成功响应
{
    "success": true,
    "data": {...},
    "message": "操作成功"
}

// 错误响应
{
    "success": false,
    "error": {
        "code": "DOCUMENT_NOT_FOUND",
        "message": "文档不存在",
        "details": {"document_id": "abc123"}
    }
}
```

### LangGraph 节点接口规范

**节点函数签名:**

```python
def node_function(state: AgentState, config: RunnableConfig) -> dict:
    """
    节点函数。

    Args:
        state (AgentState): 当前状态。
        config (RunnableConfig): 运行配置。

    Returns:
        dict: 状态更新字典。
    """
    pass
```

**状态更新规则:**
- 返回字典必须是 `AgentState` 的子集
- 使用 `Annotated[T, operator.add]` 实现累加字段
- 避免直接修改 `state` 对象，返回新字典

## 错误处理机制

### 异常层次结构

```
Exception
  └── RagAgentError (基类)
      ├── GraphBuildError (图谱构建失败)
      ├── ResponseExtractionError (响应提取失败)
      ├── MedicalAnalysisError (医疗分析异常)
      └── ToolExecutionError (工具执行异常)
```

### 错误处理流程

```python
try:
    # 业务逻辑
    result = process_data()
    return {"success": True, "data": result}

except ValidationError as e:
    # 输入验证错误
    logger.warning(f"输入验证失败: {e}")
    raise HTTPException(status_code=400, detail=str(e))

except RagAgentError as e:
    # Agent 业务错误
    logger.error(f"Agent 错误 [{e.code}]: {e.message}")
    raise HTTPException(status_code=500, detail=e.to_dict())

except Exception as e:
    # 未知错误
    logger.exception(f"未知错误: {e}")
    raise HTTPException(status_code=500, detail="内部服务器错误")
```

### 日志规范

**日志级别:**
- **DEBUG**: 详细调试信息
- **INFO**: 关键业务节点
- **WARNING**: 可恢复的异常情况
- **ERROR**: 业务错误
- **CRITICAL**: 系统级错误

**日志格式:**

```python
# 正确示例
logger.info(f"用户 {user_id} 上传文档: {filename}")
logger.warning(f"检索结果为空，查询: {query[:50]}")
logger.error(f"工具 {tool_name} 执行失败: {error}", exc_info=True)

# 错误示例
logger.info("用户上传文档")  # 缺少关键信息
logger.warning("检索失败")   # 缺少上下文
logger.error("错误")         # 过于简单
```

## 版本更新策略

### 版本号规范

采用语义化版本号: `MAJOR.MINOR.PATCH`

- **MAJOR**: 不兼容的 API 变更
- **MINOR**: 向后兼容的功能新增
- **PATCH**: 向后兼容的问题修复

### 变更日志规范

**CHANGELOG.md 格式:**

```markdown
## [1.2.0] - 2026-04-15

### Added
- 新增用户医疗文档私有检索功能

### Changed
- 优化 ParallelToolNode 并行执行性能

### Fixed
- 修复 QdrantVectorStore 初始化错误(#123)

### Deprecated
- `VectorStoreV1` 类已废弃，请使用 `VectorStoreV2`

### Security
- 增强 user_id 认证机制，防止伪造
```

## Feedback 分级策略

1. **强制调用**: 删除文件/大量代码、架构修改确认、Bug方案确认、多方案选择、任务最终验收
2. **建议调用**: 多文件修改(≥3)、需求理解不确定
3. **免调用**: 简单问答、单文件小改(<20行)、格式化/注释补充

## Skills 使用规范

- **用户需要分析项目时**: 使用 project-analyzer skills
- **改动/理解代码前**: project-analyzer 先分析
- **新增功能/修 Bug**: tdd-workflow 驱动测试与 debugger 修复
- **不确定用哪个能力**: find-skills 查找
- **需要新的可复用流程**: skill-writer 生成
- **需要参考最佳实践**: prompt-lookup 查找

## MCP 使用规范

- **对话开始**: Memory 检索记忆，结束时保存新信息
- **第三方库**: Context7 查最新文档，不用可能过时的 API
- **复杂分析/设计**: Sequential Thinking 分步推理
- **关键决策/任务完成**: Feedback 确认（简单问答免调）
- **用户给链接**: Multi Fetch 抓取内容

## 测试规范

- 生成的测试文件存放在 `test/` 文件夹中
- 运行环境通常为 conda 环境
- 测试覆盖率目标: 80%+
- 测试类型: 单元测试 + 集成测试 + E2E 测试

## 安全规范

1. **不硬编码密钥**: 所有敏感配置走环境变量或 Config 类
2. **user_id 隔离**: 从认证体系获取，防止伪造
3. **输入验证**: 使用 Pydantic 模型验证所有输入
4. **错误信息**: 不暴露敏感信息（数据库连接、内部路径等）
5. **日志脱敏**: 不记录 API Key、密码等敏感信息

## 性能优化规范

1. **批量处理**: 向量化、API 调用使用批处理
2. **缓存机制**: 高频查询结果缓存
3. **异步执行**: I/O 密集型操作使用异步
4. **资源限制**: 设置超时、重试次数、并发限制
5. **监控告警**: 关键指标监控（响应时间、错误率、资源使用）

## 文档规范

1. **代码即文档**: 通过类型注解和注释自解释
2. **README.md**: 项目概述、快速开始、配置说明
3. **API 文档**: 使用 FastAPI 自动生成 Swagger 文档
4. **架构文档**: 使用 Mermaid 图表可视化
5. **变更日志**: CHANGELOG.md 记录所有变更

---

**最后更新**: 2026-04-15  
**维护者**: 开发团队
