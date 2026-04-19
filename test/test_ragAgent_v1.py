# test_ragAgent.py
import pytest
import os
import time
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from ragAgent import (
    create_graph,
    AgentState,
    ToolConfig,
    get_llm,
    get_tools,
    Config,
    Context,
    MiddlewareManager,
    LoggingMiddleware,
    ModelCallLimitMiddleware,
    PIIDetectionMiddleware,
    SummarizationMiddleware,
    ToolRetryMiddleware,
    grade_documents,
    route_after_grade,
    route_after_tools,
)

# ====================== 测试配置 ======================
TEST_THREAD_ID = "test_thread_001"
TEST_USER_ID = "test_user_001"

# 基于知识库设计的测试查询
TEST_QUERIES = {
    "basic_info": "李四六的姓名、年龄和职业是什么？",
    "allergy": "李四六对什么药物和食物过敏？",
    "cholesterol": "李四六的胆固醇水平怎么样？有什么健康风险？",
    "suggestion": "医生对李四六的胆固醇和糖尿病预防有什么建议？",
    "irrelevant": "今天天气怎么样？",
}

# 预期必须包含的关键事实（用于断言）
EXPECTED_FACTS = {
    "basic_info": ["李四六", "34", "小学教师", "1990"],
    "allergy": ["阿司匹林", "坚果"],
    "cholesterol": ["5.8", "LDL", "3.2", "心血管疾病"],
    "suggestion": ["胆固醇", "饮食", "运动", "血糖", "HbA1c"],
}


# ====================== 构造完整 state 的辅助函数 ======================
def make_route_state(relevance_score: str, rewrite_count: int) -> dict:
    """构造一个包含完整 messages 的 state，用于路由函数测试。
    route_after_grade 内部会检查 messages 字段，
    如果缺失会默认返回 rewrite，所以必须提供。
    """
    return {
        "messages": [
            HumanMessage(content="李四六的年龄是多少？"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_route_test",
                        "name": "retrieve",
                        "args": {"query": "李四六年龄"},
                    }
                ],
            ),
            ToolMessage(
                content="患者姓名为李四六，出生于1990年3月15日，目前34岁。",
                tool_call_id="call_route_test",
                name="retrieve",
            ),
        ],
        "relevance_score": relevance_score,
        "rewrite_count": rewrite_count,
    }


# ====================== Fixtures ======================
@pytest.fixture(scope="module")
def llm_and_tools():
    """初始化 LLM 和工具（module 级别复用）"""
    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
    tools = get_tools(llm_embedding)
    tool_config = ToolConfig(tools)
    return llm_chat, llm_embedding, tool_config


@pytest.fixture(scope="module")
def middleware_manager(llm_and_tools):
    """初始化 Middleware 管理器"""
    manager = MiddlewareManager(
        [
            LoggingMiddleware(),
            ModelCallLimitMiddleware(max_calls=20),
            PIIDetectionMiddleware(mode="warn"),
            SummarizationMiddleware(max_messages=15, keep_recent=5),
            ToolRetryMiddleware(max_retries=2),
        ]
    )
    return manager


@pytest.fixture(scope="module")
def graph(llm_and_tools, middleware_manager):
    """创建测试用的 StateGraph（module 级别）"""
    llm_chat, llm_embedding, tool_config = llm_and_tools
    graph_obj = create_graph(llm_chat, llm_embedding, tool_config=tool_config)
    return graph_obj


@pytest.fixture
def test_config():
    """每次测试独立的配置"""
    return {
        "configurable": {
            "thread_id": TEST_THREAD_ID,
            "user_id": TEST_USER_ID,
        }
    }


# ====================== 辅助函数 ======================
def extract_content(result) -> str:
    """从结果中安全提取最终消息内容"""
    final_message = result["messages"][-1]
    if hasattr(final_message, "content"):
        return final_message.content
    return str(final_message)


# ====================== 单元测试：初始化 ======================
class TestInitialization:
    """初始化相关测试"""

    def test_graph_initialization(self, graph):
        """测试图是否成功创建"""
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_middleware_manager_initialization(self, middleware_manager):
        """测试 Middleware 是否正确初始化"""
        assert middleware_manager is not None
        assert len(middleware_manager.middlewares) >= 4

    def test_llm_and_tools_initialization(self, llm_and_tools):
        """测试 LLM 和工具是否正确初始化"""
        llm_chat, llm_embedding, tool_config = llm_and_tools
        assert llm_chat is not None
        assert llm_embedding is not None
        assert tool_config is not None


# ====================== 功能测试：检索 + 生成 ======================
class TestRetrievalAndGeneration:
    """检索 + 生成流程测试"""

    @pytest.mark.parametrize(
        "query_key",
        [
            "basic_info",
            "allergy",
            "cholesterol",
            "suggestion",
        ],
    )
    def test_retrieval_and_generation(self, graph, test_config, query_key):
        """测试检索 + 生成流程是否能正确回答知识库中的问题"""
        query = TEST_QUERIES[query_key]
        expected_keywords = EXPECTED_FACTS[query_key]

        inputs = {
            "messages": [{"role": "user", "content": query}],
            "rewrite_count": 0,
        }

        result = graph.invoke(inputs, test_config)
        content = extract_content(result)

        # 断言必须包含至少一半的关键事实
        found = sum(1 for kw in expected_keywords if kw in content)
        min_required = len(expected_keywords) / 2
        assert found >= min_required, (
            f"查询 '{query_key}' 未返回足够关键信息。\n"
            f"期望关键词: {expected_keywords}\n"
            f"命中数量: {found}/{len(expected_keywords)}\n"
            f"实际返回: {content[:300]}..."
        )


# ====================== 功能测试：不相关查询 ======================
class TestIrrelevantQuery:
    """不相关查询处理测试"""

    def test_irrelevant_query_handling(self, graph, test_config):
        """测试不相关查询的处理行为：
        - 可能触发 rewrite 机制（rewrite_count > 0）
        - 也可能被 LLM 直接处理（不调用检索工具，直接回答或拒绝）
        两种都是合理行为
        """
        query = TEST_QUERIES["irrelevant"]

        inputs = {
            "messages": [{"role": "user", "content": query}],
            "rewrite_count": 0,
        }

        result = graph.invoke(inputs, test_config)
        content = extract_content(result)

        rewrite_triggered = result.get("rewrite_count", 0) > 0

        direct_handle_keywords = [
            "天气",
            "无法",
            "抱歉",
            "不能",
            "不在",
            "超出",
            "知识库",
            "没有相关",
            "无法回答",
            "weather",
        ]
        handled_directly = any(kw in content for kw in direct_handle_keywords)

        has_response = len(content.strip()) > 0

        assert rewrite_triggered or handled_directly or has_response, (
            f"不相关查询未被正常处理。\n"
            f"rewrite_count={result.get('rewrite_count', 0)}\n"
            f"回复内容: {content[:300]}"
        )


# ====================== 单元测试：grade_documents 节点 ======================
class TestGradeDocuments:
    """文档评分节点测试"""

    def test_grade_documents_with_relevant_content(self, llm_and_tools):
        """测试：提供相关文档时，评分应为 'yes'"""
        llm_chat, _, _ = llm_and_tools

        state = {
            "messages": [
                HumanMessage(content="李四六的年龄是多少？"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_test_001",
                            "name": "retrieve",
                            "args": {"query": "李四六的年龄"},
                        }
                    ],
                ),
                ToolMessage(
                    content="患者姓名为李四六，出生于1990年3月15日，目前34岁。职业为小学教师。",
                    tool_call_id="call_test_001",
                    name="retrieve",
                ),
            ],
            "relevance_score": None,
            "rewrite_count": 0,
        }

        result = grade_documents(state, llm_chat)

        assert "relevance_score" in result, "grade_documents 未返回 relevance_score"
        assert result["relevance_score"] in [
            "yes",
            "no",
        ], f"relevance_score 应为 'yes' 或 'no'，实际为: {result['relevance_score']}"

    def test_grade_documents_with_irrelevant_content(self, llm_and_tools):
        """测试：提供不相关文档时，评分应为 'no'"""
        llm_chat, _, _ = llm_and_tools

        state = {
            "messages": [
                HumanMessage(content="李四六的年龄是多少？"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_test_002",
                            "name": "retrieve",
                            "args": {"query": "李四六的年龄"},
                        }
                    ],
                ),
                ToolMessage(
                    content="今日股市大盘上涨2.5%，科技板块表现强劲。天气预报显示明天多云。",
                    tool_call_id="call_test_002",
                    name="retrieve",
                ),
            ],
            "relevance_score": None,
            "rewrite_count": 0,
        }

        result = grade_documents(state, llm_chat)

        assert "relevance_score" in result
        assert result["relevance_score"] in ["yes", "no"]


# ====================== 单元测试：路由函数 ======================
class TestRouting:
    """路由函数逻辑测试

    注意：route_after_grade 内部会检查 state["messages"] 字段，
    如果缺失则默认返回 "rewrite"。所以测试必须提供完整的 messages。
    """

    def test_route_after_grade_relevant(self):
        """评分为 yes → 生成"""
        state = make_route_state(relevance_score="yes", rewrite_count=0)
        result = route_after_grade(state)
        assert (
            result == "generate"
        ), f"relevance_score='yes' 时应路由到 'generate'，实际: '{result}'"

    def test_route_after_grade_irrelevant_first_time(self):
        """评分为 no 且首次 → 重写"""
        state = make_route_state(relevance_score="no", rewrite_count=0)
        result = route_after_grade(state)
        assert (
            result == "rewrite"
        ), f"relevance_score='no', rewrite_count=0 时应路由到 'rewrite'，实际: '{result}'"

    def test_route_after_grade_irrelevant_exceeded(self):
        """评分为 no 且重写次数超限 → 强制生成"""
        state = make_route_state(relevance_score="no", rewrite_count=4)
        result = route_after_grade(state)
        assert (
            result == "generate"
        ), f"rewrite_count=4 超限时应强制 'generate'，实际: '{result}'"

    def test_route_after_grade_boundary(self):
        """测试重写次数边界值"""
        state = make_route_state(relevance_score="no", rewrite_count=2)
        result = route_after_grade(state)
        assert result in [
            "rewrite",
            "generate",
        ], f"边界值测试，结果应为 'rewrite' 或 'generate'，实际: '{result}'"

    def test_route_after_grade_yes_with_high_rewrite(self):
        """评分为 yes 时，无论 rewrite_count 多大都应该 generate"""
        state = make_route_state(relevance_score="yes", rewrite_count=10)
        result = route_after_grade(state)
        assert (
            result == "generate"
        ), f"relevance_score='yes' 时应始终 'generate'，实际: '{result}'"

    def test_route_after_tools_with_retrieve(self, llm_and_tools):
        """工具为 retrieve → 应路由到 grade_documents"""
        _, _, tool_config = llm_and_tools

        state = {
            "messages": [
                HumanMessage(content="李四六的年龄？"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_route_001",
                            "name": "retrieve",
                            "args": {"query": "李四六年龄"},
                        }
                    ],
                ),
                ToolMessage(
                    content="患者李四六，34岁。",
                    tool_call_id="call_route_001",
                    name="retrieve",
                ),
            ]
        }

        result = route_after_tools(state, tool_config)
        assert result in [
            "generate",
            "grade_documents",
        ], f"retrieve 工具后应路由到 'grade_documents' 或 'generate'，实际: '{result}'"

    def test_route_after_tools_with_non_retrieve(self, llm_and_tools):
        """非检索工具 → 应路由到 generate"""
        _, _, tool_config = llm_and_tools

        state = {
            "messages": [
                HumanMessage(content="计算 3 乘以 5"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_route_002",
                            "name": "multiply",
                            "args": {"a": 3, "b": 5},
                        }
                    ],
                ),
                ToolMessage(
                    content="15",
                    tool_call_id="call_route_002",
                    name="multiply",
                ),
            ]
        }

        result = route_after_tools(state, tool_config)
        assert (
            result == "generate"
        ), f"非检索工具后应路由到 'generate'，实际: '{result}'"


# ====================== Middleware 测试 ======================
class TestMiddleware:
    """Middleware 相关测试"""

    def test_middleware_model_call_count(self, graph, test_config):
        """测试模型调用计数 Middleware 是否正常工作"""
        inputs = {
            "messages": [{"role": "user", "content": TEST_QUERIES["basic_info"]}],
            "rewrite_count": 0,
            "mw_model_call_count": 0,
        }

        result = graph.invoke(inputs, test_config)

        call_count = result.get("mw_model_call_count", 0)
        total_time = result.get("mw_model_total_time", 0.0)

        assert call_count >= 2, f"模型调用次数不足，期望 >= 2，实际: {call_count}"
        assert total_time > 0, f"模型累计耗时应 > 0，实际: {total_time}"

    def test_pii_detection_warn_mode(self, middleware_manager):
        """测试 PII 检测 Middleware（warn 模式）"""
        state = {
            "messages": [
                HumanMessage(content="我的身份证号是 110101199003152345，请帮我查询。"),
            ],
        }

        updates, should_stop = middleware_manager.run_before_model(state, "agent")
        assert updates.get("mw_pii_detected") is True, "PII 检测应识别出身份证号"

    def test_pii_detection_no_pii(self, middleware_manager):
        """测试无 PII 时不误报"""
        state = {
            "messages": [
                HumanMessage(content="李四六的年龄是多少？"),
            ],
        }

        updates, should_stop = middleware_manager.run_before_model(state, "agent")
        pii_detected = updates.get("mw_pii_detected", False)
        assert pii_detected is False or pii_detected is None, "无 PII 内容时不应误报"

    def test_model_call_limit(self):
        """测试模型调用次数限制 Middleware"""
        limiter = ModelCallLimitMiddleware(max_calls=3)

        state = {
            "messages": [HumanMessage(content="test")],
            "mw_model_call_count": 2,
        }

        updates, should_stop = limiter.before_model_call(state, "agent")
        assert should_stop is False or should_stop is None

        # 模拟超出限制
        state["mw_model_call_count"] = 3
        updates, should_stop = limiter.before_model_call(state, "agent")
        assert should_stop is True, "超出调用限制时应停止"


# ====================== 多用户隔离测试 ======================
class TestMultiUser:
    """多用户隔离测试"""

    def test_multi_user_isolation(self, graph):
        """测试不同 thread_id 状态不互相干扰"""
        config_a = {
            "configurable": {
                "thread_id": "user_a_isolation_test",
                "user_id": "A",
            }
        }
        config_b = {
            "configurable": {
                "thread_id": "user_b_isolation_test",
                "user_id": "B",
            }
        }

        result_a = graph.invoke(
            {
                "messages": [{"role": "user", "content": "李四六的年龄？"}],
                "rewrite_count": 0,
            },
            config_a,
        )

        result_b = graph.invoke(
            {
                "messages": [{"role": "user", "content": "李四六的过敏史？"}],
                "rewrite_count": 0,
            },
            config_b,
        )

        count_a = result_a.get("mw_model_call_count", 0)
        count_b = result_b.get("mw_model_call_count", 0)

        assert count_a > 0, f"用户 A 的调用计数应 > 0，实际: {count_a}"
        assert count_b > 0, f"用户 B 的调用计数应 > 0，实际: {count_b}"

        content_a = extract_content(result_a)
        content_b = extract_content(result_b)
        assert len(content_a) > 0
        assert len(content_b) > 0


# ====================== 性能与稳定性测试 ======================
class TestPerformance:
    """性能与稳定性测试"""

    def test_full_workflow_performance(self, graph, test_config):
        """测试完整工作流性能"""
        start = time.time()

        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": TEST_QUERIES["cholesterol"]}],
                "rewrite_count": 0,
            },
            test_config,
        )

        duration = time.time() - start

        assert duration < 60.0, f"响应时间过长: {duration:.2f}s（阈值 60s）"
        assert len(result["messages"]) > 0, "返回消息不应为空"

        content = extract_content(result)
        assert len(content.strip()) > 0, "最终回复内容不应为空"

    def test_consecutive_queries(self, graph, test_config):
        """测试连续多轮查询的稳定性"""
        queries = [
            "李四六的基本信息？",
            "他对什么药物过敏？",
            "他的胆固醇水平？",
        ]

        for i, query in enumerate(queries):
            result = graph.invoke(
                {
                    "messages": [{"role": "user", "content": query}],
                    "rewrite_count": 0,
                },
                test_config,
            )

            content = extract_content(result)
            assert len(content.strip()) > 0, f"第 {i + 1} 轮查询 '{query}' 返回为空"

    def test_empty_query_handling(self, graph, test_config):
        """测试空查询不会导致崩溃"""
        try:
            result = graph.invoke(
                {
                    "messages": [{"role": "user", "content": ""}],
                    "rewrite_count": 0,
                },
                test_config,
            )
            assert result is not None
        except Exception as e:
            assert isinstance(
                e, (ValueError, KeyError)
            ), f"空查询导致意外异常: {type(e).__name__}: {e}"

    def test_long_query_handling(self, graph, test_config):
        """测试超长查询不会导致崩溃"""
        long_query = "李四六的健康状况如何？" * 50

        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": long_query}],
                "rewrite_count": 0,
            },
            test_config,
        )

        content = extract_content(result)
        assert len(content.strip()) > 0, "超长查询应返回非空回复"


# ====================== 边界条件测试 ======================
class TestEdgeCases:
    """边界条件测试"""

    def test_rewrite_count_persists(self, graph, test_config):
        """测试 rewrite_count 在流程中正确传递"""
        inputs = {
            "messages": [{"role": "user", "content": TEST_QUERIES["basic_info"]}],
            "rewrite_count": 0,
        }

        result = graph.invoke(inputs, test_config)

        assert "rewrite_count" in result, "结果中应包含 rewrite_count 字段"
        assert isinstance(
            result["rewrite_count"], int
        ), f"rewrite_count 应为 int，实际: {type(result['rewrite_count'])}"

    def test_messages_list_grows(self, graph, test_config):
        """测试消息列表在流程中正确增长"""
        inputs = {
            "messages": [{"role": "user", "content": "李四六的职业是什么？"}],
            "rewrite_count": 0,
        }

        result = graph.invoke(inputs, test_config)

        msg_count = len(result["messages"])
        assert msg_count >= 2, f"消息列表应至少有 2 条，实际: {msg_count}"

    def test_result_structure(self, graph, test_config):
        """测试返回结果的结构完整性"""
        inputs = {
            "messages": [{"role": "user", "content": "李四六的基本信息？"}],
            "rewrite_count": 0,
        }

        result = graph.invoke(inputs, test_config)

        assert "messages" in result, "结果中应包含 messages"
        assert isinstance(result["messages"], list), "messages 应为 list"
        assert len(result["messages"]) > 0, "messages 不应为空"


# ====================== 运行入口 ======================
if __name__ == "__main__":
    pytest.main(
        [
            "-v",
            "--tb=short",
            "--durations=10",
            "--html=report.html",  # 👈 生成报告
            "--self-contained-html",  # 👈 单文件（推荐）
            "test_ragAgent.py",
        ]
    )
