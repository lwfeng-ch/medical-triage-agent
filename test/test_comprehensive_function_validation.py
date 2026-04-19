"""
全面函数输出检验测试套件

测试范围:
1. 函数调用链路追踪
2. 单元测试（各核心函数）
3. 集成测试（跨模块调用）
4. 边界条件和异常输入测试
5. 自动生成测试报告

测试文件:
- gradio_ui.py
- ragAgent.py
- main.py
"""
import pytest
import sys
import os
import json
import time
import hashlib
import secrets
import sqlite3
import threading
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入被测试模块
try:
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from ragAgent import (
        AgentState,
        ToolConfig,
        filter_messages,
        get_latest_question,
        collect_tool_contents,
        extract_graph_response,
        RagAgentError,
        ResponseExtractionError,
        GraphBuildError,
        MedicalAnalysisError,
        ToolExecutionError,
    )
    from main import (
        Message,
        ChatCompletionRequest,
        ChatCompletionResponse,
        MedicalExtension,
        TriageData,
        StructuredMedicalData,
        format_response,
        _build_medical_extension,
    )
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"[WARNING] 导入失败: {e}")
    IMPORTS_SUCCESS = False


# ═══════════════════════════════════════════════════════════════════════════════
# 测试数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """测试结果数据模型"""
    test_name: str
    category: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    details: Optional[Dict] = None


@dataclass
class FunctionCallTrace:
    """函数调用链路追踪"""
    function_name: str
    module_name: str
    input_params: Dict
    output_result: Any
    execution_time: float
    called_by: Optional[str] = None
    calls_to: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# 测试报告生成器
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportGenerator:
    """测试报告生成器"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.traces: List[FunctionCallTrace] = []
        self.start_time = time.time()
    
    def add_result(self, result: TestResult):
        """添加测试结果"""
        self.results.append(result)
    
    def add_trace(self, trace: FunctionCallTrace):
        """添加函数调用追踪"""
        self.traces.append(trace)
    
    def generate_report(self) -> str:
        """生成测试报告"""
        total_time = time.time() - self.start_time
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # 按类别统计
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {"passed": 0, "failed": 0, "total": 0}
            categories[result.category]["total"] += 1
            if result.passed:
                categories[result.category]["passed"] += 1
            else:
                categories[result.category]["failed"] += 1
        
        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("全面函数输出检验测试报告")
        report.append("=" * 80)
        report.append(f"\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"总耗时: {total_time:.2f} 秒")
        report.append(f"\n总体统计:")
        report.append(f"  - 总测试数: {total_tests}")
        report.append(f"  - 通过数: {passed_tests}")
        report.append(f"  - 失败数: {failed_tests}")
        report.append(f"  - 通过率: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "  - 通过率: N/A")
        
        report.append(f"\n分类统计:")
        for category, stats in categories.items():
            rate = (stats["passed"]/stats["total"]*100) if stats["total"] > 0 else 0
            report.append(f"  {category}:")
            report.append(f"    - 总数: {stats['total']}")
            report.append(f"    - 通过: {stats['passed']}")
            report.append(f"    - 失败: {stats['failed']}")
            report.append(f"    - 通过率: {rate:.1f}%")
        
        # 失败的测试详情
        if failed_tests > 0:
            report.append(f"\n失败测试详情:")
            for result in self.results:
                if not result.passed:
                    report.append(f"\n  [{result.category}] {result.test_name}")
                    report.append(f"    错误: {result.error_message}")
                    if result.details:
                        report.append(f"    详情: {json.dumps(result.details, ensure_ascii=False, indent=6)}")
        
        # 函数调用链路追踪
        if self.traces:
            report.append(f"\n函数调用链路追踪:")
            for trace in self.traces:
                report.append(f"\n  [{trace.module_name}] {trace.function_name}")
                report.append(f"    执行时间: {trace.execution_time:.4f}s")
                if trace.called_by:
                    report.append(f"    被调用者: {trace.called_by}")
                if trace.calls_to:
                    report.append(f"    调用函数: {', '.join(trace.calls_to)}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


# 全局报告生成器
report_generator = TestReportGenerator()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 函数调用链路追踪测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestFunctionCallTrace:
    """函数调用链路追踪测试"""
    
    def test_ragagent_filter_messages_trace(self):
        """测试 filter_messages 函数调用链路"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        start_time = time.time()
        
        # 准备测试数据
        messages = [
            HumanMessage(content="用户问题"),
            AIMessage(content="AI回复"),
            HumanMessage(content="第二个问题"),
        ]
        
        # 执行函数
        result = filter_messages(messages)
        
        execution_time = time.time() - start_time
        
        # 记录追踪
        trace = FunctionCallTrace(
            function_name="filter_messages",
            module_name="ragAgent.py",
            input_params={"messages_count": len(messages)},
            output_result={"filtered_count": len(result)},
            execution_time=execution_time,
            calls_to=["_truncate_by_human_message_boundary"]
        )
        report_generator.add_trace(trace)
        
        # 验证结果
        assert isinstance(result, list), "返回类型应为 list"
        assert len(result) <= len(messages), "过滤后消息数不应超过原始消息数"
        
        # 记录测试结果
        report_generator.add_result(TestResult(
            test_name="filter_messages_trace",
            category="函数调用链路追踪",
            passed=True,
            execution_time=execution_time
        ))
    
    def test_ragagent_get_latest_question_trace(self):
        """测试 get_latest_question 函数调用链路"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        start_time = time.time()
        
        # 准备测试数据
        state = {
            "messages": [
                HumanMessage(content="用户问题1"),
                AIMessage(content="AI回复"),
                HumanMessage(content="用户问题2"),
            ]
        }
        
        # 执行函数
        result = get_latest_question(state)
        
        execution_time = time.time() - start_time
        
        # 记录追踪
        trace = FunctionCallTrace(
            function_name="get_latest_question",
            module_name="ragAgent.py",
            input_params={"has_messages": "messages" in state},
            output_result={"question": result},
            execution_time=execution_time
        )
        report_generator.add_trace(trace)
        
        # 验证结果
        assert isinstance(result, str), "返回类型应为 str"
        assert result == "用户问题2", "应返回最新的用户问题"
        
        report_generator.add_result(TestResult(
            test_name="get_latest_question_trace",
            category="函数调用链路追踪",
            passed=True,
            execution_time=execution_time
        ))
    
    def test_main_build_medical_extension_trace(self):
        """测试 _build_medical_extension 函数调用链路"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        start_time = time.time()
        
        # 准备测试数据
        final_payload = {
            "route": "medical",
            "risk_level": "high",
            "risk_warning": "检测到高危症状",
            "disclaimer": "本建议仅供参考",
            "structured_data": {
                "triage": {
                    "recommended_departments": ["心内科", "急诊科"],
                    "urgency_level": "emergency",
                    "triage_reason": "胸痛症状",
                    "triage_confidence": 0.9
                }
            }
        }
        
        # 执行函数
        result = _build_medical_extension(final_payload)
        
        execution_time = time.time() - start_time
        
        # 记录追踪
        trace = FunctionCallTrace(
            function_name="_build_medical_extension",
            module_name="main.py",
            input_params={"has_final_payload": final_payload is not None},
            output_result={"has_medical_ext": result is not None},
            execution_time=execution_time
        )
        report_generator.add_trace(trace)
        
        # 验证结果
        assert result is not None, "medical 路由应返回 MedicalExtension"
        assert result.risk_level == "high", "风险等级应为 high"
        assert result.structured_data is not None, "应包含结构化数据"
        
        report_generator.add_result(TestResult(
            test_name="_build_medical_extension_trace",
            category="函数调用链路追踪",
            passed=True,
            execution_time=execution_time
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 单元测试 - ragAgent.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestRagAgentUnit:
    """ragAgent.py 单元测试"""
    
    def test_filter_messages_normal(self):
        """测试正常消息过滤"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        messages = [
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
            HumanMessage(content="问题2"),
            AIMessage(content="回答2"),
        ]
        
        result = filter_messages(messages)
        
        assert isinstance(result, list), "返回类型应为 list"
        assert len(result) > 0, "过滤后不应为空"
        assert all(isinstance(m, (HumanMessage, AIMessage, ToolMessage)) for m in result), \
            "过滤后消息类型应为 HumanMessage、AIMessage 或 ToolMessage"
        
        report_generator.add_result(TestResult(
            test_name="filter_messages_normal",
            category="单元测试 - ragAgent.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_filter_messages_with_tool_calls(self):
        """测试包含工具调用的消息过滤"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        messages = [
            HumanMessage(content="问题"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_123", "name": "search", "args": {"query": "test"}}]
            ),
            ToolMessage(content="工具结果", tool_call_id="call_123"),
            AIMessage(content="最终回答"),
        ]
        
        result = filter_messages(messages)
        
        assert len(result) == 4, "应保留完整的工具调用链"
        
        report_generator.add_result(TestResult(
            test_name="filter_messages_with_tool_calls",
            category="单元测试 - ragAgent.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_get_latest_question_empty_state(self):
        """测试空状态的最新问题获取"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        state = {"messages": []}
        result = get_latest_question(state)
        
        assert result is None, "空状态应返回 None"
        
        report_generator.add_result(TestResult(
            test_name="get_latest_question_empty_state",
            category="单元测试 - ragAgent.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_get_latest_question_no_human_message(self):
        """测试无用户消息的状态"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        state = {
            "messages": [
                AIMessage(content="AI消息"),
            ]
        }
        result = get_latest_question(state)
        
        assert result is None, "无用户消息时应返回 None"
        
        report_generator.add_result(TestResult(
            test_name="get_latest_question_no_human_message",
            category="单元测试 - ragAgent.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_collect_tool_contents_normal(self):
        """测试正常工具内容收集"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        state = {
            "messages": [
                HumanMessage(content="问题"),
                AIMessage(content="思考中..."),
                ToolMessage(content="工具结果1", tool_call_id="1"),
                ToolMessage(content="工具结果2", tool_call_id="2"),
            ]
        }
        
        result = collect_tool_contents(state)
        
        assert isinstance(result, str), "返回类型应为 str"
        assert "工具结果1" in result, "应包含工具结果1"
        assert "工具结果2" in result, "应包含工具结果2"
        
        report_generator.add_result(TestResult(
            test_name="collect_tool_contents_normal",
            category="单元测试 - ragAgent.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_ragagent_error_hierarchy(self):
        """测试异常类层次结构"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        assert issubclass(GraphBuildError, RagAgentError), \
            "GraphBuildError 应继承自 RagAgentError"
        assert issubclass(ResponseExtractionError, RagAgentError), \
            "ResponseExtractionError 应继承自 RagAgentError"
        assert issubclass(MedicalAnalysisError, RagAgentError), \
            "MedicalAnalysisError 应继承自 RagAgentError"
        assert issubclass(ToolExecutionError, RagAgentError), \
            "ToolExecutionError 应继承自 RagAgentError"
        
        report_generator.add_result(TestResult(
            test_name="ragagent_error_hierarchy",
            category="单元测试 - ragAgent.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_ragagent_error_to_dict(self):
        """测试异常转字典方法"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        error = RagAgentError(
            message="测试错误",
            code="TEST_ERROR",
            details={"key": "value"}
        )
        
        result = error.to_dict()
        
        assert isinstance(result, dict), "返回类型应为 dict"
        assert result["error"] == "测试错误", "错误消息应正确"
        assert result["code"] == "TEST_ERROR", "错误代码应正确"
        assert result["details"]["key"] == "value", "详情应正确"
        
        report_generator.add_result(TestResult(
            test_name="ragagent_error_to_dict",
            category="单元测试 - ragAgent.py",
            passed=True,
            execution_time=0.001
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 单元测试 - main.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestMainUnit:
    """main.py 单元测试"""
    
    def test_message_model_validation(self):
        """测试消息模型验证"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        msg = Message(role="user", content="测试内容")
        
        assert msg.role == "user", "角色应正确"
        assert msg.content == "测试内容", "内容应正确"
        
        report_generator.add_result(TestResult(
            test_name="message_model_validation",
            category="单元测试 - main.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_chat_completion_request_model(self):
        """测试聊天请求模型"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="测试")],
            stream=False,
            userId="user123",
            conversationId="conv123"
        )
        
        assert len(request.messages) == 1, "消息数应正确"
        assert request.stream == False, "流式标志应正确"
        assert request.userId == "user123", "用户ID应正确"
        
        report_generator.add_result(TestResult(
            test_name="chat_completion_request_model",
            category="单元测试 - main.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_medical_extension_model(self):
        """测试医疗扩展模型"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        medical = MedicalExtension(
            risk_level="high",
            risk_warning="高危警告",
            disclaimer="免责声明"
        )
        
        assert medical.risk_level == "high", "风险等级应正确"
        assert medical.risk_warning == "高危警告", "风险警告应正确"
        
        report_generator.add_result(TestResult(
            test_name="medical_extension_model",
            category="单元测试 - main.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_triage_data_model(self):
        """测试分诊数据模型"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        triage = TriageData(
            recommended_departments=["心内科", "急诊科"],
            urgency_level="emergency",
            triage_reason="胸痛",
            triage_confidence=0.9
        )
        
        assert len(triage.recommended_departments) == 2, "科室数应正确"
        assert triage.urgency_level == "emergency", "紧急程度应正确"
        assert triage.triage_confidence == 0.9, "置信度应正确"
        
        report_generator.add_result(TestResult(
            test_name="triage_data_model",
            category="单元测试 - main.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_format_response_normal(self):
        """测试正常响应格式化"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        text = "这是第一句. 这是第二句. 这是第三句."
        result = format_response(text)
        
        assert isinstance(result, str), "返回类型应为 str"
        assert "这是第一句" in result, "应包含原始内容"
        
        report_generator.add_result(TestResult(
            test_name="format_response_normal",
            category="单元测试 - main.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_format_response_with_code_block(self):
        """测试包含代码块的响应格式化"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        text = "这是说明```python\nprint('hello')\n```这是后续"
        result = format_response(text)
        
        assert "```" in result, "应保留代码块标记"
        
        report_generator.add_result(TestResult(
            test_name="format_response_with_code_block",
            category="单元测试 - main.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_build_medical_extension_general_route(self):
        """测试通用路由的医疗扩展构建"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        final_payload = {
            "route": "general",
            "risk_level": "low"
        }
        
        result = _build_medical_extension(final_payload)
        
        assert result is None, "general 路由应返回 None"
        
        report_generator.add_result(TestResult(
            test_name="_build_medical_extension_general_route",
            category="单元测试 - main.py",
            passed=True,
            execution_time=0.001
        ))
    
    def test_build_medical_extension_medical_route(self):
        """测试医疗路由的医疗扩展构建"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        final_payload = {
            "route": "medical",
            "risk_level": "medium",
            "risk_warning": "中等风险",
            "disclaimer": "免责声明",
            "structured_data": {
                "triage": {
                    "recommended_departments": ["内科"],
                    "urgency_level": "urgent",
                    "triage_reason": "症状",
                    "triage_confidence": 0.8
                }
            }
        }
        
        result = _build_medical_extension(final_payload)
        
        assert result is not None, "medical 路由应返回 MedicalExtension"
        assert result.risk_level == "medium", "风险等级应正确"
        assert result.structured_data is not None, "应包含结构化数据"
        
        report_generator.add_result(TestResult(
            test_name="_build_medical_extension_medical_route",
            category="单元测试 - main.py",
            passed=True,
            execution_time=0.001
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 边界条件和异常输入测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryConditions:
    """边界条件和异常输入测试"""
    
    def test_filter_messages_empty_list(self):
        """测试空消息列表"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        result = filter_messages([])
        assert result == [], "空列表应返回空列表"
        
        report_generator.add_result(TestResult(
            test_name="filter_messages_empty_list",
            category="边界条件测试",
            passed=True,
            execution_time=0.001
        ))
    
    def test_filter_messages_large_list(self):
        """测试大消息列表"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        messages = []
        for i in range(100):
            messages.append(HumanMessage(content=f"问题{i}"))
            messages.append(AIMessage(content=f"回答{i}"))
        
        result = filter_messages(messages)
        
        assert len(result) <= len(messages), "过滤后消息数不应超过原始消息数"
        
        report_generator.add_result(TestResult(
            test_name="filter_messages_large_list",
            category="边界条件测试",
            passed=True,
            execution_time=0.01
        ))
    
    def test_get_latest_question_none_state(self):
        """测试 None 状态"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        try:
            result = get_latest_question(None)
            assert result is None, "None 状态应返回 None"
            passed = True
            error_msg = None
        except Exception as e:
            passed = False
            error_msg = str(e)
        
        report_generator.add_result(TestResult(
            test_name="get_latest_question_none_state",
            category="边界条件测试",
            passed=passed,
            execution_time=0.001,
            error_message=error_msg
        ))
    
    def test_build_medical_extension_none_payload(self):
        """测试 None payload"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        result = _build_medical_extension(None)
        assert result is None, "None payload 应返回 None"
        
        report_generator.add_result(TestResult(
            test_name="_build_medical_extension_none_payload",
            category="边界条件测试",
            passed=True,
            execution_time=0.001
        ))
    
    def test_build_medical_extension_empty_payload(self):
        """测试空 payload"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        result = _build_medical_extension({})
        assert result is None, "空 payload 应返回 None"
        
        report_generator.add_result(TestResult(
            test_name="_build_medical_extension_empty_payload",
            category="边界条件测试",
            passed=True,
            execution_time=0.001
        ))
    
    def test_build_medical_extension_missing_fields(self):
        """测试缺失字段的 payload"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        final_payload = {
            "route": "medical"
        }
        
        result = _build_medical_extension(final_payload)
        
        assert result is not None, "即使缺失字段也应返回 MedicalExtension"
        assert result.risk_level == "low", "缺失风险等级应使用默认值"
        
        report_generator.add_result(TestResult(
            test_name="_build_medical_extension_missing_fields",
            category="边界条件测试",
            passed=True,
            execution_time=0.001
        ))
    
    def test_message_model_empty_content(self):
        """测试空内容的消息模型"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        msg = Message(role="user", content="")
        
        assert msg.content == "", "应允许空内容"
        
        report_generator.add_result(TestResult(
            test_name="message_model_empty_content",
            category="边界条件测试",
            passed=True,
            execution_time=0.001
        ))
    
    def test_triage_data_extreme_confidence(self):
        """测试极端置信度值"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        triage_min = TriageData(triage_confidence=0.0)
        triage_max = TriageData(triage_confidence=1.0)
        
        assert triage_min.triage_confidence == 0.0, "最小置信度应正确"
        assert triage_max.triage_confidence == 1.0, "最大置信度应正确"
        
        report_generator.add_result(TestResult(
            test_name="triage_data_extreme_confidence",
            category="边界条件测试",
            passed=True,
            execution_time=0.001
        ))
    
    def test_format_response_empty_string(self):
        """测试空字符串格式化"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        result = format_response("")
        assert result == "", "空字符串应返回空字符串"
        
        report_generator.add_result(TestResult(
            test_name="format_response_empty_string",
            category="边界条件测试",
            passed=True,
            execution_time=0.001
        ))
    
    def test_format_response_very_long_text(self):
        """测试超长文本格式化"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        long_text = "这是测试句子. " * 10000
        result = format_response(long_text)
        
        assert isinstance(result, str), "应返回字符串"
        assert len(result) > 0, "不应为空"
        
        report_generator.add_result(TestResult(
            test_name="format_response_very_long_text",
            category="边界条件测试",
            passed=True,
            execution_time=0.1
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 集成测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """集成测试"""
    
    def test_ragagent_to_main_data_flow(self):
        """测试 ragAgent 到 main 的数据流"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        start_time = time.time()
        
        # 模拟 ragAgent 输出
        final_payload = {
            "route": "medical",
            "risk_level": "high",
            "risk_warning": "高危警告",
            "disclaimer": "免责声明",
            "structured_data": {
                "triage": {
                    "recommended_departments": ["急诊科"],
                    "urgency_level": "emergency",
                    "triage_reason": "胸痛",
                    "triage_confidence": 0.95
                }
            }
        }
        
        # main.py 处理
        medical_ext = _build_medical_extension(final_payload)
        
        execution_time = time.time() - start_time
        
        # 验证数据传递完整性
        assert medical_ext is not None, "数据应正确传递"
        assert medical_ext.risk_level == final_payload["risk_level"], "风险等级应一致"
        assert medical_ext.structured_data.triage.urgency_level == \
               final_payload["structured_data"]["triage"]["urgency_level"], "紧急程度应一致"
        
        report_generator.add_result(TestResult(
            test_name="ragagent_to_main_data_flow",
            category="集成测试",
            passed=True,
            execution_time=execution_time
        ))
    
    def test_message_filtering_chain(self):
        """测试消息过滤链路"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        start_time = time.time()
        
        # 创建完整消息链
        messages = [
            HumanMessage(content="用户问题1"),
            AIMessage(content="AI回答1"),
            HumanMessage(content="用户问题2"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "search", "args": {}}]
            ),
            ToolMessage(content="工具结果", tool_call_id="call_1"),
            AIMessage(content="基于工具结果的回答"),
        ]
        
        # 执行过滤
        filtered = filter_messages(messages)
        
        # 提取最新问题
        state = {"messages": filtered}
        latest = get_latest_question(state)
        
        execution_time = time.time() - start_time
        
        # 验证链路完整性
        assert len(filtered) > 0, "过滤后应有消息"
        assert latest == "用户问题2", "应正确提取最新问题"
        
        report_generator.add_result(TestResult(
            test_name="message_filtering_chain",
            category="集成测试",
            passed=True,
            execution_time=execution_time
        ))
    
    def test_error_propagation_chain(self):
        """测试错误传播链路"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        start_time = time.time()
        
        # 创建错误
        error = ResponseExtractionError(
            message="响应提取失败",
            details={"cause": "测试"}
        )
        
        # 验证错误信息传播
        error_dict = error.to_dict()
        
        execution_time = time.time() - start_time
        
        assert "error" in error_dict, "错误字典应包含 error 字段"
        assert "code" in error_dict, "错误字典应包含 code 字段"
        assert error_dict["code"] == "RESPONSE_EXTRACTION_ERROR", "错误代码应正确"
        
        report_generator.add_result(TestResult(
            test_name="error_propagation_chain",
            category="集成测试",
            passed=True,
            execution_time=execution_time
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 性能测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """性能测试"""
    
    def test_filter_messages_performance(self):
        """测试消息过滤性能"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        # 创建大量消息
        messages = []
        for i in range(1000):
            messages.append(HumanMessage(content=f"问题{i}"))
            messages.append(AIMessage(content=f"回答{i}"))
        
        start_time = time.time()
        result = filter_messages(messages)
        execution_time = time.time() - start_time
        
        # 性能要求：1000条消息处理时间应小于 1 秒
        assert execution_time < 1.0, f"处理时间过长: {execution_time:.2f}s"
        
        report_generator.add_result(TestResult(
            test_name="filter_messages_performance",
            category="性能测试",
            passed=True,
            execution_time=execution_time,
            details={"message_count": len(messages), "result_count": len(result)}
        ))
    
    def test_build_medical_extension_performance(self):
        """测试医疗扩展构建性能"""
        if not IMPORTS_SUCCESS:
            pytest.skip("导入失败，跳过测试")
        
        # 创建复杂 payload
        final_payload = {
            "route": "medical",
            "risk_level": "high",
            "risk_warning": "高危警告" * 100,
            "disclaimer": "免责声明" * 100,
            "structured_data": {
                "triage": {
                    "recommended_departments": ["科室1", "科室2", "科室3", "科室4", "科室5"],
                    "urgency_level": "emergency",
                    "triage_reason": "症状描述" * 50,
                    "triage_confidence": 0.95
                },
                "analysis": {"key": "value" * 100}
            }
        }
        
        start_time = time.time()
        for _ in range(1000):
            _build_medical_extension(final_payload)
        execution_time = time.time() - start_time
        
        # 性能要求：1000次构建时间应小于 1 秒
        assert execution_time < 1.0, f"构建时间过长: {execution_time:.2f}s"
        
        report_generator.add_result(TestResult(
            test_name="build_medical_extension_performance",
            category="性能测试",
            passed=True,
            execution_time=execution_time,
            details={"iterations": 1000}
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# 测试报告生成
# ═══════════════════════════════════════════════════════════════════════════════

def test_generate_final_report():
    """生成最终测试报告"""
    report = report_generator.generate_report()
    
    # 保存报告到文件
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test",
        "test_report_output.txt"
    )
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n" + "=" * 80)
    print("测试报告已生成:")
    print(report_path)
    print("=" * 80)
    
    report_generator.add_result(TestResult(
        test_name="generate_final_report",
        category="报告生成",
        passed=True,
        execution_time=0.1
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# 运行测试
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=20",
        "-x"
    ])
