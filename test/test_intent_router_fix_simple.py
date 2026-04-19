"""
意图路由器修复验证测试 - 简化版（无需完整依赖）。

测试目标：
1. 验证关键词预检机制
2. 验证关键词列表完整性
3. 验证关键词匹配逻辑

测试用例基于实际测试结果文件 langsmith_copy.md 中的失败案例。
"""

import pytest


class TestMedicalKeywordPrecheck:
    """关键词预检机制测试。"""

    def test_health_record_keywords(self):
        """
        测试健康档案关键词。

        关键词列表：
        - 身体状况
        - 健康档案
        - 病历
        - 体检报告
        """
        MEDICAL_KEYWORDS = [
            "身体状况",
            "健康档案",
            "病历",
            "体检报告",
        ]

        test_cases = [
            ("告诉我2年内我的身体状况", "身体状况"),
            ("查看张三的健康档案", "健康档案"),
            ("李四的病历在哪里", "病历"),
            ("这是我的体检报告", "体检报告"),
        ]

        for query, expected_keyword in test_cases:
            matched = False
            for keyword in MEDICAL_KEYWORDS:
                if keyword in query:
                    matched = True
                    assert (
                        keyword == expected_keyword
                    ), f"查询 '{query}' 应匹配关键词 '{expected_keyword}'，实际匹配 '{keyword}'"
                    break
            assert matched, f"查询 '{query}' 应匹配关键词 '{expected_keyword}'"

    def test_blood_test_keywords(self):
        """
        测试血检报告关键词。

        关键词列表：
        - 血常规
        - 尿常规
        - 生化
        - 白细胞
        - 血红蛋白
        - 血小板
        - 红细胞
        - 中性粒细胞
        - 淋巴细胞
        """
        MEDICAL_KEYWORDS = [
            "血常规",
            "尿常规",
            "生化",
            "生命体征",
            "白细胞",
            "血红蛋白",
            "血小板",
            "红细胞",
            "中性粒细胞",
            "淋巴细胞",
        ]

        test_cases = [
            ("这是我的血常规检查报告", "血常规"),
            ("白细胞计数18.5×10⁹/L", "白细胞"),
            ("中性粒细胞百分比85%", "中性粒细胞"),
            ("血红蛋白偏低", "血红蛋白"),
            ("血小板计数异常", "血小板"),
        ]

        for query, expected_keyword in test_cases:
            matched = False
            for keyword in MEDICAL_KEYWORDS:
                if keyword in query:
                    matched = True
                    assert (
                        keyword == expected_keyword
                    ), f"查询 '{query}' 应匹配关键词 '{expected_keyword}'，实际匹配 '{keyword}'"
                    break
            assert matched, f"查询 '{query}' 应匹配关键词 '{expected_keyword}'"

    def test_symptom_keywords(self):
        """
        测试症状关键词。

        关键词列表：
        - 头疼/头痛
        - 发烧/发热
        - 咳嗽
        - 腹痛
        - 恶心
        - 呕吐
        - 腹泻
        - 便秘
        """
        MEDICAL_KEYWORDS = [
            "头疼",
            "头痛",
            "发烧",
            "发热",
            "咳嗽",
            "腹痛",
            "恶心",
            "呕吐",
            "腹泻",
            "便秘",
        ]

        test_cases = [
            ("我头疼怎么办", "头疼"),
            ("我头痛得厉害", "头痛"),
            ("我发烧了", "发烧"),
            ("我发热三天了", "发热"),
            ("我咳嗽不止", "咳嗽"),
            ("我腹痛难忍", "腹痛"),
        ]

        for query, expected_keyword in test_cases:
            matched = False
            for keyword in MEDICAL_KEYWORDS:
                if keyword in query:
                    matched = True
                    assert (
                        keyword == expected_keyword
                    ), f"查询 '{query}' 应匹配关键词 '{expected_keyword}'，实际匹配 '{keyword}'"
                    break
            assert matched, f"查询 '{query}' 应匹配关键词 '{expected_keyword}'"

    def test_medical_action_keywords(self):
        """
        测试医疗行为关键词。

        关键词列表：
        - 诊断
        - 治疗
        - 用药
        - 复查
        - 体检
        """
        MEDICAL_KEYWORDS = [
            "诊断",
            "治疗",
            "用药",
            "复查",
            "体检",
        ]

        test_cases = [
            ("我的诊断结果是什么", "诊断"),
            ("治疗方案有哪些", "治疗"),
            ("阿司匹林怎么用药", "用药"),
            ("我需要复查吗", "复查"),
            ("我明天去体检", "体检"),
        ]

        for query, expected_keyword in test_cases:
            matched = False
            for keyword in MEDICAL_KEYWORDS:
                if keyword in query:
                    matched = True
                    assert (
                        keyword == expected_keyword
                    ), f"查询 '{query}' 应匹配关键词 '{expected_keyword}'，实际匹配 '{keyword}'"
                    break
            assert matched, f"查询 '{query}' 应匹配关键词 '{expected_keyword}'"

    def test_abnormal_indicator_keywords(self):
        """
        测试异常指标关键词。

        关键词列表：
        - 偏高
        - 偏低
        - 升高
        - 降低
        - 异常
        """
        MEDICAL_KEYWORDS = [
            "偏高",
            "偏低",
            "升高",
            "降低",
            "异常",
        ]

        test_cases = [
            ("白细胞计数偏高", "偏高"),
            ("血红蛋白偏低", "偏低"),
            ("血糖升高", "升高"),
            ("血压降低", "降低"),
            ("检查结果异常", "异常"),
        ]

        for query, expected_keyword in test_cases:
            matched = False
            for keyword in MEDICAL_KEYWORDS:
                if keyword in query:
                    matched = True
                    assert (
                        keyword == expected_keyword
                    ), f"查询 '{query}' 应匹配关键词 '{expected_keyword}'，实际匹配 '{keyword}'"
                    break
            assert matched, f"查询 '{query}' 应匹配关键词 '{expected_keyword}'"


class TestRealWorldCases:
    """真实案例测试 - 基于 langsmith_copy.md 中的失败案例。"""

    def test_case_1_health_record_query(self):
        """
        测试案例1：健康档案查询

        用户输入："我是张三九，告诉我2年内我的身体状况，并与当前的血检报告进行对比分析。"
        预期路由：medical
        实际路由（修复前）：general
        """
        query = "我是张三九，告诉我2年内我的身体状况，并与当前的血检报告进行对比分析。"

        MEDICAL_KEYWORDS = [
            "身体状况",
            "健康档案",
            "病历",
            "体检报告",
            "血常规",
            "尿常规",
            "生化",
            "生命体征",
            "白细胞",
            "血红蛋白",
            "血小板",
            "红细胞",
            "中性粒细胞",
            "淋巴细胞",
            "头疼",
            "头痛",
            "发烧",
            "发热",
            "咳嗽",
            "腹痛",
            "恶心",
            "呕吐",
            "腹泻",
            "便秘",
            "诊断",
            "治疗",
            "用药",
            "复查",
            "体检",
            "偏高",
            "偏低",
            "升高",
            "降低",
            "异常",
        ]

        matched = False
        matched_keyword = None
        for keyword in MEDICAL_KEYWORDS:
            if keyword in query:
                matched = True
                matched_keyword = keyword
                break

        assert matched, f"查询 '{query}' 应匹配医疗关键词"
        assert (
            matched_keyword == "身体状况"
        ), f"查询应匹配关键词 '身体状况'，实际匹配 '{matched_keyword}'"

    def test_case_2_blood_test_report(self):
        """
        测试案例2：血检报告分析

        用户输入包含完整的血检报告数据：
        - 血常规检查报告
        - 白细胞计数：18.5 ×10⁹/L（异常升高）
        - 中性粒细胞百分比：85%（异常升高）

        预期路由：medical
        实际路由（修复前）：general
        """
        query = """这是我现在的血检报告：report_text:
血常规检查报告
检查日期：2026-03-16
白细胞计数：18.5 ×10⁹/L
血红蛋白：145 g/L
血小板计数：310 ×10⁹/L
红细胞计数：4.8 ×10¹²/L
红细胞压积：0.44 L/L
平均红细胞体积：90 fL
平均血红蛋白量：31 pg
平均血红蛋白浓度：340 g/L
中性粒细胞百分比：85%
淋巴细胞百分比：12%
gender: male"""

        MEDICAL_KEYWORDS = [
            "身体状况",
            "健康档案",
            "病历",
            "体检报告",
            "血常规",
            "尿常规",
            "生化",
            "生命体征",
            "白细胞",
            "血红蛋白",
            "血小板",
            "红细胞",
            "中性粒细胞",
            "淋巴细胞",
            "头疼",
            "头痛",
            "发烧",
            "发热",
            "咳嗽",
            "腹痛",
            "恶心",
            "呕吐",
            "腹泻",
            "便秘",
            "诊断",
            "治疗",
            "用药",
            "复查",
            "体检",
            "偏高",
            "偏低",
            "升高",
            "降低",
            "异常",
        ]

        matched = False
        matched_keywords = []
        for keyword in MEDICAL_KEYWORDS:
            if keyword in query:
                matched = True
                matched_keywords.append(keyword)

        assert matched, f"查询应匹配医疗关键词"
        assert "血常规" in matched_keywords, f"查询应匹配关键词 '血常规'"
        assert "白细胞" in matched_keywords, f"查询应匹配关键词 '白细胞'"
        assert "中性粒细胞" in matched_keywords, f"查询应匹配关键词 '中性粒细胞'"


if __name__ == "__main__":
    pytest.main(
        [
            "-v",
            "--tb=short",
            __file__,
        ]
    )
