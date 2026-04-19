# gradio_ui.py
"""
Gradio WebUI - 智能分诊系统前端界面（优化版）

功能特性：
- 用户登录/注册系统（密码 sha256+salt 哈希）
- Session Token 会话鉴权
- 多会话管理（创建、切换、删除）
- 流式/非流式响应模式切换
- 医疗扩展卡片嵌入聊天气泡（去重版）：
    卡片仅展示 紧急度/置信度/科室/风险警告，
    不重复 Markdown 中已有的 摘要/指标/建议/免责声明
- 建议提示卡片（原生 Gradio Button）
- 响应式布局设计
- 左侧面板可折叠功能（visibility 切换）
- SQLite 持久化存储
- 请求频率限制
- HTML 注入防护（html.escape）

与 main.py API 接口完全对齐
"""

import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
for proxy_var in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]:
    os.environ[proxy_var] = ""
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

import gradio as gr
import requests
import json
import uuid
import re
import html
import hashlib
import secrets
import sqlite3
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8012"
CHAT_ENDPOINT = f"{API_BASE_URL}/v1/chat/completions"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# ── 常量 ─────────────────────────────────────────────────────────────────────
NO_HISTORY_LABEL = "无历史会话"
DEFAULT_CONV_TITLE = "新的对话"
DB_PATH = "triage_users.db"

# ── 频率限制配置 ──────────────────────────────────────────────────────────────
RATE_LIMIT_SECONDS = 1.5
_rate_limit_lock = threading.Lock()
_last_request_time: Dict[str, float] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite 持久化层
# ═══════════════════════════════════════════════════════════════════════════════
_db_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _db_lock, _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                username        TEXT PRIMARY KEY,
                user_id         TEXT UNIQUE NOT NULL,
                password_hash   TEXT NOT NULL,
                salt            TEXT NOT NULL,
                sidebar_collapsed INTEGER DEFAULT 0,
                created_at      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                conv_id     TEXT PRIMARY KEY,
                username    TEXT NOT NULL,
                title       TEXT NOT NULL DEFAULT '新的对话',
                created_at  TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id     TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (conv_id) REFERENCES conversations(conv_id) ON DELETE CASCADE
            );
        """)
    logger.info("数据库初始化完成")


def db_save_user(username: str, user_id: str, password_hash: str, salt: str) -> None:
    with _db_lock, _get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, user_id, password_hash, salt, created_at) VALUES (?,?,?,?,?)",
            (username, user_id, password_hash, salt, get_current_time()),
        )


def db_get_user(username: str) -> Optional[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()


def db_update_sidebar(username: str, collapsed: bool) -> None:
    with _db_lock, _get_conn() as conn:
        conn.execute(
            "UPDATE users SET sidebar_collapsed=? WHERE username=?",
            (int(collapsed), username),
        )


def db_save_conversation(conv_id: str, username: str, title: str) -> None:
    with _db_lock, _get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO conversations (conv_id, username, title, created_at) VALUES (?,?,?,?)",
            (conv_id, username, title, get_current_time()),
        )


def db_update_conv_title(conv_id: str, title: str) -> None:
    with _db_lock, _get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET title=? WHERE conv_id=?", (title, conv_id)
        )


def db_delete_conversation(conv_id: str) -> None:
    with _db_lock, _get_conn() as conn:
        conn.execute("DELETE FROM messages      WHERE conv_id=?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE conv_id=?", (conv_id,))


def db_get_conversations(username: str) -> List[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM conversations WHERE username=? ORDER BY created_at DESC",
            (username,),
        ).fetchall()


def db_save_messages(conv_id: str, messages: List[Dict]) -> None:
    with _db_lock, _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
        conn.executemany(
            "INSERT INTO messages (conv_id, role, content, created_at) VALUES (?,?,?,?)",
            [(conv_id, m["role"], m["content"], get_current_time()) for m in messages],
        )


def db_get_messages(conv_id: str) -> List[Dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conv_id=? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# 密码哈希工具
# ═══════════════════════════════════════════════════════════════════════════════


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return pw_hash, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    pw_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(pw_hash, stored_hash)


# ═══════════════════════════════════════════════════════════════════════════════
# Session Token 管理
# ═══════════════════════════════════════════════════════════════════════════════
_session_tokens: Dict[str, str] = {}
_user_api_keys: Dict[str, str] = {}  # username -> API Key 映射
_session_lock = threading.Lock()


def create_session_token(username: str) -> str:
    """
    创建 Session Token（同时生成用户 API Key）

    Args:
        username: 用户名

    Returns:
        str: Session Token
    """
    token = secrets.token_urlsafe(32)
    # 为用户生成固定的 API Key（用于文档上传等需要认证的请求）
    api_key = f"sk-{secrets.token_urlsafe(24)}"

    with _session_lock:
        old = [t for t, u in _session_tokens.items() if u == username]
        for t in old:
            del _session_tokens[t]
        _session_tokens[token] = username
        _user_api_keys[username] = api_key

    logger.info(f"用户 {username} 登录成功，生成 API Key: {api_key[:10]}...")
    return token


def get_user_api_key(username: str) -> Optional[str]:
    """
    获取用户的 API Key

    Args:
        username: 用户名

    Returns:
        str: API Key，不存在返回 None
    """
    with _session_lock:
        return _user_api_keys.get(username)


def validate_session_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    with _session_lock:
        return _session_tokens.get(token)


def revoke_session_token(token: Optional[str]) -> None:
    if not token:
        return
    with _session_lock:
        _session_tokens.pop(token, None)


# 频率限制
def check_rate_limit(token: str) -> Tuple[bool, str]:
    now = time.time()
    with _rate_limit_lock:
        last = _last_request_time.get(token, 0)
        if now - last < RATE_LIMIT_SECONDS:
            wait = round(RATE_LIMIT_SECONDS - (now - last), 1)
            return False, f"⏳ 请求过于频繁，请等待 {wait} 秒"
        _last_request_time[token] = now
    return True, ""


#   分诊数据模型
@dataclass
class TriageData:
    recommended_departments: List[str] = field(default_factory=list)
    urgency_level: str = "routine"
    triage_reason: str = ""
    triage_confidence: float = 0.8

    @classmethod
    def from_dict(cls, data: Dict) -> "TriageData":
        if not data:
            return cls()
        return cls(
            recommended_departments=data.get("recommended_departments", []),
            urgency_level=data.get("urgency_level", "routine"),
            triage_reason=data.get("triage_reason", ""),
            triage_confidence=data.get("triage_confidence", 0.8),
        )


@dataclass
class StructuredMedicalData:
    triage: TriageData = field(default_factory=TriageData)
    analysis: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "StructuredMedicalData":
        if not data:
            return cls()
        return cls(
            triage=TriageData.from_dict(data.get("triage", {})),
            analysis=data.get("analysis"),
        )


@dataclass
class MedicalExtension:
    risk_level: str = "low"
    risk_warning: str = ""
    disclaimer: str = ""
    structured_data: Optional[StructuredMedicalData] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "MedicalExtension":
        if not data:
            return cls()
        return cls(
            risk_level=data.get("risk_level", "low"),
            risk_warning=data.get("risk_warning", ""),
            disclaimer=data.get("disclaimer", ""),
            structured_data=StructuredMedicalData.from_dict(
                data.get("structured_data", {})
            ),
        )


# 工具函数


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_user_id() -> str:
    return str(uuid.uuid4())


def generate_conversation_id(username: str) -> str:
    return f"{username}_{uuid.uuid4().hex[:8]}"


def check_backend_health() -> Tuple[bool, str]:
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, f"✅ 后端服务正常: {data.get('service', 'Unknown')}"
        return False, f"❌ 后端服务异常: HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"❌ 无法连接后端服务: {str(e)}"


# 用户管理
def register_user(username: str, password: str) -> str:
    if not username or not password:
        return "❌ 用户名和密码不能为空"
    if db_get_user(username):
        return "❌ 用户名已存在"
    pw_hash, salt = hash_password(password)
    db_save_user(username, generate_user_id(), pw_hash, salt)
    logger.info(f"用户注册成功: {username}")
    return "✅ 注册成功！请登录"


def login_user(
    username: str, password: str
) -> Tuple[bool, str, Optional[str], Optional[str], Optional[str]]:
    if not username or not password:
        return False, "❌ 用户名和密码不能为空", None, None, None
    row = db_get_user(username)
    if not row:
        return False, "❌ 用户名不存在", None, None, None
    if not verify_password(password, row["password_hash"], row["salt"]):
        return False, "❌ 密码错误", None, None, None

    token = create_session_token(username)
    conv_id = generate_conversation_id(username)
    db_save_conversation(conv_id, username, DEFAULT_CONV_TITLE)
    logger.info(f"用户登录成功: {username}")
    return True, "✅ 登录成功", row["user_id"], conv_id, token


def create_new_conversation(token: str) -> Tuple[Optional[str], str]:
    username = validate_session_token(token)
    if not username:
        return None, "❌ 会话已过期，请重新登录"
    conv_id = generate_conversation_id(username)
    db_save_conversation(conv_id, username, DEFAULT_CONV_TITLE)
    return conv_id, "✅ 新会话已创建"


def get_conversation_list(token: str) -> List[str]:
    username = validate_session_token(token)
    if not username:
        return [NO_HISTORY_LABEL]
    rows = db_get_conversations(username)
    return (
        [f"{r['title']} ({r['created_at']})" for r in rows]
        if rows
        else [NO_HISTORY_LABEL]
    )


def get_conversation_id_by_display(token: str, display: str) -> Optional[str]:
    username = validate_session_token(token)
    if not username or display == NO_HISTORY_LABEL:
        return None
    for row in db_get_conversations(username):
        if f"{row['title']} ({row['created_at']})" == display:
            return row["conv_id"]
    return None


def delete_conversation(token: str, conv_id: str) -> str:
    username = validate_session_token(token)
    if not username:
        return "❌ 会话已过期，请重新登录"
    rows = db_get_conversations(username)
    if not any(r["conv_id"] == conv_id for r in rows):
        return "❌ 会话不存在或无权限"
    db_delete_conversation(conv_id)
    return "✅ 会话已删除"


# HTML 格式化 — 风险徽章（公共组件）
def format_risk_badge(risk_level: str) -> str:
    configs = {
        "low": ("🟢", "低风险", "#10b981"),
        "medium": ("🟡", "中风险", "#f59e0b"),
        "high": ("🟠", "高风险", "#f97316"),
        "critical": ("🔴", "危急", "#ef4444"),
    }
    emoji, label, color = configs.get(risk_level, ("⚪", "未知", "#6b7280"))
    return (
        f'<span style="background-color:{color}20;color:{color};padding:4px 12px;'
        f'border-radius:16px;font-weight:600;font-size:12px;border:1px solid {color}40;">'
        f"{emoji} {label}</span>"
    )


def format_medical_card_for_chat(medical: MedicalExtension) -> str:
    """
    生成嵌入聊天气泡的医疗扩展卡片 HTML。

    去重策略
    --------
    Markdown 消息体已包含：摘要、异常指标、推荐科室文本、风险等级文字、
    健康建议、免责声明。

    本卡片 **仅** 补充以下 Markdown 中缺失的视觉化信息：
      1. 风险警告横幅（risk_warning，高危醒目提示）
      2. 紧急度彩色标签 + 置信度百分比
      3. 科室快捷标签（视觉 chip，与 Markdown 文本列表互补）
    """
    if not medical:
        return ""

    # 紧急度配置
    urgency_configs = {
        "routine": ("🟢", "常规就诊", "#10b981", "非紧急，可预约就诊"),
        "urgent": ("🟡", "尽快就医", "#f59e0b", "建议 24 小时内就医"),
        "emergency": ("🔴", "立即急诊", "#ef4444", "请立即前往急诊或拨打 120"),
    }

    parts: List[str] = []

    # ── 分隔线（与上方 Markdown 区分） ───────────────────────────────────────
    parts.append(
        '<div style="margin-top:14px;border-top:1px solid #e2e8f0;padding-top:14px;">'
    )

    # ── ① 风险警告横幅（仅当有高危信息且不是默认占位文本时显示）─────────────
    if medical.risk_warning and medical.risk_warning not in ("", "无高危风险"):
        risk_bg = {
            "critical": ("#fef2f2", "#fecaca", "#dc2626"),
            "high": ("#fff7ed", "#fed7aa", "#ea580c"),
            "medium": ("#fffbeb", "#fde68a", "#d97706"),
            "low": ("#eff6ff", "#bfdbfe", "#2563eb"),
        }
        bg, border, text_color = risk_bg.get(
            medical.risk_level, ("#f8fafc", "#e2e8f0", "#374151")
        )
        safe_warning = html.escape(medical.risk_warning)
        parts.append(
            f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
            f'padding:12px 14px;margin-bottom:12px;display:flex;align-items:flex-start;gap:10px;">'
            f'<span style="font-size:18px;flex-shrink:0;">🛡️</span>'
            f"<div>"
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            f'<span style="font-weight:600;color:#374151;font-size:13px;">风险评估</span>'
            f"{format_risk_badge(medical.risk_level)}"
            f"</div>"
            f'<p style="color:{text_color};font-size:13px;line-height:1.6;margin:0;">{safe_warning}</p>'
            f"</div>"
            f"</div>"
        )

    # ── ② 紧急度标签 + 置信度（Markdown 中无此可视化信息）──────────────────
    if medical.structured_data and medical.structured_data.triage:
        triage = medical.structured_data.triage
        emoji, label, color, desc = urgency_configs.get(
            triage.urgency_level, ("⚪", "未知", "#6b7280", "")
        )
        confidence_pct = int(triage.triage_confidence * 100)

        parts.append(
            f'<div style="background:{color}08;border:1.5px solid {color}40;'
            f'border-radius:10px;padding:12px 14px;margin-bottom:12px;">'
            # 紧急度行
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="font-size:20px;">{emoji}</span>'
            f'<span style="font-weight:700;color:{color};font-size:15px;">{label}</span>'
            f"</div>"
            f'<span style="font-size:12px;color:#6b7280;'
            f'background:#f1f5f9;padding:3px 10px;border-radius:20px;">'
            f"置信度 {confidence_pct}%</span>"
            f"</div>"
            # 说明文字
            f'<p style="color:#6b7280;font-size:12px;margin:6px 0 10px 28px;">{desc}</p>'
        )

        # ── ③ 科室快捷 chip（视觉补充，Markdown 中为纯文本列表）────────────
        if triage.recommended_departments:
            chips = "".join(
                [
                    f'<span style="background:{color}15;color:{color};border:1px solid {color}30;'
                    f'padding:4px 12px;border-radius:20px;font-size:12px;font-weight:500;">'
                    f"🏥 {html.escape(d)}</span>"
                    for d in triage.recommended_departments
                ]
            )
            parts.append(
                f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-left:28px;">{chips}</div>'
            )

        parts.append("</div>")  # 关闭紧急度块

    parts.append("</div>")  # 关闭分隔线容器
    return "".join(parts)


# Markdown 消息格式化（不变，保留原字段）
def format_message_content(content: str) -> str:
    """格式化消息内容：JSON 结构体转 Markdown，普通文本直接返回。"""
    if not content:
        return content

    # 自然语言回复末尾附带 JSON 格式的数据检测并移除末尾的 JSON 部分，只保留自然语言文本
    json_pattern = r"\s*\{[\s\S]*\}\s*$"
    content_without_json = re.sub(json_pattern, "", content).strip()

    # 如果移除 JSON 后内容为空，尝试原始内容
    if not content_without_json:
        content_without_json = content.strip()

    try:
        data = json.loads(content_without_json)
        if isinstance(data, dict):
            return format_json_to_markdown(data)
    except (json.JSONDecodeError, ValueError):
        pass
    return content_without_json


def format_json_to_markdown(data: dict) -> str:
    """将结构化 JSON 回复渲染为 Markdown（供 Chatbot 组件使用）。"""
    md: List[str] = []

    if data.get("summary"):
        md.append("### 📝 分析摘要\n")
        md.append(str(data["summary"]))
        md.append("\n\n")

    if data.get("abnormal_indicators"):
        md.append("### 🔍 异常指标\n")
        for ind in data["abnormal_indicators"]:
            md.append(f"- ⚠️ {ind}\n")
        md.append("\n")

    if data.get("departments"):
        md.append("### 🏥 推荐科室\n")
        for dept in data["departments"]:
            md.append(f"- 🏥 {dept}\n")
        md.append("\n")

    if data.get("risk_level"):
        emojis = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        md.append("### ⚡ 风险等级\n")
        md.append(
            f"{emojis.get(data['risk_level'], '⚪')} **{data['risk_level'].upper()}**\n\n"
        )

    if data.get("recommendations"):
        md.append("### 💡 健康建议\n")
        for rec in data["recommendations"]:
            md.append(f"- {rec}\n")
        md.append("\n")

    md.append("---\n⚠️ *以上分析仅供参考，请以医生面诊为准。*\n")
    return "".join(md)


# ═══════════════════════════════════════════════════════════════════════════════
# API 请求（复用 Session）
# ═══════════════════════════════════════════════════════════════════════════════
_http_session = requests.Session()


def send_chat_request(
    messages: List[Dict],
    user_id: str,
    conversation_id: str,
    stream: bool = True,
    username: Optional[str] = None,
):
    """
    发送聊天请求到后端 API。

    Args:
        messages: 消息列表
        user_id: 用户 ID
        conversation_id: 会话 ID
        stream: 是否流式响应
        username: 用户名（用于获取 API Key）

    Returns:
        Response: HTTP 响应对象（流式）或 JSON（非流式）
    """
    payload = {
        "messages": messages,
        "stream": stream,
        "userId": user_id,
        "conversationId": conversation_id,
    }

    headers = {"Content-Type": "application/json"}

    if username:
        api_key = get_user_api_key(username)
        if api_key:
            headers["X-API-Key"] = api_key

    try:
        if stream:
            return _http_session.post(
                CHAT_ENDPOINT,
                headers=headers,
                data=json.dumps(payload),
                stream=True,
                timeout=120,
            )
        else:
            return _http_session.post(
                CHAT_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=60
            ).json()
    except requests.exceptions.ConnectionError as e:
        logger.error(f"连接失败: {e}")
        raise
    except requests.exceptions.Timeout as e:
        logger.error(f"请求超时: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"请求失败: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# 建议卡片数据
# ═══════════════════════════════════════════════════════════════════════════════
SUGGESTIONS = [
    ("🤒", "我头痛、发热两天了，体温38.5度"),
    ("🩸", "血常规检查白细胞12.5，中性粒细胞85%"),
    ("💓", "最近胸闷气短，活动后加重，伴有心悸"),
    ("😵", "最近经常头晕，早上起来特别明显"),
    ("🤧", "咳嗽一周了，有黄痰，夜间加重"),
    ("❓", "什么情况需要去急诊？"),
]


def create_welcome_header_html() -> str:
    return """
    <div style="text-align:center;padding:20px 20px 8px 20px;">
        <div style="margin-bottom:24px;">
            <div style="width:64px;height:64px;
                        background:linear-gradient(135deg,#3b82f6 0%,#06b6d4 100%);
                        border-radius:16px;display:inline-flex;align-items:center;
                        justify-content:center;margin-bottom:16px;
                        box-shadow:0 8px 24px rgba(59,130,246,0.3);">
                <span style="font-size:28px;">🩺</span>
            </div>
            <h2 style="font-size:24px;font-weight:bold;color:#1e293b;margin-bottom:8px;">
                智能分诊助手
            </h2>
            <p style="color:#64748b;font-size:14px;">AI-Powered Medical Triage System</p>
        </div>

        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;
                    max-width:500px;margin:0 auto 24px auto;">
            <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                        padding:16px;text-align:center;">
                <span style="font-size:24px;display:block;margin-bottom:8px;">🔍</span>
                <span style="font-weight:600;color:#374151;font-size:13px;display:block;">症状分析</span>
                <span style="color:#9ca3af;font-size:11px;">描述症状，智能分析</span>
            </div>
            <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                        padding:16px;text-align:center;">
                <span style="font-size:24px;display:block;margin-bottom:8px;">🏥</span>
                <span style="font-weight:600;color:#374151;font-size:13px;display:block;">科室推荐</span>
                <span style="color:#9ca3af;font-size:11px;">精准推荐就诊科室</span>
            </div>
            <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                        padding:16px;text-align:center;">
                <span style="font-size:24px;display:block;margin-bottom:8px;">📋</span>
                <span style="font-weight:600;color:#374151;font-size:13px;display:block;">报告解读</span>
                <span style="color:#9ca3af;font-size:11px;">解读检验报告结果</span>
            </div>
        </div>

        <p style="color:#64748b;font-size:13px;margin-bottom:12px;">
            试试这些问题，体验智能分诊
        </p>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
.gradio-container { max-width: 100% !important; }
.primary-btn   { background: linear-gradient(135deg,#3b82f6 0%,#06b6d4 100%) !important; border: none !important; }
.secondary-btn { background: transparent !important; border: 1px solid #3b82f6 !important; color: #3b82f6 !important; }
#msg_input textarea { font-size: 15px !important; min-height: 60px !important; }

/* 建议卡片按钮 */
.suggestion-btn {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    text-align: left !important;
    font-size: 14px !important;
    color: #4b5563 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    transition: all 0.2s !important;
    height: auto !important;
}
.suggestion-btn:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.2) !important;
    transform: translateY(-1px) !important;
}

/* 侧边栏折叠 */
.sidebar-toggle-btn { min-width: 36px !important; padding: 6px 10px !important; }

/* 聊天框高度自适应 */
.chat-container { height: calc(100vh - 260px) !important; min-height: 400px !important; }

/* 发送中禁用态 */
#send_btn[disabled] { opacity: 0.6 !important; cursor: not-allowed !important; }

/* 欢迎面板淡入 */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.welcome-fade { animation: fadeIn 0.3s ease-in-out; }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 文档管理 API 调用
# ═══════════════════════════════════════════════════════════════════════════════

DOCUMENTS_UPLOAD_ENDPOINT = f"{API_BASE_URL}/v1/documents/upload"
DOCUMENTS_LIST_ENDPOINT = f"{API_BASE_URL}/v1/documents"
DOCUMENTS_STATS_ENDPOINT = f"{API_BASE_URL}/v1/documents/stats"


def upload_document_to_api(
    token: str, file_path: str, doc_type: str, username: Optional[str] = None
) -> Tuple[bool, str]:
    """
    上传文档到后端 API。

    Args:
        token: 用户会话 token
        file_path: 文件路径
        doc_type: 文档类型
        username: 用户名（用于获取 API Key）

    Returns:
        Tuple[bool, str]: (成功标志, 消息)
    """
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {"doc_type": doc_type}

            headers = {}
            if username:
                api_key = get_user_api_key(username)
                if api_key:
                    headers["X-API-Key"] = api_key

            response = requests.post(
                DOCUMENTS_UPLOAD_ENDPOINT,
                files=files,
                data=data,
                headers=headers,
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    return (
                        True,
                        f"✅ 上传成功: {result.get('filename')} ({result.get('chunks_count')} 个文本块)",
                    )
                else:
                    return False, f"❌ 上传失败: {result.get('error', '未知错误')}"
            elif response.status_code == 401:
                return False, "❌ 认证失败，请重新登录"
            else:
                return False, f"❌ 上传失败: HTTP {response.status_code}"

    except Exception as e:
        return False, f"❌ 上传异常: {str(e)}"


def get_documents_from_api(
    token: str, limit: int = 10, offset: int = 0, username: Optional[str] = None
) -> Tuple[bool, List[Dict], str]:
    """
    从后端 API 获取文档列表。

    Args:
        token: 用户会话 token
        limit: 返回数量限制
        offset: 偏移量
        username: 用户名（用于获取 API Key）

    Returns:
        Tuple[bool, List[Dict], str]: (成功标志, 文档列表, 消息)
    """
    try:
        headers = {}
        if username:
            api_key = get_user_api_key(username)
            if api_key:
                headers["X-API-Key"] = api_key

        params = {"limit": limit, "offset": offset}

        response = requests.get(
            DOCUMENTS_LIST_ENDPOINT, headers=headers, params=params, timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            return (
                True,
                result.get("documents", []),
                f"共 {result.get('total', 0)} 个文档",
            )
        elif response.status_code == 401:
            return False, [], "认证失败，请重新登录"
        else:
            return False, [], f"获取失败: HTTP {response.status_code}"

    except Exception as e:
        return False, [], f"获取异常: {str(e)}"


def delete_document_from_api(
    token: str, file_md5: str, username: Optional[str] = None
) -> Tuple[bool, str]:
    """
    从后端 API 删除文档。

    Args:
        token: 用户会话 token
        file_md5: 文件 MD5 指纹
        username: 用户名（用于获取 API Key）

    Returns:
        Tuple[bool, str]: (成功标志, 消息)
    """
    try:
        headers = {}
        if username:
            api_key = get_user_api_key(username)
            if api_key:
                headers["X-API-Key"] = api_key

        response = requests.delete(
            f"{DOCUMENTS_LIST_ENDPOINT}/{file_md5}", headers=headers, timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return (
                    True,
                    f"✅ 删除成功: {result.get('deleted_chunks')} 个文本块已删除",
                )
            else:
                return False, f"❌ 删除失败: {result.get('error', '未知错误')}"
        elif response.status_code == 401:
            return False, "❌ 认证失败，请重新登录"
        else:
            return False, f"❌ 删除失败: HTTP {response.status_code}"

    except Exception as e:
        return False, f"❌ 删除异常: {str(e)}"


def get_document_stats_from_api(
    token: str, username: Optional[str] = None
) -> Tuple[bool, Dict, str]:
    """
    从后端 API 获取文档统计信息。

    Args:
        token: 用户会话 token
        username: 用户名（用于获取 API Key）

    Returns:
        Tuple[bool, Dict, str]: (成功标志, 统计信息, 消息)
    """
    try:
        headers = {}
        if username:
            api_key = get_user_api_key(username)
            if api_key:
                headers["X-API-Key"] = api_key

        response = requests.get(DOCUMENTS_STATS_ENDPOINT, headers=headers, timeout=10)

        if response.status_code == 200:
            result = response.json()
            return True, result, "统计信息获取成功"
        elif response.status_code == 401:
            return False, {}, "认证失败，请重新登录"
        else:
            return False, {}, f"获取失败: HTTP {response.status_code}"

    except Exception as e:
        return False, {}, f"获取异常: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Gradio 应用
# ═══════════════════════════════════════════════════════════════════════════════


def create_gradio_app():
    with gr.Blocks(
        title="智能分诊助手",
    ) as app:

        # ── 全局状态 ──────────────────────────────────────────────────────────
        logged_in = gr.State(False)
        current_token = gr.State(None)
        current_user = gr.State(None)
        current_user_id = gr.State(None)
        current_conv_id = gr.State(None)
        chat_history = gr.State([])
        stream_mode = gr.State(True)
        current_medical = gr.State(None)
        sidebar_collapsed = gr.State(False)

        # ══════════════════════════════════════════════════════════════════════
        # 登录页
        # ══════════════════════════════════════════════════════════════════════
        with gr.Column(visible=True) as login_page:
            gr.Markdown("""
            <div style="text-align:center;margin-bottom:24px;">
                <div style="width:56px;height:56px;
                            background:linear-gradient(135deg,#3b82f6 0%,#06b6d4 100%);
                            border-radius:14px;display:inline-flex;align-items:center;
                            justify-content:center;margin-bottom:16px;
                            box-shadow:0 6px 20px rgba(59,130,246,0.3);">
                    <span style="font-size:24px;">🩺</span>
                </div>
                <h1 style="font-size:24px;font-weight:bold;color:#1e293b;margin:0;">智能分诊助手</h1>
                <p style="color:#64748b;font-size:14px;margin-top:4px;">AI-Powered Medical Triage System</p>
            </div>
            """)

            login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
            login_password = gr.Textbox(
                label="密码", placeholder="请输入密码", type="password"
            )

            with gr.Row():
                login_button = gr.Button("登录", variant="primary", scale=1)
                register_button = gr.Button("注册", variant="secondary", scale=1)

            login_output = gr.Textbox(label="提示", interactive=False)
            gr.Markdown(
                '<p style="text-align:center;color:#64748b;font-size:13px;margin-top:16px;">还没有账号？</p>'
            )
            switch_to_register = gr.Button("立即注册", variant="secondary", size="sm")

            health_status = gr.HTML("")
            check_health_btn = gr.Button("检查后端服务", size="sm", variant="secondary")

        # ══════════════════════════════════════════════════════════════════════
        # 注册页
        # ══════════════════════════════════════════════════════════════════════
        with gr.Column(visible=False) as register_page:
            gr.Markdown("### 注册新账号")

            reg_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
            reg_password = gr.Textbox(
                label="密码", placeholder="请输入密码", type="password"
            )

            with gr.Row():
                reg_button = gr.Button("提交注册", variant="primary", scale=1)
                switch_to_login = gr.Button("返回登录", variant="secondary", scale=1)

            reg_output = gr.Textbox(label="提示", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        # 聊天主页
        # ══════════════════════════════════════════════════════════════════════
        with gr.Column(visible=False) as chat_page:
            # 聊天页独立状态提示（不复用注册页组件）
            chat_status_msg = gr.Textbox(label="", interactive=False, visible=False)

            with gr.Row(equal_height=False):

                # ── 左侧面板 ─────────────────────────────────────────────────
                with gr.Column(scale=1, min_width=60) as sidebar_col:
                    toggle_sidebar_btn = gr.Button(
                        "◀",
                        variant="secondary",
                        size="sm",
                        elem_classes=["sidebar-toggle-btn"],
                    )

                    with gr.Column(visible=True) as sidebar_content:
                        gr.Markdown("### 📊 智能分诊")

                        with gr.Accordion("⚙️ 功能设置", open=False):
                            stream_toggle = gr.Checkbox(
                                label="流式输出", value=True, interactive=True
                            )
                            logout_btn = gr.Button(
                                "🚪 退出登录", variant="secondary", size="sm"
                            )

                        with gr.Accordion("📄 文档管理", open=False):
                            gr.Markdown("""
                            <div style="background:#eff6ff;border-left:3px solid #3b82f6;padding:8px 12px;border-radius:6px;margin-bottom:12px;">
                                <p style="margin:0;font-size:12px;color:#1e40af;">
                                    📌 上传您的医疗文档（病历、体检报告等），系统将自动分析并支持智能检索。
                                </p>
                            </div>
                            """)

                            # 文档上传
                            doc_upload_file = gr.File(
                                label="上传文档",
                                file_types=[".pdf", ".docx", ".txt"],
                                type="filepath",
                            )

                            doc_type_dropdown = gr.Dropdown(
                                label="文档类型",
                                choices=[
                                    ("体检报告", "health_report"),
                                    ("病历", "medical_record"),
                                    ("检验报告", "lab_report"),
                                    ("处方", "prescription"),
                                    ("其他", "other"),
                                ],
                                value="other",
                                interactive=True,
                            )

                            upload_doc_btn = gr.Button(
                                "📤 上传文档", variant="primary", size="sm"
                            )
                            upload_doc_status = gr.Textbox(
                                label="", interactive=False, visible=False
                            )

                            # 文档统计
                            gr.Markdown("---")
                            doc_stats_display = gr.HTML(
                                "<p style='color:#64748b;font-size:12px;'>点击刷新查看统计</p>"
                            )
                            refresh_stats_btn = gr.Button(
                                "🔄 刷新统计", variant="secondary", size="sm"
                            )

                            # 文档列表
                            gr.Markdown("---")
                            doc_list_display = gr.HTML(
                                "<p style='color:#64748b;font-size:12px;'>点击查看文档列表</p>"
                            )
                            refresh_docs_btn = gr.Button(
                                "📋 查看文档", variant="secondary", size="sm"
                            )

                            # 删除文档
                            gr.Markdown("---")
                            delete_doc_md5 = gr.Textbox(
                                label="删除文档（输入 MD5）",
                                placeholder="从文档列表中复制 MD5",
                                interactive=True,
                            )
                            delete_doc_btn = gr.Button(
                                "🗑️ 删除文档", variant="stop", size="sm"
                            )
                            delete_doc_status = gr.Textbox(
                                label="", interactive=False, visible=False
                            )

                        with gr.Accordion("📋 会话管理", open=False):
                            conv_dropdown = gr.Dropdown(
                                label="历史会话",
                                choices=[NO_HISTORY_LABEL],
                                value=NO_HISTORY_LABEL,
                                interactive=True,
                                allow_custom_value=True,
                            )
                            with gr.Row():
                                new_conv_btn = gr.Button(
                                    "新建", variant="primary", size="sm", scale=1
                                )
                                del_conv_btn = gr.Button(
                                    "删除", variant="secondary", size="sm", scale=1
                                )

                # ── 主内容区 ─────────────────────────────────────────────────
                with gr.Column(scale=5) as main_col:
                    with gr.Row():
                        welcome_text = gr.Markdown("### 欢迎，用户")
                        conv_title = gr.Markdown(f"**当前会话：** {DEFAULT_CONV_TITLE}")

                    # 欢迎面板
                    with gr.Column(visible=True) as welcome_panel:
                        gr.HTML(
                            create_welcome_header_html(), elem_classes=["welcome-fade"]
                        )
                        with gr.Row():
                            sug_btns: List[Tuple[gr.Button, str]] = []
                            for i in range(0, len(SUGGESTIONS), 2):
                                with gr.Column():
                                    for emoji, text in SUGGESTIONS[i : i + 2]:
                                        btn = gr.Button(
                                            f"{emoji}  {text}",
                                            variant="secondary",
                                            elem_classes=["suggestion-btn"],
                                        )
                                        sug_btns.append((btn, text))

                    # ── 聊天框 ────────────────────────────────────────────────
                    # 医疗卡片已通过 format_medical_card_for_chat() 嵌入消息内容，
                    # 此处不再需要独立的 medical_panel 组件。
                    chatbot = gr.Chatbot(
                        label="对话",
                        height=500,
                        visible=False,
                        elem_classes=["chat-container"],
                    )

                    msg_input = gr.Textbox(
                        label="输入消息",
                        placeholder="描述您的症状，开始智能分诊...",
                        lines=2,
                        elem_id="msg_input",
                    )

                    with gr.Row():
                        send_btn = gr.Button(
                            "发送", variant="primary", scale=2, elem_id="send_btn"
                        )
                        clear_btn = gr.Button("清空对话", variant="secondary", scale=1)

        # ══════════════════════════════════════════════════════════════════════
        # 事件处理函数
        # ══════════════════════════════════════════════════════════════════════

        def check_health():
            ok, msg = check_backend_health()
            color = "#10b981" if ok else "#ef4444"
            return (
                f'<div style="padding:12px;background:{color}15;border:1px solid {color}40;'
                f'border-radius:8px;color:{color};font-size:14px;text-align:center;">{msg}</div>'
            )

        def show_login():
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        def show_register():
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        def show_chat():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            )

        # ── 登录 ──────────────────────────────────────────────────────────────
        def handle_login(username, password):
            success, message, user_id, conv_id, token = login_user(username, password)

            if success:
                row = db_get_user(username)
                collapsed = bool(row["sidebar_collapsed"]) if row else False
                conv_list = get_conversation_list(token)
                return [
                    True,
                    token,
                    username,
                    user_id,
                    conv_id,
                    "",
                    *show_chat(),
                    f"### 欢迎，{username}",
                    f"**当前会话：** {DEFAULT_CONV_TITLE}",
                    gr.update(visible=True),
                    gr.update(value=[], visible=False),
                    [],
                    None,
                    gr.update(choices=conv_list, value=NO_HISTORY_LABEL),
                    collapsed,
                    gr.update(visible=not collapsed),
                    "▶" if collapsed else "◀",
                ]

            return [
                False,
                None,
                None,
                None,
                None,
                message,
                *show_login(),
                "### 欢迎，用户",
                f"**当前会话：** {DEFAULT_CONV_TITLE}",
                gr.update(visible=True),
                gr.update(value=[], visible=False),
                [],
                None,
                gr.update(choices=[NO_HISTORY_LABEL], value=NO_HISTORY_LABEL),
                False,
                gr.update(visible=True),
                "◀",
            ]

        # ── 注册 ──────────────────────────────────────────────────────────────
        def handle_register(username, password):
            message = register_user(username, password)
            if "成功" in message:
                return [message, *show_login(), "", ""]
            return [
                message,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                username,
                password,
            ]

        # ── 登出 ──────────────────────────────────────────────────────────────
        def handle_logout(token):
            revoke_session_token(token)
            return [
                False,
                None,
                None,
                None,
                None,
                *show_login(),
                "",
                "",
                gr.update(visible=True),
                gr.update(value=[], visible=False),
                [],
                None,
                gr.update(choices=[NO_HISTORY_LABEL], value=NO_HISTORY_LABEL),
                False,
                gr.update(visible=True),
                "◀",
            ]

        # ── 新建会话 ──────────────────────────────────────────────────────────
        def handle_new_conversation(token):
            new_conv_id, _ = create_new_conversation(token)
            if not new_conv_id:
                return [
                    None,
                    f"**当前会话：** {DEFAULT_CONV_TITLE}",
                    [],
                    gr.update(visible=True),
                    gr.update(value=[], visible=False),
                    gr.update(choices=[NO_HISTORY_LABEL], value=NO_HISTORY_LABEL),
                    gr.update(value="", visible=False),
                ]
            conv_list = get_conversation_list(token)
            return [
                new_conv_id,
                f"**当前会话：** {DEFAULT_CONV_TITLE}",
                [],
                gr.update(visible=True),
                gr.update(value=[], visible=False),
                gr.update(choices=conv_list, value=NO_HISTORY_LABEL),
                gr.update(value="", visible=False),
            ]

        # ── 切换会话 ──────────────────────────────────────────────────────────
        def handle_select_conversation(token, selected, current_conv):
            if not token or selected == NO_HISTORY_LABEL:
                return [
                    current_conv,
                    f"**当前会话：** {DEFAULT_CONV_TITLE}",
                    [],
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "",
                ]

            conv_id = get_conversation_id_by_display(token, selected)
            if not conv_id:
                return [
                    current_conv,
                    f"**当前会话：** {DEFAULT_CONV_TITLE}",
                    [],
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "",
                ]

            messages = db_get_messages(conv_id)
            title = selected.rsplit(" (", 1)[0]

            if messages:
                return [
                    conv_id,
                    f"**当前会话：** {title}",
                    messages,
                    gr.update(value=messages, visible=True),
                    gr.update(visible=False),
                    "",
                ]
            else:
                return [
                    conv_id,
                    f"**当前会话：** {title}",
                    [],
                    gr.update(value=[], visible=False),
                    gr.update(visible=True),
                    "",
                ]

        # ── 删除会话 ──────────────────────────────────────────────────────────
        def handle_delete_conversation(token, conv_id):
            msg = delete_conversation(token, conv_id)
            new_conv_id, _ = create_new_conversation(token)
            conv_list = get_conversation_list(token)
            return [
                gr.update(value=msg, visible=True),
                gr.update(choices=conv_list, value=NO_HISTORY_LABEL),
                new_conv_id or conv_id,
                f"**当前会话：** {DEFAULT_CONV_TITLE}",
                [],
                gr.update(visible=True),
                gr.update(value=[], visible=False),
            ]

        # ── 侧边栏折叠 ────────────────────────────────────────────────────────
        def toggle_sidebar(collapsed, token):
            new_collapsed = not collapsed
            username = validate_session_token(token)
            if username:
                db_update_sidebar(username, new_collapsed)
            return (
                gr.update(visible=not new_collapsed),
                "▶" if new_collapsed else "◀",
                new_collapsed,
            )

        # ── 文档管理 ──────────────────────────────────────────────────────────
        def handle_upload_document(token, file_path, doc_type):
            """处理文档上传"""
            if not token:
                return gr.update(value="❌ 请先登录", visible=True)

            if not file_path:
                return gr.update(value="❌ 请选择文件", visible=True)

            # 获取用户名（从 token）
            username = validate_session_token(token)

            success, message = upload_document_to_api(
                token, file_path, doc_type, username
            )
            return gr.update(value=message, visible=True)

        def handle_refresh_stats(token):
            """刷新文档统计"""
            if not token:
                return "<p style='color:#ef4444;font-size:12px;'>❌ 请先登录</p>"

            username = validate_session_token(token)
            success, stats, message = get_document_stats_from_api(token, username)

            if success:
                doc_types_html = ""
                for doc_type, count in stats.get("doc_types", {}).items():
                    doc_type_names = {
                        "health_report": "体检报告",
                        "medical_record": "病历",
                        "lab_report": "检验报告",
                        "prescription": "处方",
                        "other": "其他",
                    }
                    doc_types_html += f"""
                    <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #e2e8f0;">
                        <span style="color:#475569;font-size:12px;">{doc_type_names.get(doc_type, doc_type)}</span>
                        <span style="color:#1e40af;font-weight:600;font-size:12px;">{count}</span>
                    </div>
                    """

                return f"""
                <div style="background:#f8fafc;border-radius:8px;padding:12px;margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                        <span style="color:#64748b;font-size:12px;">📄 文档总数</span>
                        <span style="color:#1e40af;font-weight:600;font-size:14px;">{stats.get('total_documents', 0)}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                        <span style="color:#64748b;font-size:12px;">📝 文本块总数</span>
                        <span style="color:#1e40af;font-weight:600;font-size:14px;">{stats.get('total_chunks', 0)}</span>
                    </div>
                    <div style="margin-top:12px;padding-top:8px;border-top:1px solid #e2e8f0;">
                        <p style="color:#64748b;font-size:12px;margin-bottom:4px;">分类统计：</p>
                        {doc_types_html}
                    </div>
                </div>
                """
            else:
                return f"<p style='color:#ef4444;font-size:12px;'>❌ {message}</p>"

        def handle_refresh_docs(token):
            """刷新文档列表"""
            if not token:
                return "<p style='color:#ef4444;font-size:12px;'>❌ 请先登录</p>"

            username = validate_session_token(token)
            success, documents, message = get_documents_from_api(
                token, limit=10, offset=0, username=username
            )

            if success and documents:
                docs_html = ""
                for doc in documents:
                    doc_type_names = {
                        "health_report": "🏥 体检报告",
                        "medical_record": "📋 病历",
                        "lab_report": "🔬 检验报告",
                        "prescription": "💊 处方",
                        "other": "📄 其他",
                    }

                    docs_html += f"""
                    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:10px;margin-bottom:8px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                            <span style="font-weight:600;color:#1e293b;font-size:13px;">{doc.get('filename', '未知')}</span>
                            <span style="background:#eff6ff;color:#1e40af;padding:2px 8px;border-radius:4px;font-size:11px;">
                                {doc_type_names.get(doc.get('doc_type', 'other'), '📄 其他')}
                            </span>
                        </div>
                        <div style="color:#64748b;font-size:11px;margin-bottom:4px;">
                            📅 {doc.get('upload_time', '未知时间')}
                        </div>
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="color:#94a3b8;font-size:10px;font-family:monospace;">
                                MD5: {doc.get('file_md5', 'N/A')[:16]}...
                            </span>
                            <span style="color:#64748b;font-size:11px;">
                                预览: {doc.get('content_preview', '')[:30]}...
                            </span>
                        </div>
                    </div>
                    """

                return f"""
                <div style="max-height:400px;overflow-y:auto;">
                    {docs_html}
                </div>
                """
            elif success and not documents:
                return "<p style='color:#64748b;font-size:12px;text-align:center;padding:20px;'>📭 暂无文档，请上传</p>"
            else:
                return f"<p style='color:#ef4444;font-size:12px;'>❌ {message}</p>"

        def handle_delete_document(token, file_md5):
            """处理文档删除"""
            if not token:
                return gr.update(value="❌ 请先登录", visible=True)

            if not file_md5 or not file_md5.strip():
                return gr.update(value="❌ 请输入文档 MD5", visible=True)

            username = validate_session_token(token)
            success, message = delete_document_from_api(
                token, file_md5.strip(), username
            )
            return gr.update(value=message, visible=True)

        # ── 发送消息（核心） ──────────────────────────────────────────────────
        def handle_send_message(message, history, user_id, conv_id, token, stream):
            """
            关键改动
            --------
            - 流式/非流式完成后，调用 format_medical_card_for_chat() 生成去重卡片
            - 将卡片 HTML 直接拼接在 Markdown 消息尾部，嵌入聊天气泡
            - 不再使用独立的 medical_panel 组件
            """
            # 基础校验
            if not message or not message.strip():
                yield [gr.update(), history, history, None, gr.update(interactive=True)]
                return

            username = validate_session_token(token)
            if not username or not user_id or not conv_id:
                yield [
                    gr.update(value=""),
                    history,
                    history,
                    None,
                    gr.update(interactive=True),
                ]
                return

            # 频率限制
            allowed, rate_msg = check_rate_limit(token)
            if not allowed:
                err_history = history + [
                    {"role": "user", "content": message.strip()},
                    {"role": "assistant", "content": rate_msg},
                ]
                yield [
                    gr.update(value=""),
                    err_history,
                    err_history,
                    None,
                    gr.update(interactive=True),
                ]
                return

            user_msg = {"role": "user", "content": message.strip()}
            new_history = history + [user_msg]

            # 禁用发送按钮，清空输入框
            yield [
                gr.update(value=""),
                new_history,
                new_history,
                None,
                gr.update(interactive=False),
            ]

            api_messages = [
                {"role": m["role"], "content": m["content"]} for m in new_history
            ]

            try:
                if stream:
                    response = send_chat_request(
                        api_messages, user_id, conv_id, stream=True, username=username
                    )
                    accumulated = ""
                    medical_ext = None

                    for line in response.iter_lines():
                        if not line:
                            continue
                        json_str = line.decode("utf-8").strip()
                        if json_str.startswith("data: "):
                            json_str = json_str[6:]
                        if not json_str or json_str == "[DONE]":
                            continue

                        try:
                            data = json.loads(json_str)
                            choice = data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})

                            # 文本增量（流式过程中仅显示纯文本，不附卡片）
                            if delta.get("content"):
                                accumulated += delta["content"]
                                formatted = format_message_content(accumulated)
                                tmp_history = new_history + [
                                    {"role": "assistant", "content": formatted}
                                ]
                                yield [
                                    gr.update(),
                                    tmp_history,
                                    tmp_history,
                                    None,
                                    gr.update(interactive=False),
                                ]

                            # 正确提取 medical：顶层 > choice 级 > delta 级
                            if not medical_ext:
                                med_data = (
                                    data.get("medical")
                                    or choice.get("medical")
                                    or delta.get("medical")
                                )
                                if med_data:
                                    medical_ext = MedicalExtension.from_dict(med_data)

                            if choice.get("finish_reason") == "stop":
                                # finish chunk 可能也携带 medical
                                if not medical_ext and data.get("medical"):
                                    medical_ext = MedicalExtension.from_dict(
                                        data["medical"]
                                    )
                                break

                        except json.JSONDecodeError:
                            continue

                    # ── 流式结束：拼接去重卡片到消息尾部 ─────────────────────
                    formatted_text = format_message_content(accumulated)
                    card_html = (
                        format_medical_card_for_chat(medical_ext) if medical_ext else ""
                    )
                    # Markdown + HTML 卡片合并为单条消息
                    final_content = formatted_text + card_html
                    final_history = new_history + [
                        {"role": "assistant", "content": final_content}
                    ]

                    db_save_messages(conv_id, final_history)
                    if message[:20] != DEFAULT_CONV_TITLE:
                        db_update_conv_title(conv_id, message[:20])

                    yield [
                        gr.update(),
                        final_history,
                        final_history,
                        medical_ext,
                        gr.update(interactive=True),
                    ]

                else:
                    # 非流式
                    response_json = send_chat_request(
                        api_messages, user_id, conv_id, stream=False, username=username
                    )
                    content = (
                        response_json.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "未获取到回复")
                    )
                    formatted_text = format_message_content(content)

                    med_data = response_json.get("medical")
                    medical_ext = (
                        MedicalExtension.from_dict(med_data) if med_data else None
                    )
                    card_html = (
                        format_medical_card_for_chat(medical_ext) if medical_ext else ""
                    )

                    # ── 非流式：同样拼接卡片 ───────────────────────────────
                    final_content = formatted_text + card_html
                    final_history = new_history + [
                        {"role": "assistant", "content": final_content}
                    ]

                    db_save_messages(conv_id, final_history)
                    if message[:20] != DEFAULT_CONV_TITLE:
                        db_update_conv_title(conv_id, message[:20])

                    yield [
                        gr.update(),
                        final_history,
                        final_history,
                        medical_ext,
                        gr.update(interactive=True),
                    ]

            except Exception as e:
                logger.error(f"发送消息失败: {e}")
                err_history = new_history + [
                    {"role": "assistant", "content": f"❌ 请求失败: {str(e)}"}
                ]
                yield [
                    gr.update(),
                    err_history,
                    err_history,
                    None,
                    gr.update(interactive=True),
                ]

        # ── 欢迎面板与聊天框切换 ─────────────────────────────────────────────
        def update_welcome_visibility(history):
            if history and len(history) > 0:
                return gr.update(visible=False), gr.update(visible=True)
            return gr.update(visible=True), gr.update(visible=False)

        def clear_chat():
            return [], [], None, gr.update(visible=True)

        # ══════════════════════════════════════════════════════════════════════
        # 事件绑定
        # ══════════════════════════════════════════════════════════════════════

        check_health_btn.click(check_health, None, health_status)

        register_button.click(
            show_register, None, [login_page, register_page, chat_page]
        )
        switch_to_register.click(
            show_register, None, [login_page, register_page, chat_page]
        )
        switch_to_login.click(show_login, None, [login_page, register_page, chat_page])

        login_button.click(
            handle_login,
            [login_username, login_password],
            [
                logged_in,
                current_token,
                current_user,
                current_user_id,
                current_conv_id,
                login_output,
                login_page,
                register_page,
                chat_page,
                welcome_text,
                conv_title,
                welcome_panel,
                chatbot,
                chat_history,
                current_medical,
                conv_dropdown,
                sidebar_collapsed,
                sidebar_content,
                toggle_sidebar_btn,
            ],
        )

        reg_button.click(
            handle_register,
            [reg_username, reg_password],
            [
                reg_output,
                login_page,
                register_page,
                chat_page,
                reg_username,
                reg_password,
            ],
        )

        logout_btn.click(
            handle_logout,
            [current_token],
            [
                logged_in,
                current_token,
                current_user,
                current_user_id,
                current_conv_id,
                login_page,
                register_page,
                chat_page,
                login_username,
                login_password,
                welcome_panel,
                chatbot,
                chat_history,
                current_medical,
                conv_dropdown,
                sidebar_collapsed,
                sidebar_content,
                toggle_sidebar_btn,
            ],
        )

        new_conv_btn.click(
            handle_new_conversation,
            [current_token],
            [
                current_conv_id,
                conv_title,
                chat_history,
                welcome_panel,
                chatbot,
                conv_dropdown,
                chat_status_msg,
            ],
        )

        conv_dropdown.change(
            handle_select_conversation,
            [current_token, conv_dropdown, current_conv_id],
            [
                current_conv_id,
                conv_title,
                chat_history,
                chatbot,
                welcome_panel,
                chat_status_msg,
            ],
        )

        del_conv_btn.click(
            handle_delete_conversation,
            [current_token, current_conv_id],
            [
                chat_status_msg,
                conv_dropdown,
                current_conv_id,
                conv_title,
                chat_history,
                welcome_panel,
                chatbot,
            ],
        )

        stream_toggle.change(lambda x: x, [stream_toggle], [stream_mode])

        toggle_sidebar_btn.click(
            toggle_sidebar,
            [sidebar_collapsed, current_token],
            [sidebar_content, toggle_sidebar_btn, sidebar_collapsed],
        )

        # 发送（回车 / 按钮）
        # outputs 去掉了 medical_panel，因为卡片已嵌入消息体
        _send_outputs = [msg_input, chatbot, chat_history, current_medical, send_btn]

        msg_input.submit(
            handle_send_message,
            [
                msg_input,
                chat_history,
                current_user_id,
                current_conv_id,
                current_token,
                stream_mode,
            ],
            _send_outputs,
        ).then(update_welcome_visibility, [chatbot], [welcome_panel, chatbot])

        send_btn.click(
            handle_send_message,
            [
                msg_input,
                chat_history,
                current_user_id,
                current_conv_id,
                current_token,
                stream_mode,
            ],
            _send_outputs,
        ).then(update_welcome_visibility, [chatbot], [welcome_panel, chatbot])

        clear_btn.click(
            clear_chat,
            None,
            [chatbot, chat_history, current_medical, welcome_panel],
        )

        # 建议卡片点击 → 填充输入框
        for btn, text in sug_btns:
            btn.click(lambda t=text: t, None, msg_input)

        # ══════════════════════════════════════════════════════════════════════
        # 文档管理事件绑定
        # ══════════════════════════════════════════════════════════════════════

        # 上传文档
        upload_doc_btn.click(
            handle_upload_document,
            [current_token, doc_upload_file, doc_type_dropdown],
            [upload_doc_status],
        )

        # 刷新统计
        refresh_stats_btn.click(
            handle_refresh_stats, [current_token], [doc_stats_display]
        )

        # 查看文档列表
        refresh_docs_btn.click(handle_refresh_docs, [current_token], [doc_list_display])

        # 删除文档
        delete_doc_btn.click(
            handle_delete_document, [current_token, delete_doc_md5], [delete_doc_status]
        )

    return app


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import socket

    init_db()  # 初始化数据库

    def pick_port(start, end):
        for port in range(start, end + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
        return None

    selected_port = pick_port(7860, 7860) or pick_port(7861, 7870)
    if not selected_port:
        raise OSError("No available port found")

    logger.info(f"Starting Gradio UI on port {selected_port}...")
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=selected_port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue", secondary_hue="cyan", neutral_hue="slate"
        ),
        css=CUSTOM_CSS,
    )
