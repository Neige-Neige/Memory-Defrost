# Copyright (c) 2026 Neige-Neige
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# memory_server_http.py
# 通用 MCP 记忆服务 - 云端版本 (HTTP/SSE 传输)
# 使用 PostgreSQL + Gemini Embedding 语义搜索
# 兼容任何支持 MCP 协议的客户端

import os
import requests
import numpy as np
from datetime import datetime
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn
import psycopg2
from psycopg2.extras import RealDictCursor

# 配置
DATABASE_URL = os.environ.get("DATABASE_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# 使用最新的 gemini-embedding-001（3072维，100+语言支持）
# 注意：如果从 text-embedding-004 切换，需要重新生成所有 embedding
GEMINI_EMBEDDING_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

# 工具名称前缀（用于区分多个实例，避免重复声明错误）
# 设置环境变量 TOOL_PREFIX 来自定义，例如 TOOL_PREFIX=work_ 或 TOOL_PREFIX=personal_
TOOL_PREFIX = os.environ.get("TOOL_PREFIX", "")

# Embedding 缓存（减少 API 调用，加速响应）
EMBEDDING_CACHE = {}
EMBEDDING_CACHE_MAX_SIZE = 100  # 最多缓存 100 条

# 搜索模式：semantic（语义搜索，智能但慢）或 keyword（关键词搜索，快但需精确匹配）
# 设置环境变量 SEARCH_MODE 来切换，默认为 semantic
SEARCH_MODE = os.environ.get("SEARCH_MODE", "semantic").lower()

# 返回结果数量（默认 3 条，减少传输和处理时间）
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "3"))

# 渐进式注入：追踪 recall_memory 调用次数
# 简单实现：基于时间间隔判断是否为新会话
RECALL_COUNTER = {"count": 0, "last_call": None}
RECALL_SESSION_TIMEOUT = 300  # 5 分钟无调用视为新会话

# ========== 记忆缓存 ==========
# 缓存所有记忆到内存，避免每次 recall 都查数据库
_memory_cache: list[dict] = []
_cache_initialized = False


def init_memory_cache():
    """初始化记忆缓存（从数据库加载到内存）"""
    global _memory_cache, _cache_initialized
    if not DATABASE_URL:
        _cache_initialized = True
        return

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, content, tags, embedding, priority, category, created_at, updated_at FROM memories ORDER BY id")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        _memory_cache = []
        for row in rows:
            _memory_cache.append({
                "id": row["id"],
                "content": row["content"],
                "tags": row["tags"] or [],
                "embedding": row["embedding"] or [],
                "priority": row.get("priority", 3) or 3,
                "category": row.get("category", "general") or "general",
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None
            })
        _cache_initialized = True
        print(f"[CACHE] 已加载 {len(_memory_cache)} 条记忆到内存", flush=True)
    except Exception as e:
        print(f"[CACHE ERROR] {e}", flush=True)
        _cache_initialized = True


def get_cached_memories() -> list[dict]:
    """获取缓存的记忆（如果未初始化则先初始化）"""
    global _cache_initialized
    if not _cache_initialized:
        init_memory_cache()
    return _memory_cache


def add_to_cache(memory: dict):
    """添加记忆到缓存"""
    global _memory_cache
    _memory_cache.append(memory)


def update_cache(memory_id: int, **updates):
    """更新缓存中的记忆"""
    global _memory_cache
    for m in _memory_cache:
        if m["id"] == memory_id:
            m.update(updates)
            break


def remove_from_cache(memory_id: int):
    """从缓存中删除记忆"""
    global _memory_cache
    _memory_cache = [m for m in _memory_cache if m["id"] != memory_id]

# 创建 MCP Server
server_name = os.environ.get("SERVER_NAME", "memory-server")
server = Server(server_name)


def get_embedding(text: str, use_cache: bool = True) -> list[float]:
    """使用 Gemini 获取文本的 embedding 向量（带缓存）"""
    global EMBEDDING_CACHE

    # 生成缓存 key（取前 200 字符，避免 key 过长）
    cache_key = text[:200].strip().lower()

    # 检查缓存
    if use_cache and cache_key in EMBEDDING_CACHE:
        print(f"[EMBEDDING] 缓存命中: {text[:30]}...", flush=True)
        return EMBEDDING_CACHE[cache_key]

    if not GEMINI_API_KEY:
        print("[EMBEDDING] 警告: GEMINI_API_KEY 未设置", flush=True)
        return []

    try:
        url = f"{GEMINI_EMBEDDING_URL}?key={GEMINI_API_KEY}"

        payload = {
            "content": {
                "parts": [{"text": text}]
            }
        }

        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            embedding = result.get("embedding", {}).get("values", [])
            print(f"[EMBEDDING] API 成功: {text[:30]}... (维度: {len(embedding)})", flush=True)

            # 存入缓存
            if use_cache and embedding:
                # 简单的 LRU：超过上限时删除最早的
                if len(EMBEDDING_CACHE) >= EMBEDDING_CACHE_MAX_SIZE:
                    oldest_key = next(iter(EMBEDDING_CACHE))
                    del EMBEDDING_CACHE[oldest_key]
                EMBEDDING_CACHE[cache_key] = embedding

            return embedding
        else:
            print(f"[EMBEDDING] API错误: {response.status_code}", flush=True)
    except Exception as e:
        print(f"[EMBEDDING] 异常: {e}", flush=True)
    return []


def translate_query(query: str) -> list[str]:
    """使用 Gemini Flash 将查询词翻译成多语言，返回翻译列表"""
    if not GEMINI_API_KEY:
        return []

    # 检测是否需要翻译（纯 ASCII 大概率是英文，否则可能是中日韩）
    is_ascii = all(ord(c) < 128 for c in query.replace(" ", ""))

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

        # 根据查询语言选择翻译方向
        if is_ascii:
            prompt = f"Translate '{query}' to Chinese and Japanese. Return ONLY the translations, one per line, no explanations."
        else:
            prompt = f"Translate '{query}' to English. Return ONLY the translation, no explanations."

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 50, "temperature": 0}
        }

        response = requests.post(url, json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            translations = [t.strip() for t in text.strip().split("\n") if t.strip() and t.strip() != query]
            print(f"[TRANSLATE] '{query}' -> {translations}", flush=True)
            return translations[:3]  # 最多返回 3 个翻译
    except Exception as e:
        print(f"[TRANSLATE] 错误: {e}", flush=True)

    return []


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算两个向量的余弦相似度"""
    if not vec1 or not vec2:
        return 0.0
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_db_connection():
    """获取数据库连接"""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    """初始化数据库表"""
    conn = get_db_connection()
    cur = conn.cursor()
    # 创建表，包含 embedding、priority、category 列
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT[] DEFAULT '{}',
            embedding FLOAT8[],
            priority INTEGER DEFAULT 3,
            category VARCHAR(50) DEFAULT 'general',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # 动态添加缺失的列
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'embedding'
            ) THEN
                ALTER TABLE memories ADD COLUMN embedding FLOAT8[];
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'priority'
            ) THEN
                ALTER TABLE memories ADD COLUMN priority INTEGER DEFAULT 3;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'category'
            ) THEN
                ALTER TABLE memories ADD COLUMN category VARCHAR(50) DEFAULT 'general';
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'updated_at'
            ) THEN
                ALTER TABLE memories ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
            END IF;
        END $$;
    """)
    conn.commit()
    cur.close()
    conn.close()


def load_memories() -> list[dict]:
    """从数据库加载所有记忆"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, content, tags, embedding, priority, category, created_at, updated_at FROM memories ORDER BY id")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    memories = []
    for row in rows:
        memories.append({
            "id": row["id"],
            "content": row["content"],
            "tags": row["tags"] or [],
            "embedding": row["embedding"] or [],
            "priority": row.get("priority", 3) or 3,
            "category": row.get("category", "general") or "general",
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None
        })
    return memories


def save_memory_to_db(content: str, tags: list, priority: int = 3, category: str = "general") -> dict:
    """保存新记忆到数据库并更新缓存"""
    # 生成 embedding
    embedding = get_embedding(content)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO memories (content, tags, embedding, priority, category) VALUES (%s, %s, %s, %s, %s) RETURNING id, created_at",
        (content, tags, embedding if embedding else None, priority, category)
    )
    result = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    memory_data = {
        "id": result["id"],
        "content": content,
        "tags": tags,
        "embedding": embedding or [],
        "priority": priority,
        "category": category,
        "created_at": result["created_at"].isoformat(),
        "updated_at": None
    }

    # 同时更新缓存
    add_to_cache(memory_data)
    print(f"[CACHE] 已添加记忆 #{result['id']} 到缓存", flush=True)

    return memory_data


def update_memory_in_db(memory_id: int, content: str = None, tags: list = None, priority: int = None, category: str = None) -> dict | None:
    """更新数据库中的记忆并更新缓存"""
    conn = get_db_connection()
    cur = conn.cursor()

    # 先获取原记忆
    cur.execute("SELECT id, content, tags, priority, category FROM memories WHERE id = %s", (memory_id,))
    existing = cur.fetchone()

    if not existing:
        cur.close()
        conn.close()
        return None

    # 使用原值或新值
    new_content = content if content is not None else existing["content"]
    new_tags = tags if tags is not None else existing["tags"]
    new_priority = priority if priority is not None else existing.get("priority", 3)
    new_category = category if category is not None else existing.get("category", "general")

    # 如果内容变了，重新生成 embedding
    new_embedding = None
    if content is not None and content != existing["content"]:
        new_embedding = get_embedding(new_content)

    # 更新记录
    if new_embedding:
        cur.execute(
            "UPDATE memories SET content = %s, tags = %s, embedding = %s, priority = %s, category = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s RETURNING updated_at",
            (new_content, new_tags, new_embedding, new_priority, new_category, memory_id)
        )
    else:
        cur.execute(
            "UPDATE memories SET content = %s, tags = %s, priority = %s, category = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s RETURNING updated_at",
            (new_content, new_tags, new_priority, new_category, memory_id)
        )

    result = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    updated_at = result["updated_at"].isoformat() if result else None

    # 同时更新缓存
    cache_updates = {
        "content": new_content,
        "tags": new_tags,
        "priority": new_priority,
        "category": new_category,
        "updated_at": updated_at
    }
    if new_embedding:
        cache_updates["embedding"] = new_embedding
    update_cache(memory_id, **cache_updates)

    return {
        "id": memory_id,
        "content": new_content,
        "tags": new_tags,
        "priority": new_priority,
        "category": new_category,
        "updated_at": updated_at
    }


def delete_memory_by_id(memory_id: int) -> bool:
    """从数据库删除记忆并更新缓存"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()

    # 同时更新缓存
    if deleted:
        remove_from_cache(memory_id)

    return deleted


def search_memories(query: str, memories: list[dict], category: str = None) -> list[tuple[float, dict]]:
    """混合搜索 - 语义搜索 + 关键词搜索 + 多语言翻译

    支持跨语言搜索：自动翻译查询词，用多语言同时搜索
    """
    # 按分类筛选
    if category:
        memories = [m for m in memories if m.get("category", "general") == category]

    # 纯关键词模式
    if SEARCH_MODE == "keyword":
        return search_memories_keyword(query, memories, MAX_RESULTS, category=None)

    # 获取翻译（多语言查询）
    all_queries = [query] + translate_query(query)
    print(f"[SEARCH] 多语言查询: {all_queries}", flush=True)

    # 用 dict 存储每个记忆的最高分（避免重复）
    scores_by_id = {}

    for q in all_queries:
        q_embedding = get_embedding(q)
        q_lower = q.lower()

        for m in memories:
            memory_id = m["id"]
            semantic_score = 0
            keyword_score = 0

            # 1. 语义相似度
            if q_embedding and m.get("embedding"):
                semantic_score = cosine_similarity(q_embedding, m["embedding"])

            # 2. 关键词匹配（content + tags）
            content_lower = m["content"].lower()

            # content 完全匹配
            if q_lower in content_lower:
                keyword_score += 0.3

            # tags 匹配
            for tag in m.get("tags", []):
                if q_lower in tag.lower() or tag.lower() in q_lower:
                    keyword_score += 0.25

            # 分词匹配
            for word in q_lower.split():
                if len(word) >= 2 and word in content_lower:
                    keyword_score += 0.1

            # 3. 优先级加成
            priority_boost = (6 - m.get("priority", 3)) * 0.05

            # 4. 综合得分
            base_score = max(semantic_score, keyword_score)
            if semantic_score > 0.3 and keyword_score > 0:
                base_score += 0.1

            final_score = base_score + priority_boost

            # 保留最高分
            if final_score > 0.25:
                if memory_id not in scores_by_id or final_score > scores_by_id[memory_id][0]:
                    scores_by_id[memory_id] = (final_score, m)

    # 排序返回
    results = list(scores_by_id.values())
    results.sort(key=lambda x: x[0], reverse=True)

    return results[:MAX_RESULTS]


def search_memories_keyword(query: str, memories: list[dict], top_k: int = None, category: str = None) -> list[tuple[float, dict]]:
    """关键词搜索（备用），返回 (分数, 记忆) 列表"""
    # 按分类筛选
    if category:
        memories = [m for m in memories if m.get("category", "general") == category]

    query_lower = query.lower()
    scored = []

    for m in memories:
        score = 0
        content_lower = m["content"].lower()

        if query_lower in content_lower:
            score += 10

        for tag in m.get("tags", []):
            if query_lower in tag.lower():
                score += 5

        for word in query_lower.split():
            if word in content_lower:
                score += 2

        # 优先级加成
        priority_boost = (6 - m.get("priority", 3))  # 1-5 对应 5-1
        score += priority_boost

        if score > 0:
            scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k or MAX_RESULTS]


def get_memory_stats(memories: list[dict]) -> dict:
    """获取记忆统计信息"""
    if not memories:
        return {
            "total": 0,
            "by_category": {},
            "by_priority": {},
            "by_tag": {},
            "with_embedding": 0
        }

    by_category = {}
    by_priority = {}
    by_tag = {}
    with_embedding = 0

    for m in memories:
        # 按分类统计
        cat = m.get("category", "general")
        by_category[cat] = by_category.get(cat, 0) + 1

        # 按优先级统计
        pri = str(m.get("priority", 3))
        by_priority[pri] = by_priority.get(pri, 0) + 1

        # 按标签统计
        for tag in m.get("tags", []):
            by_tag[tag] = by_tag.get(tag, 0) + 1

        # 统计有 embedding 的记忆
        if m.get("embedding"):
            with_embedding += 1

    return {
        "total": len(memories),
        "by_category": by_category,
        "by_priority": by_priority,
        "by_tag": by_tag,
        "with_embedding": with_embedding
    }


# 预定义的记忆分类
MEMORY_CATEGORIES = ["general", "preference", "work", "personal", "habit", "skill", "goal"]

# 优先级说明
PRIORITY_LEVELS = {
    1: "最高 - 核心个人信息",
    2: "高 - 重要偏好或习惯",
    3: "中 - 一般信息（默认）",
    4: "低 - 临时或次要信息",
    5: "最低 - 可能过时的信息"
}


def get_tool_name(base_name: str) -> str:
    """生成带前缀的工具名称"""
    return f"{TOOL_PREFIX}{base_name}"


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的工具（精简描述以减少 token 消耗）"""
    p = f"[{TOOL_PREFIX.rstrip('_')}] " if TOOL_PREFIX else ""

    return [
        Tool(
            name=get_tool_name("recall_memory"),
            description=f"{p}Search memories. Use when user asks about past conversations or stored info.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords"},
                    "category": {"type": "string", "enum": MEMORY_CATEGORIES}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name=get_tool_name("save_memory"),
            description=f"{p}Save important user info (preferences, habits, work, personal).",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Memory content"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 5, "description": "1=highest 5=lowest"},
                    "category": {"type": "string", "enum": MEMORY_CATEGORIES}
                },
                "required": ["content"]
            }
        ),
        Tool(
            name=get_tool_name("update_memory"),
            description=f"{p}Update existing memory by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "integer"},
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                    "category": {"type": "string", "enum": MEMORY_CATEGORIES}
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name=get_tool_name("list_all_memories"),
            description=f"{p}List all saved memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": MEMORY_CATEGORIES}
                }
            }
        ),
        Tool(
            name=get_tool_name("delete_memory"),
            description=f"{p}Delete memory by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "integer"}
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name=get_tool_name("memory_stats"),
            description=f"{p}Show memory statistics.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name=get_tool_name("reset_session"),
            description=f"{p}Reset memory session. Use at start of new conversation.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name=get_tool_name("regenerate_embeddings"),
            description=f"{p}Regenerate all embeddings (use after changing embedding model).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


def format_priority(priority: int) -> str:
    """格式化优先级显示"""
    symbols = {1: "★★★", 2: "★★☆", 3: "★☆☆", 4: "☆☆☆", 5: "·"}
    return symbols.get(priority, "★☆☆")


def get_base_tool_name(name: str) -> str:
    """从带前缀的工具名中提取基础名称"""
    if TOOL_PREFIX and name.startswith(TOOL_PREFIX):
        return name[len(TOOL_PREFIX):]
    return name


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    global RECALL_COUNTER  # 移到函数开始处

    # 提取基础工具名（去掉前缀）
    base_name = get_base_tool_name(name)

    if base_name == "recall_memory":
        query = arguments.get("query", "")
        category = arguments.get("category")

        # 服务端追踪调用次数（渐进式注入）
        now = datetime.now()

        # 检查是否为新会话（超过 5 分钟没调用）
        if RECALL_COUNTER["last_call"] is None:
            RECALL_COUNTER = {"count": 1, "last_call": now}
        else:
            time_diff = (now - RECALL_COUNTER["last_call"]).total_seconds()
            if time_diff > RECALL_SESSION_TIMEOUT:
                # 新会话，重置计数
                RECALL_COUNTER = {"count": 1, "last_call": now}
            else:
                # 同一会话，计数 +1
                RECALL_COUNTER["count"] += 1
                RECALL_COUNTER["last_call"] = now

        recall_count = RECALL_COUNTER["count"]

        # 调试日志
        print(f"[RECALL] query={query}, category={category}, recall_count={recall_count}, raw_args={arguments}", flush=True)

        memories = get_cached_memories()

        if not memories:
            return [TextContent(type="text", text="没有找到任何记忆。")]

        # 搜索记忆（根据 SEARCH_MODE 自动选择语义或关键词搜索）
        results = search_memories(query, memories, category=category)

        if not results:
            cat_hint = f"（分类: {category}）" if category else ""
            return [TextContent(type="text", text=f"没有找到与「{query}」相关的记忆{cat_hint}。")]

        # 渐进式注入（新版）
        # 第 1 次: 返回 3 条核心记忆（身份/关系/语言风格 - 优先级 1-2 或 personal 分类）
        # 第 2 次: 返回 1 条最相关的，提示还有几条相关记忆
        # 第 3+ 次: 返回 MAX_RESULTS 条，由模型自行判断

        total_related = len(results)  # 记录总相关数量

        if recall_count == 1:
            # 第一次：优先返回核心记忆（身份确认、关系确认、语言风格）
            # 筛选优先级 1-2 或 personal 分类的记忆
            core_memories = [
                (score, m) for score, m in results
                if m.get("priority", 3) <= 2 or m.get("category") == "personal"
            ]

            if core_memories:
                # 有核心记忆，返回最多 3 条
                display_results = core_memories[:3]
            else:
                # 没有核心记忆，返回相关度最高的 3 条
                display_results = results[:3]

            result = "📌 核心记忆:\n"
            for i, (score, m) in enumerate(display_results, 1):
                tags_str = ", ".join(m.get("tags", [])[:3]) if m.get("tags") else ""
                content_short = m["content"][:50] + "..." if len(m["content"]) > 50 else m["content"]
                result += f"{i}. {content_short}"
                if tags_str:
                    result += f" ({tags_str})"
                result += "\n"

            if total_related > len(display_results):
                result += f"\n💡 还有 {total_related - len(display_results)} 条相关记忆"

            return [TextContent(type="text", text=result.strip())]

        elif recall_count == 2:
            # 第二次：返回与当前话题最相关的 1 条，提示还有几条
            score, top_mem = results[0]
            tags_str = ", ".join(top_mem.get("tags", [])) if top_mem.get("tags") else ""
            priority_str = format_priority(top_mem.get("priority", 3))

            result = f"🎯 最相关记忆:\n{top_mem['content']}"
            if tags_str:
                result += f"\n标签: {tags_str}"
            result += f"\n优先级: {priority_str} | 分类: {top_mem.get('category', 'general')}"

            if total_related > 1:
                result += f"\n\n💡 此话题还有 {total_related - 1} 条相关记忆"
                result += f"\n📋 使用 list_all_memories 可查看全部记忆"

            return [TextContent(type="text", text=result)]

        else:
            # 第 3+ 次：返回 MAX_RESULTS 条，正常显示
            display_results = results[:MAX_RESULTS]

            result = f"🔍 找到 {total_related} 条相关记忆:\n"
            for i, (score, m) in enumerate(display_results, 1):
                tags_str = ", ".join(m.get("tags", [])[:2]) if m.get("tags") else ""
                content_short = m["content"][:40] + "..." if len(m["content"]) > 40 else m["content"]
                result += f"{i}. {content_short}"
                if tags_str:
                    result += f" ({tags_str})"
                result += "\n"

            if total_related > MAX_RESULTS:
                result += f"\n(显示前 {MAX_RESULTS} 条，共 {total_related} 条相关)"

            return [TextContent(type="text", text=result.strip())]

    elif base_name == "save_memory":
        content = arguments.get("content", "")
        tags = arguments.get("tags", [])
        priority = arguments.get("priority", 3)
        category = arguments.get("category", "general")

        if not content:
            return [TextContent(type="text", text="记忆内容不能为空。")]

        # 验证优先级
        if priority < 1 or priority > 5:
            priority = 3

        # 验证分类
        if category not in MEMORY_CATEGORIES:
            category = "general"

        new_memory = save_memory_to_db(content, tags, priority, category)
        priority_str = format_priority(priority)
        return [TextContent(type="text", text=f"已保存记忆 [{new_memory['id']}]: {content}\n优先级: {priority_str} | 分类: {category}")]

    elif base_name == "update_memory":
        memory_id = arguments.get("memory_id")
        content = arguments.get("content")
        tags = arguments.get("tags")
        priority = arguments.get("priority")
        category = arguments.get("category")

        if memory_id is None:
            return [TextContent(type="text", text="请提供要更新的记忆 ID。")]

        # 验证优先级
        if priority is not None and (priority < 1 or priority > 5):
            return [TextContent(type="text", text="优先级必须在 1-5 之间。")]

        # 验证分类
        if category is not None and category not in MEMORY_CATEGORIES:
            return [TextContent(type="text", text=f"分类必须是以下之一: {', '.join(MEMORY_CATEGORIES)}")]

        updated = update_memory_in_db(memory_id, content, tags, priority, category)

        if not updated:
            return [TextContent(type="text", text=f"未找到 ID 为 {memory_id} 的记忆。")]

        result = f"已更新记忆 [{memory_id}]:\n"
        result += f"- 内容: {updated['content']}\n"
        result += f"- 标签: {', '.join(updated['tags']) if updated['tags'] else '无'}\n"
        result += f"- 优先级: {format_priority(updated['priority'])}\n"
        result += f"- 分类: {updated['category']}"

        return [TextContent(type="text", text=result)]

    elif base_name == "list_all_memories":
        category = arguments.get("category")
        memories = get_cached_memories()

        # 按分类筛选
        if category:
            memories = [m for m in memories if m.get("category", "general") == category]

        if not memories:
            cat_hint = f"（分类: {category}）" if category else ""
            return [TextContent(type="text", text=f"目前没有保存任何记忆{cat_hint}。")]

        result = f"共有 {len(memories)} 条记忆"
        if category:
            result += f"（分类: {category}）"
        result += "：\n"

        for m in memories:
            tags_str = ", ".join(m.get("tags", [])) if m.get("tags") else "无"
            priority_str = format_priority(m.get("priority", 3))
            cat_str = m.get("category", "general")
            result += f"- [{m['id']}] {priority_str} {m['content']}\n"
            result += f"  └ 分类: {cat_str} | 标签: {tags_str}\n"

        return [TextContent(type="text", text=result)]

    elif base_name == "delete_memory":
        memory_id = arguments.get("memory_id")

        if memory_id is None:
            return [TextContent(type="text", text="请提供要删除的记忆 ID。")]

        if delete_memory_by_id(memory_id):
            return [TextContent(type="text", text=f"已删除记忆 [{memory_id}]。")]
        else:
            return [TextContent(type="text", text=f"未找到 ID 为 {memory_id} 的记忆。")]

    elif base_name == "reset_session":
        old_count = RECALL_COUNTER["count"]
        RECALL_COUNTER = {"count": 0, "last_call": None}
        print(f"[RESET] Session reset. Previous recall_count was {old_count}", flush=True)
        return [TextContent(type="text", text=f"会话已重置。(之前调用次数: {old_count})")]

    elif base_name == "memory_stats":
        memories = get_cached_memories()
        stats = get_memory_stats(memories)

        if stats["total"] == 0:
            return [TextContent(type="text", text="目前没有保存任何记忆。")]

        result = "📊 记忆统计\n"
        result += "=" * 30 + "\n"
        result += f"总记忆数: {stats['total']}\n"
        result += f"语义搜索支持: {stats['with_embedding']}/{stats['total']}\n\n"

        result += "按分类:\n"
        for cat, count in sorted(stats["by_category"].items()):
            result += f"  - {cat}: {count}\n"

        result += "\n按优先级:\n"
        for pri in ["1", "2", "3", "4", "5"]:
            if pri in stats["by_priority"]:
                result += f"  - {format_priority(int(pri))} ({pri}): {stats['by_priority'][pri]}\n"

        if stats["by_tag"]:
            result += "\n热门标签 (Top 5):\n"
            sorted_tags = sorted(stats["by_tag"].items(), key=lambda x: x[1], reverse=True)[:5]
            for tag, count in sorted_tags:
                result += f"  - {tag}: {count}\n"

        return [TextContent(type="text", text=result)]

    elif base_name == "regenerate_embeddings":
        # 重新生成所有记忆的 embedding（切换模型后使用）
        memories = get_cached_memories()
        if not memories:
            return [TextContent(type="text", text="没有记忆需要重新生成。")]

        updated = 0
        failed = 0
        for m in memories:
            try:
                new_embedding = get_embedding(m["content"], use_cache=False)
                if new_embedding:
                    # 更新数据库
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("UPDATE memories SET embedding = %s WHERE id = %s", (new_embedding, m["id"]))
                    conn.commit()
                    cur.close()
                    conn.close()
                    # 更新缓存
                    m["embedding"] = new_embedding
                    updated += 1
                    print(f"[REGEN] 已更新记忆 #{m['id']} 的 embedding (维度: {len(new_embedding)})", flush=True)
                else:
                    failed += 1
            except Exception as e:
                print(f"[REGEN ERROR] 记忆 #{m['id']}: {e}", flush=True)
                failed += 1

        result = f"✅ Embedding 重新生成完成\n"
        result += f"- 成功: {updated}\n"
        result += f"- 失败: {failed}\n"
        if updated > 0:
            # 获取新的维度
            sample_dim = len(memories[0].get("embedding", [])) if memories else 0
            result += f"- 新维度: {sample_dim}"
        return [TextContent(type="text", text=result)]

    return [TextContent(type="text", text=f"未知工具: {name}")]


# 创建 SSE transport
sse = SseServerTransport("/messages/")


async def handle_sse(request):
    """处理 SSE 连接"""
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )


async def health_check(request):
    """健康检查端点"""
    embedding_status = "enabled" if GEMINI_API_KEY else "disabled"
    return JSONResponse({
        "status": "ok",
        "service": "memory-mcp",
        "storage": "postgresql",
        "semantic_search": embedding_status
    })


# 创建 Starlette 应用
app = Starlette(
    routes=[
        Route("/health", health_check),
        Route("/sse", handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]
)


if __name__ == "__main__":
    # 初始化数据库
    if DATABASE_URL:
        print("初始化数据库...")
        init_db()
        print("数据库初始化完成!")

        # 初始化记忆缓存
        print("加载记忆缓存...")
        init_memory_cache()

        # 检测 embedding 维度，如果是旧版（768维）则自动重新生成
        if GEMINI_API_KEY and _memory_cache:
            sample = _memory_cache[0].get("embedding", [])
            if sample and len(sample) == 768:
                print(f"[AUTO-REGEN] 检测到旧版 embedding (768维)，正在自动升级到 3072 维...")
                updated = 0
                for m in _memory_cache:
                    try:
                        new_embedding = get_embedding(m["content"], use_cache=False)
                        if new_embedding:
                            conn = get_db_connection()
                            cur = conn.cursor()
                            cur.execute("UPDATE memories SET embedding = %s WHERE id = %s", (new_embedding, m["id"]))
                            conn.commit()
                            cur.close()
                            conn.close()
                            m["embedding"] = new_embedding
                            updated += 1
                    except Exception as e:
                        print(f"[AUTO-REGEN ERROR] 记忆 #{m['id']}: {e}", flush=True)
                print(f"[AUTO-REGEN] 完成！已更新 {updated} 条记忆的 embedding")
            elif sample:
                print(f"[EMBEDDING] 当前维度: {len(sample)} (已是最新)")
    else:
        print("警告: 未设置 DATABASE_URL，将无法保存数据")

    if GEMINI_API_KEY:
        print(f"Gemini Embedding: 已启用 (缓存上限: {EMBEDDING_CACHE_MAX_SIZE})")
    else:
        print("Gemini Embedding: 未启用（将使用关键词搜索）")

    print(f"搜索模式: {SEARCH_MODE} ({'语义搜索' if SEARCH_MODE == 'semantic' else '关键词搜索'})")
    print(f"返回结果数: {MAX_RESULTS}")

    # Railway 使用 PORT 环境变量
    port = int(os.environ.get("PORT", 8000))

    print("=" * 50)
    print("Memory MCP Server (PostgreSQL + Embedding)")
    print("=" * 50)
    print(f"服务名称: {server_name}")
    print(f"服务端口: {port}")
    print(f"工具前缀: {TOOL_PREFIX if TOOL_PREFIX else '(无)'}")
    print("SSE 端点: /sse")
    print("健康检查: /health")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=port)
