# memory_server.py
# 通用 MCP 记忆服务 - 本地版本 (stdio 传输)
# 支持按需召回、优先级和分类管理
# 兼容任何支持 MCP 协议的客户端

import json
import os
from datetime import datetime
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 记忆存储路径
MEMORY_FILE = Path(__file__).parent / "memories.json"

# 创建 MCP Server
server = Server("memory-server")

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


def load_memories() -> list[dict]:
    """加载所有记忆"""
    if not MEMORY_FILE.exists():
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memories = json.load(f)
        # 兼容旧格式：添加缺失的字段
        for m in memories:
            if "priority" not in m:
                m["priority"] = 3
            if "category" not in m:
                m["category"] = "general"
            if "updated_at" not in m:
                m["updated_at"] = m.get("created_at")
        return memories


def save_memories(memories: list[dict]):
    """保存记忆到文件"""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)


def search_memories(query: str, memories: list[dict], top_k: int = 5, category: str = None) -> list[tuple[float, dict]]:
    """关键词匹配搜索，返回 (分数, 记忆) 列表"""
    # 按分类筛选
    if category:
        memories = [m for m in memories if m.get("category", "general") == category]

    query_lower = query.lower()
    scored = []

    for m in memories:
        score = 0
        content_lower = m["content"].lower()

        # 关键词完全匹配
        if query_lower in content_lower:
            score += 10

        # 标签匹配
        for tag in m.get("tags", []):
            if query_lower in tag.lower():
                score += 5

        # 部分词匹配
        for word in query_lower.split():
            if word in content_lower:
                score += 2

        # 优先级加成：priority 越高（1最高），分数加成越大
        priority_boost = (6 - m.get("priority", 3))  # 1-5 对应 5-1
        score += priority_boost

        if score > 0:
            # 归一化分数到 0-1 范围
            normalized_score = min(1.0, score / 20.0)
            scored.append((normalized_score, m))

    # 按分数排序，返回 top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def get_memory_stats(memories: list[dict]) -> dict:
    """获取记忆统计信息"""
    if not memories:
        return {
            "total": 0,
            "by_category": {},
            "by_priority": {},
            "by_tag": {}
        }

    by_category = {}
    by_priority = {}
    by_tag = {}

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

    return {
        "total": len(memories),
        "by_category": by_category,
        "by_priority": by_priority,
        "by_tag": by_tag
    }


def format_priority(priority: int) -> str:
    """格式化优先级显示"""
    symbols = {1: "★★★", 2: "★★☆", 3: "★☆☆", 4: "☆☆☆", 5: "·"}
    return symbols.get(priority, "★☆☆")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的工具"""
    return [
        Tool(
            name="recall_memory",
            description="当且仅当用户询问历史对话、之前聊过的内容、或需要回忆过去信息时调用。不要在普通对话中调用。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要回忆的内容关键词，如'咖啡偏好'、'工作'、'宠物'等"
                    },
                    "category": {
                        "type": "string",
                        "enum": MEMORY_CATEGORIES,
                        "description": "按分类筛选记忆（可选）"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="save_memory",
            description="保存用户提到的重要个人信息，如偏好、习惯、工作、家庭等。只保存有长期价值的信息。",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要保存的记忆内容"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "记忆标签，用于分类检索，如['偏好', '饮食']"
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "重要性：1=最高(核心信息) 2=高 3=中(默认) 4=低 5=最低"
                    },
                    "category": {
                        "type": "string",
                        "enum": MEMORY_CATEGORIES,
                        "description": "分类：general/preference/work/personal/habit/skill/goal"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="update_memory",
            description="更新已存在的记忆。可以修改内容、标签、优先级或分类。",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "integer",
                        "description": "要更新的记忆 ID"
                    },
                    "content": {
                        "type": "string",
                        "description": "新的记忆内容（可选，不填则保持原内容）"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "新的标签（可选）"
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "新的优先级（可选）"
                    },
                    "category": {
                        "type": "string",
                        "enum": MEMORY_CATEGORIES,
                        "description": "新的分类（可选）"
                    }
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="list_all_memories",
            description="列出所有已保存的记忆。可按分类筛选。",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": MEMORY_CATEGORIES,
                        "description": "按分类筛选（可选）"
                    }
                }
            }
        ),
        Tool(
            name="delete_memory",
            description="删除指定的记忆。需要用户确认后才能删除。",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "integer",
                        "description": "要删除的记忆 ID"
                    }
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="memory_stats",
            description="显示记忆统计信息，包括总数、各分类数量、各优先级数量等。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""

    if name == "recall_memory":
        query = arguments.get("query", "")
        category = arguments.get("category")
        memories = load_memories()

        if not memories:
            return [TextContent(type="text", text="没有找到任何记忆。")]

        results = search_memories(query, memories, category=category)

        if not results:
            cat_hint = f"（分类: {category}）" if category else ""
            return [TextContent(type="text", text=f"没有找到与「{query}」相关的记忆{cat_hint}。")]

        result = "找到以下相关记忆：\n"
        for score, m in results:
            tags_str = ", ".join(m.get("tags", [])) if m.get("tags") else "无"
            priority_str = format_priority(m.get("priority", 3))
            cat_str = m.get("category", "general")
            score_pct = int(score * 100)
            result += f"- [{m['id']}] {priority_str} {m['content']}\n"
            result += f"  └ 分类: {cat_str} | 标签: {tags_str} | 匹配度: {score_pct}%\n"

        return [TextContent(type="text", text=result)]

    elif name == "save_memory":
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

        memories = load_memories()

        # 生成新 ID
        new_id = max([m["id"] for m in memories], default=0) + 1

        new_memory = {
            "id": new_id,
            "content": content,
            "tags": tags,
            "priority": priority,
            "category": category,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        memories.append(new_memory)
        save_memories(memories)

        priority_str = format_priority(priority)
        return [TextContent(type="text", text=f"已保存记忆 [{new_id}]: {content}\n优先级: {priority_str} | 分类: {category}")]

    elif name == "update_memory":
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

        memories = load_memories()

        # 查找并更新记忆
        updated = None
        for m in memories:
            if m["id"] == memory_id:
                if content is not None:
                    m["content"] = content
                if tags is not None:
                    m["tags"] = tags
                if priority is not None:
                    m["priority"] = priority
                if category is not None:
                    m["category"] = category
                m["updated_at"] = datetime.now().isoformat()
                updated = m
                break

        if not updated:
            return [TextContent(type="text", text=f"未找到 ID 为 {memory_id} 的记忆。")]

        save_memories(memories)

        result = f"已更新记忆 [{memory_id}]:\n"
        result += f"- 内容: {updated['content']}\n"
        result += f"- 标签: {', '.join(updated['tags']) if updated['tags'] else '无'}\n"
        result += f"- 优先级: {format_priority(updated['priority'])}\n"
        result += f"- 分类: {updated['category']}"

        return [TextContent(type="text", text=result)]

    elif name == "list_all_memories":
        category = arguments.get("category")
        memories = load_memories()

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

    elif name == "delete_memory":
        memory_id = arguments.get("memory_id")

        if memory_id is None:
            return [TextContent(type="text", text="请提供要删除的记忆 ID。")]

        memories = load_memories()
        original_count = len(memories)
        memories = [m for m in memories if m["id"] != memory_id]

        if len(memories) == original_count:
            return [TextContent(type="text", text=f"未找到 ID 为 {memory_id} 的记忆。")]

        save_memories(memories)
        return [TextContent(type="text", text=f"已删除记忆 [{memory_id}]。")]

    elif name == "memory_stats":
        memories = load_memories()
        stats = get_memory_stats(memories)

        if stats["total"] == 0:
            return [TextContent(type="text", text="目前没有保存任何记忆。")]

        result = "📊 记忆统计\n"
        result += "=" * 30 + "\n"
        result += f"总记忆数: {stats['total']}\n\n"

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

    return [TextContent(type="text", text=f"未知工具: {name}")]


async def main():
    """启动 MCP 服务"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
