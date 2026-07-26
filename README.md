# Memory MCP Server

> **License: AGPL-3.0-or-later** · Copyright (c) 2026 Neige-Neige
> If you distribute this software or offer it as a network service (modified or not),
> you must make the complete corresponding source available under AGPL-3.0-or-later.
> For closed-source commercial licensing, please [open an issue](../../issues).
> 本ソフトウェアを配布する、またはネットワークサービスとして提供する場合（改変の有無を問わず）、
> 対応するソースコード全体を AGPL-3.0-or-later で公開する必要があります。
> 分发本软件或将其作为网络服务提供时（无论是否修改），必须以 AGPL-3.0-or-later 公开完整的对应源代码。


一个通用的 MCP (Model Context Protocol) 记忆服务，为任何支持 MCP 协议的 AI 客户端提供长期记忆能力。支持语义搜索、记忆分类和优先级管理。

## 特性

- **语义搜索** - 使用 Gemini gemini-embedding-001（3072维，100+语言支持，免费）
- **多语言搜索** - 自动翻译查询词，支持中英日跨语言检索
- **记忆分类** - 7 种预设分类，方便组织管理
- **优先级系统** - 5 级重要性标记，影响搜索排序
- **内存缓存** - 启动时加载记忆到内存，搜索更快
- **自动升级** - 检测旧版 embedding 并自动重新生成
- **渐进式注入** - 对话初期返回少量记忆，后期返回更多
- **双版本支持** - 本地 JSON 存储 / 云端 PostgreSQL 存储
- **MCP 标准协议** - 兼容任何 MCP 客户端

## 可用工具

| 工具 | 说明 | 触发场景 |
|------|------|----------|
| `save_memory` | 保存重要的用户信息 | 用户提到重要个人信息、偏好、习惯 |
| `recall_memory` | 按关键词/语义检索相关记忆 | 用户问「之前聊过什么」「你还记得吗」 |
| `update_memory` | 更新已保存的记忆 | 需要修改之前保存的信息 |
| `list_all_memories` | 查看所有已保存的记忆 | 用户要求列出所有记忆 |
| `delete_memory` | 删除指定记忆 | 用户要求忘记某些信息 |
| `memory_stats` | 显示记忆统计信息 | 查看记忆概览和分布 |
| `regenerate_embeddings` | 重新生成所有记忆的 embedding | 切换 embedding 模型后使用 |

## 记忆优先级

每条记忆可以设置优先级（1-5），搜索时高优先级记忆会获得加成：

| 级别 | 说明 | 显示 |
|------|------|------|
| 1 | 最高 - 核心个人信息 | ★★★ |
| 2 | 高 - 重要偏好或习惯 | ★★☆ |
| 3 | 中 - 一般信息（默认） | ★☆☆ |
| 4 | 低 - 临时或次要信息 | ☆☆☆ |
| 5 | 最低 - 可能过时的信息 | · |

## 记忆分类

每条记忆可以归类到以下分类：

- `general` - 通用（默认）
- `preference` - 偏好设置
- `work` - 工作相关
- `personal` - 个人信息
- `habit` - 习惯
- `skill` - 技能
- `goal` - 目标

## 渐进式注入

`recall_memory` 会根据会话中的调用次数，智能调整返回的记忆数量和类型：

| 调用次数 | 返回内容 | 目的 |
|---------|---------|------|
| 第 1 次 | 最多 3 条核心记忆（优先级 1-2 或 personal 分类） | 身份确认、关系确认、语言风格 |
| 第 2 次 | 1 条最相关记忆 + 提示剩余数量 | 聚焦当前话题，提示可查看更多 |
| 第 3+ 次 | MAX_RESULTS 条（默认 3） | 正常搜索，完整返回 |

> **会话重置**：超过 5 分钟未调用 `recall_memory`，计数器自动归零

这个设计确保：
- 对话开始时快速建立用户画像
- 对话中期聚焦当前话题
- 对话后期提供完整信息

## 两种部署方式

### 1. 本地版本 (memory_server.py)

适合本地运行的 MCP 客户端，使用 stdio 传输协议。

- 记忆保存在本地 `memories.json` 文件
- 使用关键词搜索
- 数据完全本地化，隐私保护

**运行方式：**
```bash
python memory_server.py
```

**MCP 客户端配置示例：**
```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["/path/to/memory_server.py"]
    }
  }
}
```

### 2. 云端版本 (memory_server_http.py)

适合远程访问的 MCP 客户端，使用 HTTP/SSE 传输协议。

- 使用 PostgreSQL 数据库存储
- 集成 Gemini Embedding API 进行语义搜索
- 支持多客户端连接
- **需要部署到云平台**（Railway、Render、Fly.io 等）或有公网 IP 的服务器

> **什么时候需要云端版本？**
> - 使用 **Claude.ai 网页版** 连接 MCP 服务时（需要公网可访问的 URL）
> - 需要从多台设备访问同一份记忆时
> - 本地版本使用 stdio 协议，仅支持 Claude Desktop 等本地客户端

**运行方式：**
```bash
# 设置环境变量
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
export GEMINI_API_KEY="your-gemini-api-key"  # 可选，用于语义搜索
export PORT=8000

# 可选：设置工具前缀（用于运行多个实例时避免工具名冲突）
export TOOL_PREFIX="work_"      # 工具名将变为 work_save_memory, work_recall_memory 等
export SERVER_NAME="work-memory"  # 服务器名称

python memory_server_http.py
```

### 多实例部署

如果需要同时运行多个 Memory MCP 实例（例如工作记忆和个人记忆分开存储），需要设置不同的 `TOOL_PREFIX` 避免工具名冲突：

**实例 1 - 工作记忆：**
```bash
export TOOL_PREFIX="work_"
export SERVER_NAME="work-memory"
export DATABASE_URL="postgresql://..."  # 工作数据库
```

**实例 2 - 个人记忆：**
```bash
export TOOL_PREFIX="personal_"
export SERVER_NAME="personal-memory"
export DATABASE_URL="postgresql://..."  # 个人数据库
```

这样工具名会变成：
- `work_save_memory`, `work_recall_memory`, ...
- `personal_save_memory`, `personal_recall_memory`, ...

**MCP 客户端连接：**
- SSE 端点: `http://your-server:8000/sse`
- 健康检查: `http://your-server:8000/health`

## 安装

```bash
pip install -r requirements.txt
```

**依赖说明：**
- `mcp` - MCP 协议库
- `starlette`, `uvicorn` - HTTP 服务器（云端版本）
- `psycopg2-binary` - PostgreSQL 驱动（云端版本）
- `numpy` - 向量计算
- `requests` - HTTP 客户端

## 云端部署

### Railway 部署

1. Fork 此仓库到你的 GitHub
2. 在 Railway 创建新项目，连接 GitHub 仓库
3. 添加 PostgreSQL 数据库服务
4. 设置环境变量：
   - `DATABASE_URL` - 自动由 Railway 设置
   - `GEMINI_API_KEY` - 你的 Gemini API 密钥（可选）
5. 部署完成后获取服务 URL

### 其他云平台

项目包含 `Procfile`，兼容 Heroku、Render 等平台：
```
web: python memory_server_http.py
```

## 使用示例

### 保存记忆

```
"记住我喜欢用 TypeScript 写代码"
"我的项目用的是 React + Vite，这是工作相关的"
"记住我不喜欢在代码里加太多注释，优先级设为高"
```

### 回忆信息

```
"我之前告诉过你什么编程偏好？"
"你还记得我的项目用什么技术栈吗？"
"只在工作分类里找一下相关记忆"
```

### 更新记忆

```
"把记忆 3 的优先级改成最高"
"更新记忆 5，把分类改成 work"
```

### 查看统计

```
"显示记忆统计"
"列出所有 work 分类的记忆"
```

## 数据格式

### 本地 JSON 格式

```json
[
  {
    "id": 1,
    "content": "用户喜欢喝咖啡",
    "tags": ["偏好", "饮食"],
    "priority": 2,
    "category": "preference",
    "created_at": "2024-01-01T12:00:00",
    "updated_at": "2024-01-01T12:00:00"
  }
]
```

### 云端数据库 Schema

```sql
CREATE TABLE memories (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    tags TEXT[] DEFAULT '{}',
    embedding FLOAT8[],
    priority INTEGER DEFAULT 3,
    category VARCHAR(50) DEFAULT 'general',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

## API 端点（云端版本）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/sse` | GET | MCP SSE 连接端点 |
| `/messages/` | POST | MCP 消息处理 |
| `/health` | GET | 健康检查 |

**健康检查响应示例：**
```json
{
  "status": "ok",
  "service": "memory-mcp",
  "storage": "postgresql",
  "semantic_search": "enabled"
}
```

## 技术架构

```
┌──────────────────────────────────────────────────────────┐
│                    MCP Client (任意)                      │
│         (Claude Desktop, API Client, Custom App)         │
└─────────────────────────┬────────────────────────────────┘
                          │ MCP Protocol
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐      ┌─────────────────────────────┐
│   本地版本 (stdio)   │      │     云端版本 (HTTP/SSE)     │
│  memory_server.py   │      │   memory_server_http.py     │
├─────────────────────┤      ├─────────────────────────────┤
│ - JSON 文件存储      │      │ - PostgreSQL 数据库          │
│ - 关键词搜索         │      │ - Gemini Embedding 语义搜索  │
│ - 优先级/分类管理    │      │ - 优先级/分类管理             │
└─────────────────────┘      └─────────────────────────────┘
```

## 兼容的 MCP 客户端

本服务兼容任何支持 MCP 协议的客户端，包括但不限于：

- Claude Desktop
- 自定义 API 客户端
- MCP Inspector（调试工具）
- 其他 MCP 兼容应用

## 使用建议

> **关于客户端内置记忆功能**
>
> 部分 MCP 客户端（如 Claude Desktop）自带记忆功能，其内置记忆的权重通常比外部 MCP 服务更高。
>
> - **建议**：根据自身需求决定是否关闭客户端的内置记忆功能
> - **注意**：如果关闭客户端内置记忆，所有记忆检索都会走本服务的 `recall_memory`，响应时间可能会稍长（需要进行语义搜索和数据库查询）
>
> 请根据实际使用场景自行权衡。

## 故障排除

### 语义搜索不工作

1. 检查 `GEMINI_API_KEY` 环境变量是否设置
2. 查看服务日志中的 `[EMBEDDING]` 相关信息
3. 如果 API 出错，会自动降级为关键词搜索

### 数据库连接失败

1. 检查 `DATABASE_URL` 格式是否正确
2. 确认数据库服务正在运行
3. 检查网络连接和防火墙设置

### SSE 连接问题

1. 确认客户端支持 SSE 传输
2. 检查 CORS 设置（如果是跨域访问）
3. 确认端口没有被占用

## 自定义 Embedding 服务

本项目默认使用 Google Gemini Embedding API（免费），但你可以替换为其他 embedding 服务。

### 当前默认行为

- **有 GEMINI_API_KEY**: 使用 Gemini `gemini-embedding-001` 模型进行语义搜索
  - 3072 维向量，精度更高
  - 支持 100+ 语言，跨语言检索效果更好
  - 启动时自动检测旧版 embedding (768维) 并升级
- **无 GEMINI_API_KEY**: 自动降级为关键词搜索（仍然可用）

### 替换为其他 Embedding 服务

如果你想使用 OpenAI、Cohere、本地模型等其他 embedding 服务，需要修改 `memory_server_http.py` 中的 `get_embedding()` 函数：

```python
def get_embedding(text: str) -> list[float] | None:
    """获取文本的 embedding 向量"""
    # 替换为你的 embedding API 调用
    # 返回值应为 float 列表，如 [0.1, 0.2, ...]
    # 返回 None 表示失败，会降级为关键词搜索
    pass
```

**常见替代方案：**
- OpenAI `text-embedding-3-small`
- Cohere `embed-multilingual-v3.0`
- 本地部署的 sentence-transformers
- 其他支持中文的 embedding 模型

## 后续优化方向

- [x] 语义搜索（Gemini Embedding）
- [x] 多语言搜索（中英日跨语言检索）
- [x] 记忆优先级
- [x] 记忆分类管理
- [x] 记忆更新功能
- [x] 记忆统计功能
- [x] Embedding 自动升级
- [ ] 记忆过期机制
- [ ] 记忆导出/导入
- [ ] 记忆关联功能
- [ ] WebSocket 传输支持

## License

Copyright (c) 2026 Neige-Neige

This project is licensed under the **GNU Affero General Public License, version 3
or (at your option) any later version** (`AGPL-3.0-or-later`).
See [LICENSE](LICENSE) for the full text.

### What triggers the copyleft

AGPL is triggered by **distribution** and by **offering the software as a network
service** — not by "commercial use" as such. Concretely:

- Running an unmodified or modified copy **privately**, without distributing it and
  without offering it to others over a network → no obligation, even commercially
- **Distributing** the software, modified or not → you must provide the complete
  corresponding source under AGPL-3.0-or-later
- **Offering it as a network service** (SaaS, hosted MCP endpoint, bot, etc.), modified
  or not → you must make the complete corresponding source available **to its users**,
  even though you never hand out a copy of the software itself. This is the clause that
  distinguishes AGPL from plain GPL.
- In all of the above you must **retain this copyright notice and license**

### Commercial licensing

If you want to distribute this project, or run it as a service, **without releasing your
source code**, AGPL-3.0-or-later does not permit that. A separate commercial license is
available — please [open an issue](../../issues) to discuss.

### 中文

本项目采用 **AGPL-3.0-or-later**（GNU Affero 通用公共许可证第 3 版或任何更新版本）授权。

**触发 copyleft 的是"分发"和"以网络服务提供"，而不是"商业使用"本身：**

- 私下运行（不分发、不对外提供网络服务）→ 无义务，即使是商业环境
- **分发**本软件（无论是否修改）→ 必须以 AGPL-3.0-or-later 提供完整的对应源代码
- **作为网络服务提供**（SaaS、托管 MCP 端点、bot 等，无论是否修改）→ 必须向**使用者**公开
  完整的对应源代码，即使你从未分发过软件本体。这是 AGPL 区别于普通 GPL 的核心条款
- 以上任何情形都必须**保留原始的版权声明与授权条款**

如需在**不公开源代码**的前提下分发或提供服务，AGPL 不允许，请[开 issue](../../issues)联系作者
获取单独的商业授权。

### 日本語

本プロジェクトは **AGPL-3.0-or-later** ライセンスで公開されています。

copyleft が発動するのは「**配布**」と「**ネットワークサービスとしての提供**」であり、
「商用利用」そのものではありません:

- 私的に実行するだけ（配布せず、ネットワーク経由で他者に提供しない）→ 商用でも義務なし
- **配布**する場合（改変の有無を問わず）→ 対応するソースコード全体を AGPL-3.0-or-later で提供
- **ネットワークサービスとして提供**する場合（改変の有無を問わず）→ ソフトウェア本体を
  配布していなくても、**利用者に対して**対応するソースコード全体を開示する義務があります
- いずれの場合も**著作権表示とライセンスの保持**が必要です

ソースコードを公開せずに配布・サービス提供をご希望の場合は、[issue](../../issues) にて
商用ライセンスをご相談ください。

---

<p align="center">
  <a href="https://buymeacoffee.com/neige_neige">
    <img src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-yellow?style=flat-square&logo=buy-me-a-coffee">
  </a>
</p>

<p align="center">
  <sub>Built with 🌀 <a href="https://github.com/anthropics/claude-code">Claude Code</a></sub>
</p>
