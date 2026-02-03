# Handoffs Agent

基于 LangChain Handoffs 模式的智能消费记录提取工具

## 项目简介

Handoffs Agent 是一个使用本地和远程 Ollama 模型进行两阶段处理的智能工具:
1. **本地模型**: 分析图片内容, 提取文本信息
2. **远程模型**: 从文本中提取结构化的消费记录

本项目实现了 LangChain Handoffs 模式, 通过状态驱动的方式管理处理流程, 工具函数通过更新状态变量来触发状态转换.

参考文档: [LangChain Handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs)

## 功能特性

- 📸 **图片分析**: 使用本地 OCR 模型分析图片, 提取文本内容
- 📝 **文本提取**: 支持直接输入文本或从图片中提取的文本
- 💰 **消费记录提取**: 自动提取消费时间、项目、分类和金额
- 🔄 **状态驱动**: 基于 Handoffs 模式的状态管理, 清晰的流程控制
- ✅ **数据验证**: 使用 Pydantic 进行严格的数据验证和类型检查
- 🌐 **多模型支持**: 支持本地和远程 Ollama 模型配置

## 项目结构

```
skills/
├── handoffs/
│   ├── __init__.py
│   ├── client_all.py          # Ollama 客户端配置
│   ├── handoffs_agent.py      # 核心 Handoffs Agent 实现
│   └── main.py                 # 辅助工具函数
├── pyproject.toml              # 项目依赖配置
├── README.md                   # 项目文档
└── uv.lock                     # 依赖锁定文件
```

## 安装

### 前置要求

- Python >= 3.11
- Ollama 服务 (本地和/或远程)

### 安装依赖

使用 `uv` 安装依赖:

```bash
uv pip install -e .
```

或使用 `pip`:

```bash
pip install -e .
```

### 环境变量配置

创建 `.env` 文件并配置以下环境变量:

```env
# 本地 Ollama 配置
OLLAMA_API_BASE_LOCAL=http://127.0.0.1:11434
OLLAMA_API_KEY_LOCAL=your_local_api_key

# 远程 Ollama 配置
OLLAMA_API_BASE_REMOTE=https://your-remote-ollama-host
OLLAMA_API_KEY_REMOTE=your_remote_api_key
```

## 使用方法

### 命令行使用

#### 从图片提取消费记录

```bash
python -m handoffs.handoffs_agent -i /path/to/image.jpg
```

#### 从文本提取消费记录

```bash
python -m handoffs.handoffs_agent -t "今天在大丰买衣服花了四十五元"
```

#### 指定模型

```bash
python -m handoffs.handoffs_agent \
  -i /path/to/image.jpg \
  --local-model deepseek-ocr \
  --remote-model deepseek-v3.1:671b-cloud
```

### Python API 使用

```python
from handoffs.handoffs_agent import HandoffsAgent

# 创建 agent
agent = HandoffsAgent(
    model_local="deepseek-ocr",
    model_remote="deepseek-v3.1:671b-cloud"
)

# 从图片处理
result = agent.process(image_path="/path/to/image.jpg")

# 从文本处理
result = agent.process(text="今天吃饭花了四块钱")

# 查看结果
print(result)
```

## 数据模型

### Consumption (消费记录)

```python
{
    "date": "2026-02-03 12:22:51",  # 消费时间
    "item": "衣服(大丰)",            # 消费项目
    "category": "服装-购买",          # 消费分类
    "amount": 45.00                  # 消费金额(负数表示支出)
}
```

### 字段说明

- **date**: 消费时间, 格式 `yyyy-mm-dd hh:mm:ss` 或 `yyyy-mm-dd`
- **item**: 消费内容/项目, 如果包含地点信息会格式化为 `项目(地点)`
- **category**: 消费类型/分类, 格式为 `具体分类-动作` (如 `服装-购买`, `餐饮-外卖`)
- **amount**: 消费金额, 使用 `Decimal` 类型保证精度, 支持负数表示支出

## Handoffs 模式

本项目实现了 LangChain Handoffs 模式的核心特性:

### 状态驱动行为

处理流程通过 `ProcessingStep` 枚举管理:

- `INITIAL`: 初始状态
- `IMAGE_ANALYSIS`: 图片分析中
- `EXTRACTION`: 内容提取中
- `COMPLETED`: 处理完成
- `ERROR`: 处理错误

### 工具函数

- `analyze_image()`: 分析图片并更新状态
- `extract_content()`: 提取结构化内容并更新状态

### 路由机制

`route()` 函数根据当前状态决定下一步操作, 实现状态驱动的流程控制.

## 配置说明

### 默认模型

- **本地模型**: `deepseek-ocr` (用于图片 OCR)
- **远程模型**: `deepseek-v3.1:671b-cloud` (用于信息提取)

### 自定义配置

可以通过环境变量或代码参数自定义模型和客户端配置.

## 输出格式

处理结果以 JSON 格式输出:

```json
{
    "current_step": "completed",
    "image_path": "/path/to/image.jpg",
    "input_text": null,
    "local_description": "提取的图片文本内容...",
    "extracted_content": [...],
    "consumptions": [
        {
            "date": "2026-02-03 12:22:51",
            "item": "衣服(大丰)",
            "category": "服装-购买",
            "amount": 45.00
        }
    ],
    "error": null,
    "metadata": {...}
}
```

## 开发

### 代码结构

- `handoffs_agent.py`: 核心实现, 包含 `HandoffsAgent` 类和状态管理
- `client_all.py`: Ollama 客户端配置
- `main.py`: 辅助工具函数

### 扩展

可以通过以下方式扩展功能:

1. 添加新的处理步骤到 `ProcessingStep` 枚举
2. 实现新的工具函数并更新路由逻辑
3. 自定义提取提示词以适配不同的数据格式

## 依赖

- `ollama>=0.6.0`: Ollama Python 客户端
- `pydantic>=2.0.0`: 数据验证和模型定义
- `dotenv>=0.9.9`: 环境变量管理
- `openai>=1.0.0`: OpenAI 兼容接口 (可选)

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request!

