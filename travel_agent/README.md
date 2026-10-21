# 智能旅行助手（hello-agents 1.3 节示例）

## 安装依赖

```bash
python3 -m venv .venv && .venv/bin/pip install requests tavily-python openai
```

## 运行

**纯本地验证**（不需要任何密钥，LLM 输出为脚本模拟，天气为真实数据）：

```bash
.venv/bin/python -W ignore agent.py --mock-llm
```

**本地模型（Ollama，无需密钥，已在本机验证）**：

```bash
ollama serve          # 另开一个终端保持运行；或 brew services start ollama
```

```bash
API_KEY=ollama BASE_URL=http://localhost:11434/v1 MODEL_ID=qwen2.5:3b .venv/bin/python -W ignore agent.py
```

**接入云端 LLM**（在你自己的终端里设置环境变量，不要把密钥写进文件或聊天）：

```bash
export API_KEY="你的密钥"
export BASE_URL="https://api.openai.com/v1"      # 或 Ollama: http://localhost:11434/v1
export MODEL_ID="gpt-4o-mini"                      # 或 Ollama 中的模型名
export TAVILY_API_KEY="你的 Tavily 密钥"            # 可选，缺省时景点工具返回本地模拟结果
.venv/bin/python -W ignore agent.py
```

## 与教程原文的差异

| 项目 | 教程原文 | 本实现 |
|---|---|---|
| 密钥与地址 | 写死在代码里的占位符 | 从环境变量读取，缺失时报错退出 |
| 无 Tavily 密钥 | 工具返回 `Error: ...` | 返回标注 `[MOCK]` 的本地模拟结果 |
| 无 LLM 凭据 | 无法运行 | `--mock-llm` 用脚本化三轮输出驱动循环 |
| 网络请求 | 无超时 | `requests.get` 加 15 秒超时 |
| 参数解析正则 | `(\w+)="([^"]*)"` 等价 | 相同 |
