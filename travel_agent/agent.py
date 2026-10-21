"""
hello-agents 第一章 1.3 节：智能旅行助手（Thought-Action-Observation 循环）

与教程原文的差异：
  1. 所有密钥和地址从环境变量读取，不写进代码。
  2. 没有 TAVILY_API_KEY 时，get_attraction 退化为本地模拟结果（会明确标注 [MOCK]）。
  3. 增加 --mock-llm 参数：用脚本化的三轮模型输出代替真实 LLM，
     用于在没有 LLM 凭据时验证整个循环逻辑。

运行方式：
  真实 LLM：  export API_KEY=... BASE_URL=... MODEL_ID=... [TAVILY_API_KEY=...]
             python agent.py
  纯本地验证： python agent.py --mock-llm
"""

import os
import re
import sys

import requests

# ---------------------------------------------------------------------------
# (1) 指令模板：智能体的"说明书"，作为 system_prompt 传给 LLM
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT = """
You are an intelligent travel assistant. Your task is to analyze user requests and use available tools to solve problems.

# Available Tools:
- `get_weather(city: str)`: Query real-time weather for a specified city.
- `get_attraction(city: str, weather: str)`: Search for recommended tourist attractions based on city and weather.

# Output Format Requirements:
Each response must contain exactly two lines, in this order:

Thought: <your reasoning and plan for the next step>
Action: <exactly one of the two forms below>

The Action line must be either a tool call or Finish. Examples of valid Action lines:

Action: get_weather(city="Beijing")
Action: get_attraction(city="Beijing", weather="Sunny")
Action: Finish[Today in Beijing it is sunny and 20 degrees, so I recommend the Summer Palace.]

# Important Notes:
- Always start the second line with the literal word "Action:" followed by a space.
- Never write "Finish:" or "Call a tool:" -- the only valid prefixes are "Thought:" and "Action:".
- Output only one Thought line and one Action line each time; do not add anything else.
- Action must be on a single line.
- Once you have enough information to answer the user's question, your Action must be Finish[...] with the full final answer inside the brackets.

Let's begin!
"""


# ---------------------------------------------------------------------------
# (2) 工具一：查真实天气（wttr.in）
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Query real weather information by calling the wttr.in API."""
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]
        return f"{city} current weather: {weather_desc}, temperature {temp_c} degrees Celsius"
    except requests.exceptions.RequestException as e:
        return f"Error: Network problem encountered when querying weather - {e}"
    except (KeyError, IndexError, ValueError) as e:
        return f"Error: Failed to parse weather data, city name may be invalid - {e}"


# ---------------------------------------------------------------------------
# (3) 工具二：搜索并推荐景点（Tavily；无密钥时使用本地模拟）
# ---------------------------------------------------------------------------
def get_attraction(city: str, weather: str) -> str:
    """Based on city and weather, use Tavily Search API to return attraction recommendations."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        # 教程原文在这里直接返回错误；这里改为本地模拟，便于无密钥时跑通循环
        return (
            f"[MOCK] Based on search, the most worthwhile attractions in {city} "
            f"during '{weather}' weather are the Summer Palace (boating on Kunming Lake) "
            f"and the Temple of Heaven Park (open-air royal architecture)."
        )

    from tavily import TavilyClient

    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' most worthwhile tourist attractions and reasons in '{weather}' weather"
    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        if response.get("answer"):
            return response["answer"]
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        if not formatted_results:
            return "Sorry, no relevant tourist attraction recommendations found."
        return "Based on search, found the following information for you:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"Error: Problem occurred when executing Tavily search - {e}"


# 工具注册表：主循环按名字分发
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


# ---------------------------------------------------------------------------
# 1.3.2 连接大语言模型：兼容 OpenAI 接口的通用客户端
# ---------------------------------------------------------------------------
class OpenAICompatibleClient:
    """A client for calling any LLM service compatible with the OpenAI interface."""

    def __init__(self, model: str, api_key: str, base_url: str):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        print("Calling large language model...")
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=False
            )
            answer = response.choices[0].message.content
            print("Large language model responded successfully.")
            return answer
        except Exception as e:
            print(f"Error occurred when calling LLM API: {e}")
            return "Error: Error occurred when calling language model service."


class MockLLMClient:
    """脚本化的三轮输出，用于无 LLM 凭据时验证循环逻辑。"""

    def __init__(self):
        self.turn = 0

    def generate(self, prompt: str, system_prompt: str) -> str:
        print("[MOCK] Calling large language model...")
        last_obs = ""
        m = re.findall(r"Observation: (.*)", prompt)
        if m:
            last_obs = m[-1]
        self.turn += 1
        if self.turn == 1:
            out = (
                "Thought: First need to get Beijing's weather today, then recommend tourist attractions based on the weather.\n"
                'Action: get_weather(city="Beijing")'
            )
        elif self.turn == 2:
            # 从上一轮观察里提取天气描述，模拟模型"读懂"了 Observation
            wm = re.search(r"weather: ([^,]+),", last_obs)
            weather = wm.group(1) if wm else "Sunny"
            out = (
                f"Thought: Now I know Beijing's weather today is {weather}, I can recommend a suitable tourist attraction.\n"
                f'Action: get_attraction(city="Beijing", weather="{weather}")'
            )
        else:
            out = (
                "Thought: I have obtained attraction suggestions, now I can provide a satisfactory response to the user.\n"
                f"Action: Finish[Based on today's weather and the search result: {last_obs}]"
            )
        print("[MOCK] Large language model responded successfully.")
        return out


# ---------------------------------------------------------------------------
# 解析器（Parser）：把模型的自然语言输出翻译成可执行的指令
# 教程原文只用一条正则 `Action: (.*)`；这里做了容错，能接住小模型常见的格式偏差。
# ---------------------------------------------------------------------------
def parse_action(llm_output: str):
    """返回 Action 文本；找不到时返回 None。"""
    # 标准格式：Action: xxx
    m = re.search(r"^\s*Action:\s*(.+)$", llm_output, re.MULTILINE | re.DOTALL)
    if m:
        action = m.group(1).strip()
        # 容错 1：去掉模型从格式说明里照搬的标签前缀
        action = re.sub(r"^(Call a tool|Finish task)\s*:\s*", "", action, flags=re.IGNORECASE)
        return action
    # 容错 2：模型把 "Action: Finish[...]" 写成了 "Finish: ..." 或 "Finish[...]"
    m = re.search(r"^\s*Finish\s*[:\[]\s*(.+)$", llm_output, re.MULTILINE | re.DOTALL)
    if m:
        answer = m.group(1).strip().rstrip("]")
        return f"Finish[{answer}]"
    return None


def parse_finish(action_str: str):
    """如果是 Finish 动作，返回最终答案；否则返回 None。"""
    m = re.match(r"Finish\s*[\[\(:]?\s*(.*?)\s*[\]\)]?\s*$", action_str, re.DOTALL)
    if m and action_str.lower().startswith("finish"):
        return m.group(1).strip()
    return None


def parse_tool_call(action_str: str):
    """解析 name(arg="value", ...)，返回 (name, kwargs)；解析失败返回 None。"""
    m = re.match(r"\s*(\w+)\s*\((.*)\)\s*$", action_str, re.DOTALL)
    if not m:
        return None
    name, args_str = m.group(1), m.group(2)
    # 同时接受双引号和单引号
    # re.findall 对未匹配的分组返回空串，所以取非空的那个
    kwargs = {k: v1 or v2
              for k, v1, v2 in re.findall(r"(\w+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')", args_str)}
    return name, kwargs


# ---------------------------------------------------------------------------
# 1.3.3 执行行动循环
# ---------------------------------------------------------------------------
def build_llm():
    if "--mock-llm" in sys.argv:
        print("[MOCK] Using scripted LLM output (no real model call).")
        return MockLLMClient()
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    model_id = os.environ.get("MODEL_ID")
    missing = [n for n, v in (("API_KEY", api_key), ("BASE_URL", base_url), ("MODEL_ID", model_id)) if not v]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}. Use --mock-llm to run without a real LLM.")
        sys.exit(2)
    return OpenAICompatibleClient(model=model_id, api_key=api_key, base_url=base_url)


def main():
    llm = build_llm()

    user_prompt = (
        "Hello, please help me check today's weather in Beijing, "
        "and then recommend a suitable tourist attraction based on the weather."
    )
    prompt_history = [f"User request: {user_prompt}"]
    print(f"User input: {user_prompt}\n" + "=" * 40)

    for i in range(5):  # 最大轮数：防止死循环
        print(f"--- Loop {i + 1} ---\n")

        # 3.1 拼接历史作为完整 prompt
        full_prompt = "\n".join(prompt_history)

        # 3.2 调用 LLM 思考，并截断多余的 Thought-Action 对
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        match = re.search(
            r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
            llm_output,
            re.DOTALL,
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("Truncated extra Thought-Action pairs")
        print(f"Model output:\n{llm_output}\n")
        prompt_history.append(llm_output)

        # 3.3 解析并执行 Action
        action_str = parse_action(llm_output)
        if action_str is None:
            observation = (
                "Error: No action found. Your reply must contain a line starting with "
                '"Action: " followed by either a tool call like get_weather(city="Beijing") '
                "or Finish[final answer]."
            )
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        final_answer = parse_finish(action_str)
        if final_answer is not None:
            print(f"Task completed, final answer: {final_answer}")
            break

        call = parse_tool_call(action_str)
        if call is None:
            observation = f"Error: Could not parse tool call from '{action_str}'. Use function_name(arg=\"value\")."
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue
        tool_name, kwargs = call

        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"Error: Undefined tool '{tool_name}'"

        # 3.4 记录观察结果
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)


if __name__ == "__main__":
    main()
