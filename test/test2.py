# langchain 调用 deepseek
# 自己写，在trace有报错

import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks import LangChainTracer
from langsmith import Client

# 1. 加载 .env 文件
load_dotenv()

# 2. 获取 API Key
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

# 3. 确保 API Key 正确加载
if not deepseek_api_key or not langsmith_api_key:
    raise ValueError("API Key 未正确加载，请检查 .env 文件！")

# 4. 初始化 LangSmith 监控
client = Client(api_key=langsmith_api_key)



# 5. 创建监控记录
# run = client.create_run(
#     run_type="llm",
#     name="deepseek_langsmith_monitoring",
#     inputs={"model": "deepseek-chat"},
#     config={"tags": ["monitoring", "deepseek"]}
# )
run = client.create_run(
    run_type="llm",
    name="deepseek_langsmith_monitoring",
    inputs={"model": "deepseek-chat"},
    config={"tags": ["monitoring", "deepseek"]},
    project_name="comp9444_pro008"
)

if not run:
    raise RuntimeError("LangSmith create_run() 失败，请检查 API Key 和参数！")

print(f"LangSmith Run ID: {run.get('id', '创建失败')}")

# 6. 初始化 LangChain 追踪器
tracer = LangChainTracer()

# 7. 配置 DeepSeek 模型
llm = ChatDeepSeek(
    model="deepseek-chat",  # 选择合适的 DeepSeek 模型
    temperature=0.7,
    max_tokens=512,
    timeout=10,
    api_key=deepseek_api_key
)

# 8. 构造对话消息
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Translate 'I love programming' to chinese.")
]

# 9. 调用 DeepSeek 并监控请求
response = llm.invoke(messages, callbacks=[tracer])

# 10. 记录结果到 LangSmith
client.update_run(
    run["id"],
    outputs={"response": response},
    status="completed"
)

print("DeepSeek Response:", response)
