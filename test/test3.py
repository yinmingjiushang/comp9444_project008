# langchain 调用 deepseek api + langsmith trace
# syccess
# Todo: 加入评估分析


import os
import uuid
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage, SystemMessage
from langsmith import Client
from langchain.callbacks.tracers.langchain import LangChainTracer


# --------------------- 初始化阶段 ---------------------
def init_environment():
    """环境初始化"""
    load_dotenv()
    # 验证关键配置
    required_keys = ["DEEPSEEK_API_KEY", "LANGSMITH_API_KEY"]
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        raise ValueError(f"缺少环境变量: {', '.join(missing)}")

    # LangSmith 环境变量（V2 追踪）
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "DeepSeek_Monitoring"  # 你想追踪的项目名


# --------------------- LangSmith 配置 ---------------------
class LangSmithManager:
    def __init__(self):
        """初始化 LangSmith Client"""
        self.client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

    def setup_project(self):
        """项目初始化（幂等操作）"""
        try:
            self.client.create_project(
                project_name=os.environ["LANGCHAIN_PROJECT"],
                description="DeepSeek API 调用监控"
            )
        except Exception as e:
            # 如果项目已存在，会抛出 "already exists" 异常，可忽略
            if "already exists" not in str(e):
                raise RuntimeError(f"项目初始化失败: {str(e)}")

    def create_monitored_run(self, input_data: dict) -> str:
        """
        创建带监控的运行记录，并返回自定义 run_id。
        LangSmith >= 0.0.9中，create_run 不再返回 Run 对象，所以需手动生成 ID。
        """
        try:
            run_id = str(uuid.uuid4())  # 生成唯一 ID
            self.client.create_run(
                id=run_id,
                name="deepseek_api_call",
                run_type="llm",
                inputs=input_data,
                project_name=os.environ["LANGCHAIN_PROJECT"],
                config={
                    "tags": ["prod", "deepseek-v1"],
                    "metadata": {"env": "production"}
                }
            )
            return run_id
        except Exception as e:
            raise RuntimeError(f"监控记录创建失败: {str(e)}") from e


# --------------------- DeepSeek 集成 ---------------------
class DeepSeekClient:
    def __init__(self):
        """
        初始化 DeepSeek LLM。

        注意：为避免 “multiple values for 'callbacks'” 报错，
        不再在 invoke(...) 中手动传递 callbacks=[...] 参数。
        LangChain V2 tracing 可以通过环境变量自动生效。
        """
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        # 你也可以保留这一行，用于其他自定义情况，但**不要**在 invoke 时重复传入 callbacks
        self.tracer = LangChainTracer()

    def generate(self, messages: list) -> str:
        """执行生成 (自动追踪由 V2 环境变量负责)"""
        try:
            # 不再手动传 callbacks=[self.tracer]
            # 仅调用 self.llm.invoke(messages) 即可
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"API 调用失败: {str(e)}") from e


# --------------------- 使用示例 ---------------------
if __name__ == "__main__":
    # 1. 初始化环境
    init_environment()

    # 2. 配置 LangSmith
    langsmith = LangSmithManager()
    langsmith.setup_project()

    # 3. 准备输入
    input_data = {
        "instruction": "翻译任务",
        "text": "I love artificial intelligence"
    }

    # 4. 创建监控记录，返回自定义 run_id
    run_id = langsmith.create_monitored_run(input_data)
    print(f"监控启动 | Run ID: {run_id}")

    # 5. 初始化 DeepSeek
    deepseek = DeepSeekClient()

    # 6. 构造对话
    messages = [
        SystemMessage(content="你是一名专业翻译"),
        HumanMessage(content=f"翻译这句话: {input_data['text']}")
    ]

    # 7. 执行 LLM 调用
    try:
        result = deepseek.generate(messages)
        print(f"翻译结果: {result}")

        # 8. 追踪成功结果
        langsmith.client.update_run(
            run_id,
            outputs={"translation": result},
            status="completed"
        )
    except Exception as e:
        # 追踪失败状态
        langsmith.client.update_run(
            run_id,
            outputs={"error": str(e)},
            status="failed"
        )
        raise
