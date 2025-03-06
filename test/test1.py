# langchain 调用 openai
#  未成功目前没开openai api
LANGSMITH_TRACING= True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="comp9444_pro008"
OPENAI_API_KEY="<your-openai-api-key>"


from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")