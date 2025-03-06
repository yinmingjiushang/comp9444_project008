# langchain 调用 openai
#  未成功目前没开openai api


from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")