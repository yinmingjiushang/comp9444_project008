import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# 定义数据路径和持久化数据库路径
DATA_PATH = "../data/ScienceQA/data/scienceqa/problems.json"
PERSIST_DIR = "../data/chroma_db"


# 加载 ScienceQA problems.json 数据集
def load_scienceqa_problems(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 初始化 SentenceTransformer 嵌入模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 初始化Chroma（持久化存储）
client = chromadb.PersistentClient(path=PERSIST_DIR)

# 创建或获取Chroma向量数据库集合
# if "scienceqa" in [col.name for col in client.list_collections()]:
#     collection = client.get_collection("scienceqa")
# else:
#     collection = client.create_collection("scienceqa")
# 创建或获取Chroma向量数据库集合
existing_collections = client.list_collections()
if "scienceqa" in existing_collections:
    collection = client.get_collection("scienceqa")
else:
    collection = client.create_collection("scienceqa")


# 加载ScienceQA数据
def load_and_process_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts, ids = [], []

    for qid, item in data.items():
        question = item['question']
        choices = item['choices']
        answer_idx = item['answer']
        answer = choices[answer_idx]
        hint = item.get('hint', '')
        subject = item.get('subject', '')
        grade = item.get('grade', '')

        # 将所有文本拼接成更丰富的上下文，更好地捕捉语义
        text = f"Subject: {subject}. Grade: {grade}. Question: {question} Choices: {choices}. Answer: {answer}. Hint: {hint}"

        texts.append(text)
        ids.append(qid)

    # 生成嵌入向量
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    # 导入数据到Chroma
    collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=ids
    )

    print("数据成功导入到本地持久化的Chroma数据库。")



