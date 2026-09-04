import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rag_engine import RAGFlowEngine

load_dotenv()

# 严格对齐曲阜师范大学《学生手册》实际收录章节规范
test_cases = [
    {
        "question": "考试违纪或作弊会受到什么处分？",
        "ground_truth": "根据违规情节轻重给予相应纪律处分：一般作弊给予严重警告；较重作弊给予记过或留校察看；严重作弊、替考或使用通讯工具等给予开除学籍处分；该门课程成绩记为零分。"
    },
    {
        "question": "国家奖学金的评选条件是什么？",
        "ground_truth": "基本申请条件包括：热爱社会主义祖国，拥护中国共产党的领导；遵守宪法和法律，遵守学校规章制度；诚实守信，道德品质优良；在校期间学习成绩优异，社会实践、创新能力、综合素质等方面特别突出；为在校生中二年级以上（含二年级）的学生。"
    },
    {
        "question": "学生因故不能参加考试时如何申请缓考？",
        "ground_truth": "学生因故不能参加期末考核，除不可抗力原因外，必须提前一天提交缓考申请，经学院核实、开课单位批准后允许缓考。"
    }
]

engine = RAGFlowEngine()
eval_records = []

for case in test_cases:
    q = case["question"]
    res = engine.chat(q)
    ans = res.get("answer", "") or "暂无明确回答"

    raw_refs = res.get("references", [])
    valid_chunks = [ref.get("content", "").strip() for ref in raw_refs if ref.get("content", "").strip()]
    contexts_text = "\n\n".join(valid_chunks) if valid_chunks else "未检索到规章切片文本。"

    eval_records.append({
        "question": q,
        "answer": ans,
        "contexts": contexts_text,
        "ground_truth": case["ground_truth"]
    })

judge_llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)

results = []
print("开始执行规章文本级自动化评估...")

for idx, item in enumerate(eval_records, 1):
    prompt = f"""你是一名严格的 RAG 评估裁判专家。请仔细对比给定的规章数据，从三个维度分别给出 0.0 到 1.0 之间的分数（保留两位小数），并给出判定理由。

【用户提问】: {item['question']}
【标准答案 (Ground Truth)】: {item['ground_truth']}
【检索到的实际上下文 (Contexts)】: {item['contexts']}
【系统生成回答 (Answer)】: {item['answer']}

评估维度：
1. faithfulness (忠实度): 系统回答的内容是否能从【检索到的实际上下文】中找到依据，不得脱离检索事实或捏造规则。
2. context_recall (召回率): 【检索到的实际上下文】是否覆盖了【标准答案】里的规章核心条件。
3. answer_correctness (准确度): 系统的回答与【标准答案】在制度事实层面的一致性。

请严格返回如下 JSON 格式，不要附带额外标记：
{{
    "faithfulness": 0.95,
    "context_recall": 0.90,
    "answer_correctness": 0.92,
    "reason": "简短判定理由"
}}
"""
    response = judge_llm.invoke(prompt).content.strip()
    cleaned_json = response.replace("```json", "").replace("```", "").strip()

    try:
        score_data = json.loads(cleaned_json)
    except Exception:
        score_data = {
            "faithfulness": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.0,
            "reason": response
        }

    results.append({
        "question": item["question"],
        "retrieved_contexts": item["contexts"],
        "answer": item["answer"],
        "faithfulness": score_data.get("faithfulness", 0.0),
        "context_recall": score_data.get("context_recall", 0.0),
        "answer_correctness": score_data.get("answer_correctness", 0.0),
        "reason": score_data.get("reason", "")
    })
    print(f"评测进度: [{idx}/{len(eval_records)}]")

df = pd.DataFrame(results)

print("\n===== 评估结果汇总 =====")
print(df[["question", "faithfulness", "context_recall", "answer_correctness"]])

print("\n===== 详细切片与裁判判定 =====")
for idx, row in df.iterrows():
    print("--------------------------------------------------")
    print(f"【问题 {idx + 1}】: {row['question']}")
    print(
        f"【分数】 -> 忠实度: {row['faithfulness']} | 召回率: {row['context_recall']} | 准确度: {row['answer_correctness']}")
    print(f"【评语】: {row['reason']}")

output_file = "ragas_evaluation_report.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n完整报告已保存至: {output_file}")