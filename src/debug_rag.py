import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
app_id = os.getenv("BAILIAN_APP_ID")

if not app_id:
    print("未在 .env 中找到 BAILIAN_APP_ID，当前走的是常规模型对话接口，未走应用流检索！")
else:
    url = f"https://dashscope.aliyuncs.com/api/v1/apps/{app_id}/completion"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {"prompt": "国家奖学金的评选条件是什么？"},
        "parameters": {}
    }
    resp = requests.post(url, headers=headers, json=payload)
    print("===== 百炼返回原始完整数据 =====")
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))