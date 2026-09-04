import os
import random
import gradio as gr
from dotenv import load_dotenv
from rag_engine import RAGFlowEngine

load_dotenv()
try:
    rag_engine = RAGFlowEngine()
except Exception as e:
    rag_engine = None

QUESTION_BANK = [
    "考试违纪或作弊会受到什么处分？",
    "国家奖学金的评选条件是什么？",
    "学生请假制度是怎么规定的？",
    "学业成绩不及格需要怎样重修？",
    "本科生申请转专业有哪些要求与流程？",
    "综合素质测评的加分细则有哪些？",
    "学生宿舍用电与违章电器管理规定是什么？",
    "达到什么条件可以提前毕业或申请辅修学位？",
    "受到纪律处分后如何申请解除或撤销？",
    "考勤旷课累计达到多少学时会受到纪律处分？"
]


def get_random_three_questions():
    return random.sample(QUESTION_BANK, 3)


def user_predict(message, history):
    if not message.strip():
        return "", history, "请输入有效的问题。"

    if rag_engine is None:
        bot_reply = "系统错误：RAG 引擎未成功初始化，请检查 .env 配置。"
        source_info = "无可用数据源"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_reply})
        return "", history, source_info

    res = rag_engine.chat(message)
    bot_reply = res.get("answer", "未获取到有效回答。")

    references = res.get("references", [])
    source_lines = []
    for idx, ref in enumerate(references, 1):
        doc_name = ref.get("doc_name", "未知文档")
        similarity = ref.get("similarity", "中")
        content = ref.get("content", "")
        source_lines.append(f"[{idx}] {doc_name} (匹配度: {similarity})\n摘要: {content}\n")

    source_info = "\n--------------------\n".join(source_lines) if source_lines else "暂无匹配的文档片段。"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_reply})

    return "", history, source_info


def refresh_questions():
    q1, q2, q3 = get_random_three_questions()
    return gr.update(value=q1), gr.update(value=q2), gr.update(value=q3)


modern_dark_css = """
:root, body, .gradio-container {
    --background-fill-primary: #1e293b !important;
    --background-fill-secondary: #0f172a !important;
    --block-background-fill: #1e293b !important;
    --panel-background-fill: #1e293b !important;
    --body-background-fill: #0b0f19 !important;
    --border-color-primary: rgba(148, 163, 184, 0.2) !important;
    --body-text-color: #ffffff !important;
    --block-label-text-color: #fef08a !important;
    background: radial-gradient(circle at 20% 20%, #1e1b4b 0%, #0b0f19 60%, #030712 100%) !important;
    color: #ffffff !important;
}

header {
    display: none !important;
}

* {
    color: #ffffff !important;
}

.chatbot,
[data-testid="chatbot"],
.chatbot > div,
.wrap,
.panel,
.block {
    background-color: #1e293b !important;
    border-color: rgba(148, 163, 184, 0.2) !important;
}

.message-row.user-row .message-bubble,
.message.user,
[data-testid="user"] {
    background-color: #334155 !important;
    border: 1px solid rgba(251, 191, 36, 0.3) !important;
}
.message-row.user-row .message-bubble *,
.message.user *,
[data-testid="user"] * {
    color: #fef08a !important;
}

.message-row.bot-row .message-bubble,
.message.bot,
[data-testid="bot"] {
    background-color: #1e293b !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
}
.message-row.bot-row .message-bubble *,
.message.bot *,
[data-testid="bot"] * {
    color: #f8fafc !important;
}

textarea, input, .gr-input {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
}
textarea::placeholder, input::placeholder {
    color: #94a3b8 !important;
}

button.primary {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 50%, #b45309 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

button.secondary, .gr-button {
    background: #334155 !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    color: #fef3c7 !important;
}
button.secondary:hover, .gr-button:hover {
    background: #475569 !important;
    border-color: #f59e0b !important;
}

h1, h2, h3 {
    color: #ffffff !important;
}
.recommend-header * {
    color: #ffffff !important;
    font-weight: 600 !important;
}
"""

initial_q1, initial_q2, initial_q3 = get_random_three_questions()

with gr.Blocks(title="曲阜师范大学学生手册智能问答系统") as demo:
    gr.Markdown("# 🎓 曲阜师范大学《学生手册》智能问答系统")
    gr.Markdown("基于阿里云百炼大模型与 RAG 引擎，为您提供精准、严谨的校园规章制度解答。")

    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="智能咨询助手",
                height=500
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="请输入关于学籍、违纪、评优等规定...",
                    show_label=False,
                    scale=5,
                    container=False
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)

            with gr.Row(elem_classes=["recommend-header"]):
                gr.Markdown("**💡 热门咨询推荐（点击可直接发送）：**")
                refresh_btn = gr.Button("🔄 换一批", size="sm", variant="secondary", scale=0)

            with gr.Row():
                btn1 = gr.Button(initial_q1, size="sm")
                btn2 = gr.Button(initial_q2, size="sm")
                btn3 = gr.Button(initial_q3, size="sm")

            clear_btn = gr.Button("清空当前对话", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("### 📖 知识库引用与溯源")
            source_output = gr.Textbox(
                label="召回片段与匹配详情",
                value="暂无检索记录，请在左侧提问...",
                lines=21,
                interactive=False
            )

    submit_btn.click(user_predict, inputs=[msg, chatbot], outputs=[msg, chatbot, source_output])
    msg.submit(user_predict, inputs=[msg, chatbot], outputs=[msg, chatbot, source_output])

    btn1.click(lambda b_val: (b_val, []), inputs=[btn1], outputs=[msg, chatbot]).then(
        user_predict, inputs=[msg, chatbot], outputs=[msg, chatbot, source_output]
    )
    btn2.click(lambda b_val: (b_val, []), inputs=[btn2], outputs=[msg, chatbot]).then(
        user_predict, inputs=[msg, chatbot], outputs=[msg, chatbot, source_output]
    )
    btn3.click(lambda b_val: (b_val, []), inputs=[btn3], outputs=[msg, chatbot]).then(
        user_predict, inputs=[msg, chatbot], outputs=[msg, chatbot, source_output]
    )

    refresh_btn.click(refresh_questions, outputs=[btn1, btn2, btn3])

    clear_btn.click(lambda: ([], "暂无检索记录，请在左侧提问..."), outputs=[chatbot, source_output])

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7899,
        share=False,
        css=modern_dark_css
    )