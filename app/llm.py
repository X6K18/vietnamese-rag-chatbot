import ollama

def build_prompt(query, docs, history=None):
    context = "\n\n".join([
        f"[{i+1}] {doc['text']}\n(Nguồn: {doc['source']})"
        for i, doc in enumerate(docs)
    ])

    # Định dạng lịch sử hội thoại (nếu có)
    history_text = ""
    if history:
        history_text = "## LỊCH SỬ HỘI THOẠI\n"
        for msg in history:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"

    return f"""
Bạn là chatbot tiếng Việt thân thiện.

## QUY TẮC
- CHỈ dùng CONTEXT dưới đây và LỊCH SỬ HỘI THOẠI (nếu có) để trả lời.
- KHÔNG được bịa thông tin.
- Nếu không có thông tin trong CONTEXT, hãy dựa vào lịch sử hoặc nói "Tôi không tìm thấy thông tin phù hợp."

{history_text}
## CONTEXT
{context}

## CÂU HỎI HIỆN TẠI
{query}

## TRẢ LỜI (ngắn gọn, có thể trích dẫn [1], [2] nếu dùng context):
"""

def generate_answer(query, docs, history=None):
    prompt = build_prompt(query, docs, history=history)
    try:
        response = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        return response["message"]["content"]
    except Exception as e:
        return f"⚠️ Lỗi kết nối Ollama: {str(e)}"