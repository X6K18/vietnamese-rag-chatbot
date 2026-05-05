import streamlit as st
import torch
import joblib
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import predict_with_rag_llm
from rag import RAG

# ---------- CẤU HÌNH ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/phobert_model")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models/phobert_tokenizer")
LABEL_PATH = os.path.join(BASE_DIR, "models/label_encoder.joblib")
INDEX_PATH = os.path.join(BASE_DIR, "data/faiss.index")
DATA_PATH = os.path.join(BASE_DIR, "data/data.pkl")

st.set_page_config(page_title="Vietnamese RAG Chatbot", page_icon="🧠", layout="wide")

# ---------- HÀM XÁC THỰC ĐƠN GIẢN ----------
# Có thể thay bằng database thật
VALID_USERS = {
    "phantrongnguyen0618@gmail.com": "123",
    "smoking": "456"
}

def check_password(username, password):
    return VALID_USERS.get(username) == password

# ---------- TẢI MODELS (CACHE) ----------
@st.cache_resource
def load_all():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    label_encoder = joblib.load(LABEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=len(label_encoder.classes_)
    )
    rag = RAG(INDEX_PATH, DATA_PATH)
    return tokenizer, model, label_encoder, rag

# Chỉ load models một lần (không phụ thuộc login)
tokenizer, model, label_encoder, rag = load_all()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---------- KHỞI TẠO SESSION STATE ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.messages = []

# ---------- GIAO DIỆN ĐĂNG NHẬP ----------
def login_page():
    st.title("🔐 Đăng nhập")
    with st.form("login_form"):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submitted = st.form_submit_button("Đăng nhập")
        if submitted:
            if check_password(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.messages = []  # Xoá lịch sử cũ
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu")

def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.messages = []
    st.rerun()

# ---------- GIAO DIỆN CHAT CHÍNH ----------
def chat_interface():
    # Sidebar
    with st.sidebar:
        st.markdown(f"👤 **Người dùng:** `{st.session_state.username}`")
        if st.button("🚪 Đăng xuất"):
            logout()
        st.markdown("---")
        st.title("💬 Tuỳ chỉnh")
        if st.button("🗑️ Xoá lịch sử chat"):
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")
        st.info("Bot sẽ phân loại chủ đề, truy xuất tài liệu liên quan và trả lời dựa trên ngữ cảnh hội thoại.")

    # Hiển thị lịch sử
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Xem nguồn"):
                    for i, doc in enumerate(msg["sources"]):
                        st.markdown(f"**{i+1}. {doc['title']}**  \n{doc['url']}")

    # Ô nhập tin nhắn
    if query := st.chat_input("Nhập câu hỏi của bạn..."):
        # Thêm tin nhắn người dùng
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Gọi RAG + LLM (có truyền lịch sử)
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                label, answer, docs = predict_with_rag_llm(
                    query, tokenizer, model, label_encoder, device, rag,
                    history=st.session_state.messages[:-1]  # lịch sử trước câu hỏi hiện tại
                )
            st.markdown(f"**📂 Chủ đề:** `{label}`")
            st.markdown(answer)
            if docs:
                with st.expander("📚 Nguồn tham khảo"):
                    for i, d in enumerate(docs):
                        st.markdown(f"**{i+1}. {d['title']}**  \n{d['url']}")

        # Lưu phản hồi assistant
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": docs
        })

# ---------- ĐIỀU HƯỚNG ----------
if st.session_state.authenticated:
    chat_interface()
else:
    login_page()