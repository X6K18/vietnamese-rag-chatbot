import ollama

def test_qwen_connection():
    model_name = 'qwen2.5:1.5b'
    prompt = "Chào bạn, bạn có nghe rõ tôi nói không? Hãy giới thiệu ngắn gọn về bản thân bằng tiếng Việt."

    print(f"--- Đang kết nối với {model_name} ---")
    
    try:
        # Gửi yêu cầu đến Ollama
        stream = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
        )

        print("Phản hồi từ AI: ", end="", flush=True)
        
        # In kết quả dạng stream
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
        
        print("\n\n--- Kiểm tra thành công! ---")

    except ollama.ResponseError as e:
        print(f"\nLỗi: Không tìm thấy model '{model_name}'.")
        print("Hãy đảm bảo bạn đã chạy lệnh 'ollama pull qwen2.5:1.5b' trong CMD.")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi kết nối: {e}")

if __name__ == "__main__":
    test_qwen_connection()