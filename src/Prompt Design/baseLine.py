import json
import sys
import requests


# ─────────────────────────────────────────────
# Ollama API Client
# ─────────────────────────────────────────────
class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.base_url = base_url
        self.model = model

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def chat(self, messages: list[dict], stream: bool = True) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=stream,
            timeout=120,
        )
        response.raise_for_status()

        if stream:
            full_response = ""
            print("\n\033[36mAssistant:\033[0m ", end="", flush=True)
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    print(token, end="", flush=True)
                    full_response += token
                    if chunk.get("done"):
                        break
            print()
            return full_response
        else:
            return response.json()["message"]["content"]


# ─────────────────────────────────────────────
# Conversation Manager
# ─────────────────────────────────────────────
class RawAssistant:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.client = OllamaClient(base_url=ollama_url, model=model)
        self.conversation_history: list[dict] = []

    def reset_conversation(self):
        self.conversation_history = []
        print("\n\033[33m[Conversation history cleared]\033[0m\n")

    def chat(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})

        # احتفظ بآخر 10 رسائل بس
        recent_history = self.conversation_history[-10:]

        assistant_response = self.client.chat(recent_history, stream=True)
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response


# ─────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────
def main():
    model = "llama3.2:3b"
    assistant = RawAssistant(model=model)

    print(f"\n\033[36m[Raw Ollama Chat — {model} — No System Prompt]\033[0m")
    print("Commands: /reset  /exit\n")

    if not assistant.client.check_connection():
        print(
            "\033[31m[ERROR] Cannot connect to Ollama at http://localhost:11434\033[0m\n"
            "Make sure Ollama is running:  ollama serve\n"
            f"And the model is pulled:      ollama pull {model}\n"
        )
        sys.exit(1)

    print(f"\033[32m[OK] Connected to Ollama. Model: {model}\033[0m\n")

    while True:
        try:
            user_input = input("\033[33mYou:\033[0m ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print("\nGoodbye.")
                break
            elif user_input.lower() == "/reset":
                assistant.reset_conversation()
                continue

            assistant.chat(user_input)
            print()

        except KeyboardInterrupt:
            print("\n\nExiting.")
            break
        except requests.exceptions.ConnectionError:
            print("\033[31m[ERROR] Lost connection to Ollama.\033[0m")
        except requests.exceptions.Timeout:
            print("\033[31m[ERROR] Request timed out.\033[0m")
        except Exception as e:
            print(f"\033[31m[ERROR] {e}\033[0m")


if __name__ == "__main__":
    main()