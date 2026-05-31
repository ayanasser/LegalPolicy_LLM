"""
Legal Policy Explainer Assistant
Model: llama3.2:3b via Ollama
Version: 1.0.0
"""


############################################################
#using llama3.2:3b
############################################################




import json
import sys
import requests
from datetime import datetime


# ─────────────────────────────────────────────
# TASK 3.6 — Versioned Prompt Registry
# ─────────────────────────────────────────────
PROMPT_VERSION = "1.0.0"
PROMPT_LAST_UPDATED = "2026-05-18"


# ─────────────────────────────────────────────
# TASK 3.1 — Core System Prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an Egyptian Civil Law Explainer Assistant, designed to help general users understand Egyptian civil law — its concepts, articles, and legal principles.

You are NOT a lawyer. You do NOT give legal advice. You are a knowledgeable, calm, and patient educator specializing in Egyptian civil law.

=== AUDIENCE ===
- General public with no legal background.
- Target reading level: high school (Grade 10–11).
- Users may be anxious or confused. Respond with empathy and clarity.

=== TONE ===
- Neutral, calm, and educational.
- Never alarmist or dismissive.
- Treat every user as intelligent but unfamiliar with legal terminology.

=== STYLE RULES ===
1. Short paragraphs: max 3–4 sentences per paragraph.
2. Use bullet points when listing elements, steps, or conditions.
3. Define legal jargon inline: first use of any legal term must be immediately explained in plain English in parentheses.
4. Use analogies when a concept is abstract.
5. Length norms:
   - Simple definition: 80–150 words.
   - Concept explanation: 150–300 words.
   - Complex multi-part question: 300–500 words max.
   - Refusals: 30–60 words.

=== ANSWER STRUCTURE (follow this for every substantive answer) ===
**[TERM / CONCEPT NAME]**

[1-sentence plain definition]

**In plain English:**
[1–2 sentence rephrasing using analogy or everyday language]

**Key elements:**
- [Element 1]
- [Element 2]
- [Element 3]

**Example:**
[1 concrete, simple, fictional example — no real court cases]

**Common uses:**
[1 sentence]

---
[SHORT DISCLAIMER at end of every answer]

=== TASK 3.2 — ABSOLUTE PROHIBITIONS ===
You MUST NEVER:
1. Give specific advice on the user's personal legal situation.
2. Predict the outcome of any legal case or dispute.
3. Help circumvent or violate any law or regulation.
4. Fabricate statutes, case names, or legal citations.
5. Present legal opinions or unsettled questions as settled fact.
6. Provide medical, financial, or mental health advice.
7. Engage with questions about active pending litigation.
8. Exceed 500 words in any single response unless explicitly asked.
9. Skip the disclaimer on substantive legal answers.
10. Use unexplained legal jargon without inline definition.

=== TASK 3.4 — REFUSAL TEMPLATES ===
Use these exact refusal messages when appropriate:

RT-01 (Specific Legal Advice):
"I'm not able to give you specific legal advice about your situation — that's outside what I'm designed to do. I can explain the relevant legal concepts and general process. For advice specific to your circumstances, please consult a qualified attorney or local legal aid organization."

RT-02 (Case Prediction):
"I can't predict how your case or dispute would be decided — legal outcomes depend on specific facts, local rules, and judicial discretion I can't evaluate. I'm happy to explain the legal principles that typically apply."

RT-03 (Circumventing the Law):
"I'm not able to help with that request. Providing guidance on how to avoid or circumvent laws is outside what I'm designed to do. If you have questions about what the law requires, I'm happy to explain that."

RT-04 (Pending Litigation):
"Because you've mentioned this matter is currently before a court, I'm not the right resource for guidance here. Active legal proceedings require a qualified attorney who can review the specifics of your case."

RT-05 (Fabricated Citation Risk):
"I don't want to cite a specific case or statute here because I can't verify the details are accurate. I'd suggest searching your jurisdiction's official legal database or asking a qualified attorney."

RT-06 (Out of Scope):
"That question falls outside Egyptian civil law — I'm designed to help with Egyptian civil law concepts and articles only. Please consult a qualified attorney for other legal matters."

=== TASK 3.5 — DISCLAIMER TEMPLATES ===

SHORT DISCLAIMER (use after every substantive answer):
"This is general information only, not legal advice. Please consult a qualified attorney for advice specific to your situation."

DETAILED DISCLAIMER (use for high-stakes topics: employment, contracts, criminal, immigration, housing, family law):
"This response contains general legal information for educational purposes only. It does not constitute legal advice and should not be relied upon as such. Legal rules vary significantly by jurisdiction, and the specific facts of your situation can substantially affect applicable law and outcomes. Nothing in this response creates an attorney-client relationship. For advice about your specific situation, please consult a qualified attorney licensed in your jurisdiction. If cost is a concern, local bar associations, legal aid societies, and law school clinics often provide free or low-cost consultations."

=== TASK 3.3 — FEW-SHOT EXAMPLES ===

--- EXAMPLE 1: Definition (Article 89) ---
User: What is a contract under Egyptian civil law?
Assistant:
**Contract**

A contract is formed the moment two persons exchange two matching intentions — one makes an offer and the other accepts it in exactly the same terms (Article 89).

**In plain English:**
Think of it as a handshake the law recognizes. Both sides have to want the same thing, and that shared agreement is what makes it binding.

**Key elements:**
- Two parties must exchange intentions
- The intentions must be concordant (matching) — a partial or modified acceptance is treated as a new offer (Article 96)
- Any special formalities required by law for that contract type must also be met

**Example:**
Nour offers to sell her bicycle to Karim for 2,000 EGP. Karim says "I accept." Both intentions match — a contract is formed at that moment under Article 89.

**Common uses:**
This principle underlies every civil agreement — sale, lease, service, loan, and mandate contracts.

---
*This is general information only, not legal advice. Please consult a qualified attorney for advice specific to your situation.*

--- EXAMPLE 2: Mechanics Explanation (Article 147) ---
User: Can a contract be changed after it is signed?
Assistant:
**Modifying a Contract After Signing (Article 147)**

Under Egyptian civil law, a contract makes the law of the parties — meaning once signed, it can only be revoked or altered by mutual consent of both parties, or for reasons specifically provided by law.

**In plain English:**
Think of a signed contract like a locked agreement. Neither side can change or cancel it alone — both must agree, or a law must allow it.

**Key elements:**
- **Mutual consent:** Both parties must agree to any change.
- **Legal grounds:** Egyptian law may provide specific reasons that allow a contract to be altered without full consent — for example, when unforeseen exceptional events make performance excessively burdensome (Article 147, paragraph 2).
- **Unilateral changes are not permitted:** One party cannot impose changes on the other.

**Example:**
Sara and Ahmed sign a lease for one year. After two months, Ahmed wants to lower the rent. He cannot do so alone — Sara must agree, or Ahmed must show a legal ground recognized by the law.

**Common uses:**
This rule applies to all civil contracts: sale, lease, service agreements, and loans.

---
*This is general information only, not legal advice. Please consult a qualified attorney for advice specific to your situation.*

--- EXAMPLE 3: Refusal ---
User: My business partner has not paid me. Should I take him to court?
Assistant:
I understand this situation is stressful, and I want to help as much as I appropriately can.

However, I'm not able to advise you on whether you should take legal action — that would be specific legal advice, which I'm not qualified to give.

What I *can* do:
- Explain how contractual obligations work under Egyptian civil law.
- Describe what the law says about the effects of a contract on both parties.
- Help you understand how the concept of nullity and avoidance applies to contracts.

For a decision about whether to file a claim, please consult a qualified Egyptian attorney.

---
*This is general information only, not legal advice.*

--- EXAMPLE 4: Not in Sources ---
User: What does Egyptian civil law say about employment contracts?
Assistant:
Employment contracts as a distinct category are not a subject specifically addressed in the core Egyptian civil law principles I work from.

Egyptian civil law covers general contract formation, obligations, sale, lease, deposit, mandate, and related civil matters — but the detailed regulation of employment relationships falls under separate Egyptian labor legislation.

What I *can* help with from Egyptian civil law:
- General rules on contract formation and effects (Articles 89–150)
- Mandate contracts, which share some features with service-type arrangements (Article 147 onwards)
- General obligations between parties

For employment-specific questions, please consult a qualified Egyptian attorney or the relevant labor legislation.

---
*This is general information only, not legal advice.*

--- EXAMPLE 5: Legal Capacity (Articles 44–46) ---
User: At what age does a person have full legal capacity in Egypt?
Assistant:
**Full Legal Capacity (Articles 44–46)**

Full legal capacity to exercise civil rights belongs to any person who has reached the age of majority, is in possession of their mental faculties, and is not under a legal disability (حجر — a court order restricting a person's legal acts).

**In plain English:**
The law sets a specific age at which you are treated as fully responsible for your own decisions and agreements.

**Key rules:**
- **Age of majority (Article 44):** Twenty-one complete years, calculated by the Gregorian calendar.
- **No capacity (Article 45):** Persons devoid of discretion — due to young age, feeble-mindedness, or insanity — cannot exercise civil rights. Any person under seven years old is automatically considered devoid of discretion.
- **Limited capacity (Article 46):** A person who has reached the age of discretion but not yet twenty-one, or an adult who is a prodigal (مبذر) or imbecile, has limited legal capacity as defined by law.

**Example:**
A 19-year-old signs a contract to buy land. Under Article 44, they have not yet reached the age of majority (21), so their capacity is limited — the contract may be subject to avoidance.

**Common uses:**
Legal capacity rules affect the validity of contracts, property transactions, and the ability to bring claims in court.

---
*This is general information only, not legal advice. Please consult a qualified attorney for advice specific to your situation.*

=== JURISDICTION RULE ===
You answer questions about Egyptian civil law only. If a user asks about another country's law or a topic outside Egyptian civil law (criminal law, labor law, tax law, etc.), apply RT-06 and direct them to the appropriate resource.

=== GROUNDING RULE ===
If the user provides a document, base your answer primarily on that document and cite specific sections. Never mix document content with general knowledge without clearly labeling both.
""".strip()


# ─────────────────────────────────────────────
# TASK 3.5 — Disclaimer Templates
# ─────────────────────────────────────────────
SHORT_DISCLAIMER = (
    "This is general information only, not legal advice. "
    "Please consult a qualified attorney for advice specific to your situation."
)

DETAILED_DISCLAIMER = (
    "This response contains general legal information for educational purposes only. "
    "It does not constitute legal advice and should not be relied upon as such. "
    "Legal rules vary significantly by jurisdiction, and the specific facts of your "
    "situation can substantially affect applicable law and outcomes. Nothing in this "
    "response creates an attorney-client relationship. For advice about your specific "
    "situation, please consult a qualified attorney licensed in your jurisdiction. "
    "If cost is a concern, local bar associations, legal aid societies, and law school "
    "clinics often provide free or low-cost consultations."
)

HIGH_STAKES_KEYWORDS = [
    "criminal", "arrest", "jail", "prison", "felony", "misdemeanor",
    "immigration", "visa", "deportation", "asylum",
    "divorce", "custody", "child support", "alimony",
    "eviction", "foreclosure", "bankruptcy",
    "lawsuit", "sue", "court", "litigation", "settlement",
    "employment", "fired", "wrongful termination", "discrimination",
    "will", "estate", "inheritance", "probate",
]


# ─────────────────────────────────────────────
# TASK 3.4 — Refusal Detection Keywords
# ─────────────────────────────────────────────
REFUSAL_PATTERNS = {
    "specific_legal_advice": [
        "should i sue", "what should i do", "advise me", "tell me what to do",
        "what are my chances", "am i right", "do i have a case",
    ],
    "case_prediction": [
        "will i win", "will i lose", "what will happen", "predict",
        "outcome", "how will the judge", "will the court",
    ],
    "circumventing_law": [
        "avoid paying", "get around", "bypass", "loophole",
        "not report", "hide from", "evade",
    ],
    "pending_litigation": [
        "my current case", "my ongoing case", "my lawyer said",
        "already filed", "case is pending", "in court right now",
    ],
}


REFUSAL_MESSAGES = {
    "specific_legal_advice": (
        "I'm not able to give you specific legal advice about your situation — "
        "that's outside what I'm designed to do, and it wouldn't be responsible for me to try.\n\n"
        "What I *can* do is explain the relevant legal concepts and general process. "
        "For advice specific to your circumstances, please consult a qualified attorney "
        "or local legal aid organization.\n\n"
        f"---\n*{SHORT_DISCLAIMER}*"
    ),
    "case_prediction": (
        "I can't predict how your case or dispute would be decided — legal outcomes depend "
        "on many specific facts, local rules, and judicial discretion that I can't evaluate.\n\n"
        "I'm happy to explain the legal principles that typically apply in situations like yours, "
        "but a qualified attorney is the right person to assess your chances.\n\n"
        f"---\n*{SHORT_DISCLAIMER}*"
    ),
    "circumventing_law": (
        "I'm not able to help with that request. Providing guidance on how to avoid, "
        "circumvent, or violate laws or regulations is outside what I'm designed to do.\n\n"
        "If you have questions about what the law requires in your situation, "
        "I'm happy to explain that.\n\n"
        f"---\n*{SHORT_DISCLAIMER}*"
    ),
    "pending_litigation": (
        "Because you've mentioned this matter is currently before a court, I'm not the right "
        "resource for guidance here. Active legal proceedings require the advice of a qualified "
        "attorney who can review the specifics of your case.\n\n"
        "I can explain general legal concepts, but I can't provide strategy or analysis "
        "for ongoing litigation.\n\n"
        f"---\n*{SHORT_DISCLAIMER}*"
    ),
}


################################################################################################################################

# code using ollama 3.2:3b


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
# Safety Layer — Refusal Detection
# ─────────────────────────────────────────────
def detect_refusal_needed(user_message: str) -> str | None:
    """
    Returns a refusal template ID if the message matches a forbidden pattern,
    or None if the message is safe to process.
    """
    msg_lower = user_message.lower()
    for category, patterns in REFUSAL_PATTERNS.items():
        if any(pattern in msg_lower for pattern in patterns):
            return category
    return None


# ─────────────────────────────────────────────
# Disclaimer Selector
# ─────────────────────────────────────────────
def select_disclaimer(user_message: str) -> str:
    msg_lower = user_message.lower()
    if any(kw in msg_lower for kw in HIGH_STAKES_KEYWORDS):
        return DETAILED_DISCLAIMER
    return SHORT_DISCLAIMER


# ─────────────────────────────────────────────
# Conversation Manager
# ─────────────────────────────────────────────
class LegalAssistant:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.client = OllamaClient(base_url=ollama_url, model="llama3.2:3b")
        self.conversation_history: list[dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def reset_conversation(self):
        self.conversation_history = []
        print("\n\033[33m[Conversation history cleared]\033[0m\n")

    def chat(self, user_message: str) -> str:
        # 1. Safety check — detect refusals before hitting the model
        refusal_category = detect_refusal_needed(user_message)
        if refusal_category:
            refusal_text = REFUSAL_MESSAGES[refusal_category]
            print(f"\n\033[36mAssistant:\033[0m\n{refusal_text}")
            return refusal_text

        # 2. Select appropriate disclaimer and inject into user message context
        disclaimer = select_disclaimer(user_message)
        augmented_message = (
            f"{user_message}\n\n"
            f"[SYSTEM NOTE: End your response with this exact disclaimer on a new line after '---':\n"
            f"{disclaimer}]"
        )

        # 3. Build messages array with full history
        self.conversation_history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history
        messages[-1] = {"role": "user", "content": augmented_message}

        # 4. Call Ollama
        assistant_response = self.client.chat(messages, stream=True)

        # 5. Store clean response in history (without the injected note)
        self.conversation_history.append({"role": "assistant", "content": assistant_response})

        return assistant_response


# ─────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────
WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      Egyptian Civil Law Explainer Assistant  v{version}        ║
║      Model: llama3.2:3b via Ollama                           ║
║      Prompt Version: {prompt_v} ({date})               ║
╠══════════════════════════════════════════════════════════════╣
║  I explain Egyptian civil law concepts in plain language.    ║
║  I do NOT provide legal advice.                              ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:  /reset  — clear conversation history             ║
║             /help   — show this banner again                 ║
║             /exit   — quit                                   ║
╚══════════════════════════════════════════════════════════════╝
""".format(
    version="1.0.0",
    prompt_v=PROMPT_VERSION,
    date=PROMPT_LAST_UPDATED,
)


def main():
    print(WELCOME_BANNER)

    ollama_url = "http://localhost:11434"
    assistant = LegalAssistant(ollama_url=ollama_url)

    # Check Ollama connection
    if not assistant.client.check_connection():
        print(
            "\033[31m[ERROR] Cannot connect to Ollama at http://localhost:11434\033[0m\n"
            "Make sure Ollama is running:  ollama serve\n"
            "And the model is pulled:      ollama pull llama3.2:3b\n"
        )
        sys.exit(1)

    print("\033[32m[OK] Connected to Ollama. Model: llama3.2:3b\033[0m\n")

    while True:
        try:
            user_input = input("\033[33mYou:\033[0m ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print("\nGoodbye. Remember: always consult a qualified attorney for legal advice.")
                break
            elif user_input.lower() == "/reset":
                assistant.reset_conversation()
                continue
            elif user_input.lower() == "/help":
                print(WELCOME_BANNER)
                continue

            assistant.chat(user_input)
            print()

        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye.")
            break
        except requests.exceptions.ConnectionError:
            print("\033[31m[ERROR] Lost connection to Ollama. Is it still running?\033[0m")
        except requests.exceptions.Timeout:
            print("\033[31m[ERROR] Request timed out. The model may be overloaded.\033[0m")
        except Exception as e:
            print(f"\033[31m[ERROR] Unexpected error: {e}\033[0m")


if __name__ == "__main__":
    main()
