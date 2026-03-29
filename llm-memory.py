from openai import OpenAI
from setup.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ messages lives OUTSIDE — this is the memory
messages = [
    {"role": "system", "content": "You are a helpful assistant. Remember everything the user tells you."}
]

print("\n-------------------------------")
def chat(user_input):
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    for msg in messages:
        print(f"{msg['role']}: {msg['content']}")


    response = client.chat.completions.create(model="gpt-4o-mini",messages=messages)
    reply = response.choices[0].message.content

    # Add assistant reply to history — this is what gives it memory
    messages.append({"role": "assistant", "content": reply})

    print(f"Assistant: {reply}\n")
    return reply


# ▶️ Test — each line is a separate call
chat("Hi! My name is Nagarjuna and I live in Hyderabad.")
chat("I am currently learning AI agents.")
chat("My favourite programming language is Python.")
print("\n--------------CHECK LLM MEMORY NOW-----------------\n")
chat("What is my name?")  # should remember ✅
chat("Where do I live?")  # should remember ✅
chat("What am I learning?")  # should remember ✅
chat("What language do I like?")  # should remember ✅
