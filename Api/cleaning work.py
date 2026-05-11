import requests

# def chat_with_mistral(prompt):
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "mistral",
#             "prompt": prompt,
#             "stream": False,
#             "options": {
#                 "num_predict": 150,  # limit response length (faster replies)
#                 "temperature": 0.3
#             }
#         }
#     )
#     return response.json()["response"]
#
#
# print("Mistral Chatbot Ready (type exit to stop)\n")
#
# while True:
#     user_input = input("You: ")
#
#     if user_input.lower() == "exit":
#         break
#
#     reply = chat_with_mistral(user_input)
#     print("Mistral:", reply)


def chat(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False,
        }
    )
    return res.json()["response"]

print("Fast Chatbot Ready\n")

while True:
    msg = input("You: ")
    if msg.lower() == "exit":
        break
    print("Bot:", chat(msg))


