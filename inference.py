from transformers import pipeline

model_path = "phi-4-28.3B-zero"
model_path = "./" + model_path

pipe = pipeline("text-generation", model=model_path)

messages = []

print("/exit to exit\n/reset to reset")

while True:
    user = input("User: ")
    if user == "/exit":
        break

    if user == "/reset":
        messages = []
        print("The conversation has reset!")

    messages.append({"role": "user", "content": user})

    role_assistant = pipe(messages)[0]["generated_text"][-1]
    messages.append(role_assistant)
    print("Assistant:", role_assistant["content"])
    