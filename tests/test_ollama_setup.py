import ollama

print("Sende Anfrage an Llama3...")

response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': 'Erkläre in einem kurzen Satz, was eine Zeitreihe ist.',
  },
])

print("\nANTWORT VON KI:")
print(response['message']['content'])