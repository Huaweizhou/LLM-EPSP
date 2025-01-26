from openai import OpenAI

api_key = "sk-2EDY9F7TMNULFOhV890953EcBd93437dA534865745366e57"
api_base = "https://api.freegpt.art/v1"
client = OpenAI(api_key=api_key, base_url=api_base)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo-0301",
#   stream: False,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
