
import openai

# Set your OpenAI API key
openai.api_key = "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"

# Make a simple chat completion request
try:
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")
