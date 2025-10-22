from summarizer_agent import summarize

text = "LangChain is a framework for developing applications powered by language models. It enables developers to build applications that can understand and generate human-like text.using the project's Python to verify the TypeError is resolved and capture output. This will run the LLM call; if the LLM endpoint isn't reachable the script may raise a networking error — that's okay, we'll capture it and suggest fallbacks. I'll execute the script now."
summary = summarize(text)
print("Summary:", summary)