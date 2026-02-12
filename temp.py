import google.generativeai as genai

genai.configure(api_key="AIzaSyB4oPyYCI41WJkhiXONjCLPsUa9hFwxFic")

for m in genai.list_models():
    print(m.name, "->", m.supported_generation_methods)
