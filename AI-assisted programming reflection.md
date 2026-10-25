Write AI-assisted programming reflection# AI-Assisted Development

## 1. Which AI tool(s) did you use?

- **ChatGPT (OpenAI):**
- **Copilot**

## 2. What prompts did you use?
Examples of prompts I used:

- "How can I load a pretrained machine learning model from Hugging Face?"
- "How do I integrate a machine learning model with FastAPI?"
- "Show me an example how to use Streamlit and take user input and connect to the fast api server."
- "How can I write unit tests using pytest?"
- "What is the best way to load a model only once instead of loading it for every prediction?"
- "Help me understand this Python import error."
- "Write the read me file and explain exactly how to run this application."


## 3. Which generated suggestions or code were useful?

AI assistance was particularly useful for:

- The mose useful suggestion was how to load the model dirctly on application level using fastAPI method  "app.state.model = load_model()" to load the model only once.

## 4. What errors or issues did the AI-generated code contain?

Not all AI-generated suggestions worked immediately.

- **Import errors:** Some suggested import structures caused `ModuleNotFoundError` during pytest execution.
- **Project structure:** Certain suggestions required adjustments to match the actual project directory structure.

## 5. What did you learn about AI-assisted programming?

- AI-assisted programming is a great tool that helps complete repetitive tasks faster and more efficiently. It also makes learning new concepts much easier. For example, I had never worked with Streamlit before, as most of my experience was with React, but AI helped me understand the basics very quickly.
- AI is a great assistant, but its generated code should not be treated as absolute truth. It is important to review, understand, and test the code rather than blindly trusting it.