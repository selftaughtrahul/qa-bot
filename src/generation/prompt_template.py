from langchain_core.prompts import PromptTemplate

CONVERSATIONAL_QA_PROMPT_TEMPLATE = """You are a helpful and precise assistant.
Answer the user's question ONLY based on the provided context.
Use the conversation history to understand follow-up questions and resolve references like "it", "that", "the above", etc.
If the answer cannot be found in the context, say: "I don't know based on the provided documents."

Context (from documents):
{context}

Conversation History:
{chat_history}

Current Question: {question}

Answer:"""

CONVERSATIONAL_QA_PROMPT = PromptTemplate(
    template=CONVERSATIONAL_QA_PROMPT_TEMPLATE,
    input_variables=["context", "chat_history", "question"]
)