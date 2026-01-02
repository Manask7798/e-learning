# Dieser Code ist geschrieben Anlehung an
# https://mohdmus99.medium.com/strategies-and-techniques-for-managing-the-size-of-the-context-window-when-using-llm-large-3c2dbc5dcc3a
from typing import List
import tiktoken
from langchain_core.messages import AnyMessage

class MessageHandler:
    def __init__(self, model, max_tokens):
        self.max_tokens = max_tokens
        self.conversation: List[AnyMessage] = []
        self.total_tokens = 0
        self.model = model

    def count_tokens(self, message):
        # Anzahl Token, mit tiktoken berechnet
        # Kann meist nur präzise für OpenAI-Modelle verwendet werden
        encoding = tiktoken.encoding_for_model(self.model)
        tokens = encoding.encode(message)
        return len(tokens)

    def add_message(self, message):
        message_content = message.content
        message_tokens = self.count_tokens(message_content)
        self.conversation.append(message)
        self.total_tokens += message_tokens

        # Entfernen der ältesten Nachricht, falls das Tokenlimit überschritten wurde
        while self.total_tokens > self.max_tokens:
            removed_message = self.conversation.pop(0)
            self.total_tokens -= self.count_tokens(removed_message)

    def get_conversation(self):
        return self.conversation
