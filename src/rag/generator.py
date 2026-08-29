"""
LLM-based answer generator for the Steam Reviews RAG system.

Uses the Groq API to generate answers based on retrieved reviews.
Supports conversation history for multi-turn interactions.
"""

import requests


class ReviewGenerator:
    """Generates answers from retrieved reviews using Groq API."""

    def __init__(self, api_key, model="openai/gpt-oss-20b",
                 api_url="https://api.groq.com/openai/v1/chat/completions",
                 max_tokens=512, temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conversation_history = []

        self.system_prompt = (
            "You are a helpful gaming assistant that answers questions "
            "based ONLY on the provided Steam reviews. "
            "Do not make up information. If the reviews don't contain "
            "enough information to answer, say so. "
            "Always mention which games the information comes from. "
            "Be concise but informative. Use the review sentiments "
            "(Recommended / Not Recommended) to support your answers."
        )

    def _build_context(self, retrieved_reviews):
        """Format retrieved reviews into a context string."""
        if not retrieved_reviews:
            return ""
        
        parts = []
        for i, (sentence, score) in enumerate(retrieved_reviews):
            parts.append(f"[Review {i+1} | Relevance: {score:.2f}] {sentence}")
        return "\n\n".join(parts)

    def generate(self, query, retrieved_reviews, use_history=True):
        """Generate an answer using the LLM with retrieved reviews as context.

        Args:
            query: user's question
            retrieved_reviews: list of (sentence, score) from retriever
            use_history: whether to include conversation history

        Returns:
            str: generated answer or error message
        """
        if not retrieved_reviews:
            return "I couldn't find any relevant reviews to answer your question."

        context = self._build_context(retrieved_reviews)

        user_message = (
            f"Based on the following Steam reviews, answer this question: {query}\n\n"
            f"Reviews:\n{context}"
        )

        # Build messages list
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history for context
        if use_history and self.conversation_history:
            # Keep last 3 exchanges to avoid token limit
            recent_history = self.conversation_history[-6:]
            messages.extend(recent_history)

        messages.append({"role": "user", "content": user_message})

        # Call Groq API
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=30,
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]

            # Store in history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})

            return answer

        except requests.exceptions.Timeout:
            return "Error: The request timed out. Please try again."
        except requests.exceptions.HTTPError as e:
            return f"Error: API returned status {e.response.status_code}. Check your API key."
        except Exception as e:
            return f"Error: {str(e)}"

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_history(self):
        """Return conversation history for display."""
        return self.conversation_history