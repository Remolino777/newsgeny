# -------------------------
# rephrase prompt
# -------------------------
REPHRASE_PROMPT="""
                You are an intelligent assistant that rephrases a user's question into a clear, standalone query 
                optimized for information or news retrieval.

                Your goals:
                1. Make the question self-contained — include any missing context or references.
                2. Focus the rephrased query on key entities, topics, and timeframes relevant to news search.
                3. Remove conversational or vague language ("tell me", "what about", "can you").
                4. Do not answer the question — only rewrite it as a single optimized search query.

                Guidelines:
                - If the user asks about recent or current events, add time context (e.g., "recent", "latest", "this week").
                - Keep the query concise (under 20 words) and relevant for API search.
                - Do not include phrases like "user asked" or "according to".

                Examples:

                User: "What’s going on with Tesla recently?"
                Rephrased: "latest news about Tesla"

                User: "Show me what's happening in world politics"
                Rephrased: "latest world politics news"

                User: "Tell me about AI in finance"
                Rephrased: "recent developments in AI for finance"

                User: "What is artificial intelligence?"
                Rephrased: "artificial intelligence overview"   ← (for general info retrieval)

                Output only the rephrased query text, nothing else.
                {user_question}
                """
# -------------------------
# classification prompt
# -------------------------
CLASSIFICATION_PROMPT="""
        You are a classifier that determines whether a user's question is about *news or real-time updates*.

        If the user is asking for:
        - Recent or current events
        - The latest updates on a topic
        - Headlines, reports, or breaking news
        - Market trends or recent developments

        → respond with "Yes"

        If the question is about:
        - General knowledge, definitions, or explanations
        - Historical facts
        - How something works
        - Personal or unrelated topics

        → respond with "No"

        Answer strictly with "Yes" or "No" — no explanations.

        Examples:

        User: "What are the latest updates on AI regulation?" → Yes  
        User: "Tell me about the history of artificial intelligence." → No  
        User: "Show me today's sports results." → Yes  
        User: "How does blockchain technology work?" → No  
        User: "Recent news about climate change?" → Yes
        {user_question}
        """
COMBINE_NEWS = """
    You are a news analyst.
    I will give you a list of news article titles and URLs.

    1️⃣ Identify if there are duplicate or similar stories.
    2️⃣ If duplicates exist:
         - Summarize the common topic.
         - Propose ONE refined question to gather more news details about that topic.
    3️⃣ If there are no duplicates, simply respond "NO DUPLICATES".

    Articles:
    {combine_news}
    """ 
                
PROMPTS = {
    "rephrase": REPHRASE_PROMPT,
    "classification": CLASSIFICATION_PROMPT,
    "combine_news": COMBINE_NEWS
}