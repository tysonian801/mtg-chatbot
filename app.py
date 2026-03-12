"""
MTG Rules Assistant - A Streamlit app for Magic: The Gathering rules questions.

This app provides judge-like responses to MTG rules questions using GPT-4o-mini
with a RAG (Retrieval-Augmented Generation) pipeline backed by the official MTG
Comprehensive Rules. On startup the rules are fetched, chunked into named rule
sections, and embedded with OpenAI's text-embedding-3-small model. Each query
retrieves the most semantically relevant chunks before calling the LLM, ensuring
accurate rule citations including newly released mechanics.

Features:
- GPT-4o-mini powered MTG rules assistance with RAG over official rules
- Format-specific responses (Standard, Modern, Legacy, Commander, Limited)
- Adjustable response detail levels
- Mobile-optimized interface

Author: Tyson Clegg
Version: 2.0
"""

import hashlib
import os
import re
from typing import Optional

import numpy as np
import openai
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="MTG Rules Assistant",
    page_icon="🃏",
    layout="wide"
)

# Rule sections irrelevant to supported formats — omitted from the knowledge base
# to reduce token usage and noise. Covers niche/discontinued casual variants,
# Unfinity-only mechanics, and ante (banned everywhere).
# TODO: determine if we should uncomment this- for now I
OMIT_SECTIONS = {
#     "123",  # Stickers (Unfinity/acorn only)
#     "311",  # Planes (Planechase only)
#     "312",  # Phenomena (Planechase only)
#     "313",  # Vanguards (Vanguard format, discontinued)
#     "314",  # Schemes (Archenemy only)
#     "315",  # Conspiracies (Conspiracy Draft only)
#     "407",  # Ante (banned in all sanctioned formats)
#     "717",  # Attraction Cards (Unfinity/acorn only)
#     "727",  # Rad Counters (Unfinity/acorn only)
#     "802",  # Attack Multiple Players Option (niche multiplayer)
#     "803",  # Attack Left and Attack Right Options (niche multiplayer)
#     "804",  # Deploy Creatures Option (niche multiplayer)
#     "807",  # Grand Melee Variant (10+ player games)
#     "808",  # Team vs. Team Variant (niche)
#     "809",  # Emperor Variant (niche)
#     "811",  # Alternating Teams Variant (niche)
#     "901",  # Planechase
#     "902",  # Vanguard (discontinued)
#     "904",  # Archenemy
#     "905",  # Conspiracy Draft (also contains the glossary as trailing content)
}


# Initialize OpenAI client (cached so it's only created once per session)
@st.cache_resource
def init_openai_client():
    """
    Initialize OpenAI client with API key from environment variables or Streamlit secrets.
    Cached with st.cache_resource so the client is reused across reruns.
    """
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("❌ OPENAI_API_KEY not found. Please set it in your environment or Streamlit secrets.")
        st.stop()

    if api_key.startswith("sk-"):
        return openai.OpenAI(api_key=api_key)
    else:
        st.error("❌ Invalid API key format. Expected an OpenAI key starting with 'sk-'.")
        st.stop()


@st.cache_data(ttl=86400)  # Refresh once per day
def fetch_rules_text() -> tuple[str | None, str | None]:
    """
    Fetch the current MTG Comprehensive Rules .txt file.

    Visits magic.wizards.com/en/rules, finds the current .txt download link
    (which changes with each set release), then fetches and returns the full
    rules text. Cached for 24 hours to avoid re-downloading on every request.

    Returns:
        (rules_text, txt_url) on success, or (None, error_message) on failure.
    """
    try:
        page = requests.get("https://magic.wizards.com/en/rules", timeout=15)
        page.raise_for_status()
        match = re.search(r'https?://[^"\'<>]+MagicCompRules[^"\'<>]+\.txt', page.text)
        if not match:
            return None, "Could not find rules .txt link on the rules page."
        txt_url = match.group(0)
        rules = requests.get(txt_url, timeout=30)
        rules.raise_for_status()
        return rules.text, txt_url
    except Exception as e:
        return None, str(e)


def build_rule_chunks(rules_text: str) -> list[str]:
    """
    Split the rules text into named section chunks suitable for embedding.

    Splits at fine-grained named section headers (e.g. '702.171. Saddle',
    '117. Timing and Priority') so each chunk is a self-contained rule or
    keyword definition. TOC stubs and irrelevant format sections are excluded.

    Args:
        rules_text: Full rules text from fetch_rules_text().

    Returns:
        List of rule section strings, each starting with its section header.
    """
    sections = re.split(r'(?m)(?=^\d{3,}(?:\.\d+)*\. [A-Z])', rules_text)
    chunks = []
    for s in sections:
        m = re.match(r'^(\d{3})\.', s)
        if m:
            if m.group(1) in OMIT_SECTIONS:
                continue
            if len(s.strip()) < 100:  # TOC stub — just a header line with no content
                continue
        if s.strip():
            chunks.append(s.strip())
    return chunks


@st.cache_data(show_spinner=False)
def build_embeddings(rules_hash: str, api_key: str) -> tuple[list[str], np.ndarray] | tuple[None, None]:
    """
    Build a vector knowledge base from the MTG Comprehensive Rules.

    Chunks the rules into named sections, then embeds each chunk using
    OpenAI's text-embedding-3-small model. Results are cached keyed on
    rules_hash so the knowledge base is automatically rebuilt when the
    rules file updates (e.g. after a new set release).

    Args:
        rules_hash: MD5 hash of the rules text — used as the cache key.
        api_key: OpenAI API key for the embeddings request.

    Returns:
        (chunks, embeddings) where embeddings is a float32 numpy array of
        shape (n_chunks, 1536) with L2-normalized rows for cosine similarity,
        or (None, None) on failure.
    """
    rules_text, _ = fetch_rules_text()
    if not rules_text:
        return None, None

    chunks = build_rule_chunks(rules_text)
    client = openai.OpenAI(api_key=api_key)

    all_embeddings = []
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in response.data])

    embeddings = np.array(all_embeddings, dtype=np.float32)
    # L2-normalize so retrieval is a simple dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)

    return chunks, embeddings


def retrieve_relevant_chunks(
        client: openai.OpenAI,
        query: str,
        chunks: list[str],
        embeddings: np.ndarray,
        top_k: int = 20,
    ) -> str:
    """
    Retrieve the rule chunks most semantically relevant to the query.

    Embeds the query with text-embedding-3-small, computes cosine similarity
    against the pre-built knowledge base, and returns the top-k chunks
    concatenated in rule-number order.

    Args:
        client: OpenAI client instance.
        query: The user's question.
        chunks: List of rule section strings from build_embeddings().
        embeddings: Normalized embedding matrix from build_embeddings().
        top_k: Number of chunks to retrieve.

    Returns:
        Concatenated string of the most relevant rule sections.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    query_emb = np.array(response.data[0].embedding, dtype=np.float32)
    query_emb /= max(np.linalg.norm(query_emb), 1e-9)

    scores = embeddings @ query_emb
    top_indices = np.argsort(scores)[-top_k:][::-1]
    # Return in original rule order so context reads coherently
    top_indices_sorted = sorted(top_indices)
    return "\n\n".join(chunks[i] for i in top_indices_sorted)


def get_gpt_response(
        client: openai.OpenAI,
        question: str,
        card_context: Optional[str] = None,
        format_type: str = "Any Format",
        response_style: str = "Judge-Level",
        chunks: Optional[list[str]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> str:
    """
    Get a response from GPT-4o-mini for an MTG rules question.

    For Judge-Level responses, relevant rule sections are retrieved from the
    pre-built RAG knowledge base and injected into the prompt, ensuring the
    model always cites the current official rules. Other response styles use
    training data, which is faster and sufficient for common questions.

    Args:
        client: OpenAI client instance.
        question: User's MTG rules question.
        card_context: Optional card text from Scryfall API.
        format_type: MTG format (Any Format, Standard, Modern, etc.)
        response_style: Detail level (Extra-Concise, Concise, Detailed, Judge-Level).
        chunks: Pre-built rule chunks from build_embeddings().
        embeddings: Pre-built embedding matrix from build_embeddings().

    Returns:
        GPT-4o-mini response as a formatted string.
    """
    format_context = (
        f"Focus on {format_type} format rules and interactions."
        if format_type != "Any Format"
        else "Consider all formats when answering."
    )

    style_instruction = {
        "Extra-Concise": "Be extremely concise. Answer in one or two sentences unless accuracy requires more. Bullet points are helpful.",
        "Concise": "Keep your answer brief and to the point. Don't include more detail than necessary. Bullet points are helpful.",
        "Detailed": "Provide a thorough explanation with examples.",
        "Judge-Level": "Give a comprehensive, judge-quality answer with exact rule citations (e.g. CR 702.12b) and step-by-step breakdowns.",
    }.get(response_style, "Provide a detailed explanation.")

    # For Judge-Level, retrieve the most relevant rule sections from the
    # knowledge base and inject them directly into the prompt.
    rules_context = ""
    if response_style == "Judge-Level":
        if chunks is None or embeddings is None:
            return (
                "❌ **Could not load the official MTG Comprehensive Rules.**\n\n"
                "Please try switching to a different response style (Detailed, Concise, or Extra-Concise), "
                "which will use the model's training data instead."
            )
        relevant = retrieve_relevant_chunks(client, question, chunks, embeddings)
        rules_context = f"\n\n**Relevant Official MTG Comprehensive Rules:**\n{relevant}"

    system_prompt = f"""You are an expert Magic: The Gathering judge assistant.

**Format Context:** {format_context}
**Response Style:** {style_instruction}

**Rules Priority (strictly follow this order):**
1. FIRST, cite the official Magic: The Gathering Comprehensive Rules provided below — this is the authoritative source. Reference rule numbers directly (e.g., "CR 116.3c").
2. Only supplement with other sources (community rulings, judge blogs, etc.) if the official rules do not cover the question.

**When answering:**
1. Cite the governing rule number(s) from the Comprehensive Rules provided
2. Use precise MTG terminology
3. Explain the reasoning behind the ruling
4. If card text is provided, reference it
5. If the question involves the stack or timing, explain the sequence using priority rules (CR 116)
6. For complex interactions, break down each step with rule citations
7. When evaluating tap restrictions (e.g. summoning sickness), carefully identify *whose* ability is being activated. Summoning sickness (CR 302.6) restricts a creature's *own* activated abilities that include {{T}} in their cost — it does NOT prevent a creature from being tapped to pay the cost of another permanent's ability (e.g. Crew, Station, Convoke).

**Format your answers with:**
- A clear ruling summary upfront
- Governing rule number(s) after the summary
- Step-by-step breakdown for complex interactions{rules_context}"""

    user_message = "**Question:** " + question
    if card_context:
        user_message += f"\n\n**Relevant card text:**\n{card_context}"

    max_output_tokens = {
        "Extra-Concise": 300,
        "Concise": 600,
        "Detailed": 1200,
        "Judge-Level": 4000,
    }.get(response_style, 2000)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=system_prompt,
            input=user_message,
            max_output_tokens=max_output_tokens,
            temperature=0,
        )
        for block in response.output:
            if block.type == "message":
                for content in block.content:
                    if content.type == "output_text":
                        return content.text
        return "❌ **No response returned.** Please try again."
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg and "rate_limit" in error_msg.lower():
            return (
                "❌ **Rate Limit Reached**\n\n"
                "Too many requests in a short period. Please wait a moment and try again."
            )
        elif "insufficient_quota" in error_msg or "429" in error_msg:
            return (
                "❌ **API Quota Exceeded**\n\n"
                "You've reached your OpenAI API usage limit. Please:\n\n"
                "1. **Check your billing** at https://platform.openai.com/account/billing\n"
                "2. **Add credits** to your account\n"
                "3. **Try again later**"
            )
        elif "invalid_api_key" in error_msg:
            return "❌ **Invalid API Key**\n\nPlease check your OpenAI API key in the .env file."
        else:
            return f"❌ **Error getting response:** {error_msg}"


def main():
    """Main Streamlit app function."""

    client = init_openai_client()

    st.title("🃏 MTG Rules Assistant")
    st.markdown("Get judge-quality answers to any MTG rules question.")

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")

        # Format selection
        st.subheader("Format")
        format_type = st.selectbox(
            "Select format for context:",
            ["Any Format", "Standard", "Modern", "Legacy", "Commander", "Limited"]
        )

        # Response style
        st.subheader("Response Style")
        response_style = st.selectbox(
            "How detailed should the response be?",
            ["Extra-Concise", "Concise", "Detailed", "Judge-Level"],
            index=3
        )

        # Build the RAG knowledge base on startup with clear step-by-step status.
        # Both fetch_rules_text and build_embeddings are cached, so after the first
        # load this completes instantly and the status collapses automatically.
        st.subheader("Rules Status")
        chunks, embeddings = None, None
        with st.status("Loading rules...", expanded=True) as status:
            st.write("Fetching official rules from magic.wizards.com...")
            rules_text, rules_info = fetch_rules_text()

            if not rules_text:
                status.update(label="Failed to load rules", state="error")
                st.error(f"Error: {rules_info}")
            else:
                st.write("Building knowledge base (may take ~30s on first load)...")
                rules_hash = hashlib.md5(rules_text.encode()).hexdigest()
                api_key = client.api_key
                chunks, embeddings = build_embeddings(rules_hash, api_key)

                if chunks is None:
                    status.update(label="Failed to build knowledge base", state="error")
                else:
                    status.update(
                        label=f"Rules ready ({len(chunks):,} sections indexed)",
                        state="complete",
                        expanded=False,
                    )

    col1, col2 = st.columns([2, 1])

    with col1:
        # Question input — pre-populated when an example button is clicked
        question = st.text_area(
            "Enter your MTG rules question:",
            value=st.session_state.pop("example_question", ""),
            placeholder="e.g., How does indestructible interact with -1/-1 counters?",
            height=120
        )

        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("Looking up the rules..."):
                    card_context = None
                    response = get_gpt_response(
                        client, question, card_context,
                        format_type, response_style,
                        chunks, embeddings,
                    )
                    st.subheader("Answer")
                    st.markdown(response)
            else:
                st.warning("Please enter a question.")

    with col2:
        # Example questions
        st.subheader("Examples")
        examples = [
            "How does indestructible interact with -1/-1 counters?",
            "Can I counter a spell that can't be countered?",
            "What happens when a creature with protection from red is targeted by a red spell?",
            "How do multiple instances of the same ability work?",
            "What's the difference between 'destroy' and 'exile'?",
            "Can I respond to a triggered ability?",
            "How does the stack work with multiple spells?",
            "What happens if a creature loses all abilities?"
        ]

        for example in examples:
            if st.button(example, key=example, use_container_width=True):
                st.session_state.example_question = example
                st.rerun()

    # Footer
    st.markdown("---")
    st.caption("⚖️ This assistant provides guidance but is not a substitute for official MTG rules or judge rulings.")


if __name__ == "__main__":
    main()
