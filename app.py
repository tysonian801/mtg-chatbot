"""
MTG Rules Assistant - A Streamlit app for Magic: The Gathering rules questions.

This app provides judge-like responses to MTG rules questions using GPT-4o and
can optionally fetch card information from the Scryfall API.

Features:
- GPT-4o powered MTG rules assistance
- Format-specific responses (Standard, Modern, Legacy, Commander, Limited)
- Adjustable response detail levels
- Mobile-optimized interface

Author: Tyson Clegg
Version: 1.0
"""

import streamlit as st
import openai
import os
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="MTG Rules Assistant",
    page_icon="🃏",
    layout="wide"
)

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

def get_gpt_response(
        client: openai.OpenAI,
        question: str,
        card_context: Optional[str] = None,
        format_type: str = "Any Format",
        response_style: str = "Judge-Level"
    ) -> str:
    """
    Get response from GPT-4o for MTG rules questions.

    Uses the OpenAI Responses API with web search so it can look up the
    official MTG Comprehensive Rules at magic.wizards.com/en/rules before
    answering. Official rules are always consulted first; other sources are
    only used when the official rules don't cover the question.

    Args:
        client: OpenAI client instance
        question: User's MTG rules question
        card_context: Optional card text from Scryfall API
        format_type: MTG format to focus on (Any Format, Standard, Modern, etc.)
        response_style: Detail level (Extra-Concise, Concise, Detailed, Judge-Level)

    Returns:
        str: GPT-4o response as formatted string

    Raises:
        Exception: If API call fails (handled with user-friendly error messages)
    """
    format_context = f"Focus on {format_type} format rules and interactions." if format_type != "Any Format" else "Consider all formats when answering."

    style_instruction = {
        "Extra-Concise": "Be extremely concise. Answer in one or two sentences unless accuracy requires more. Bullet points are helpful.",
        "Concise": "Keep your answer brief and to the point. Don't include more detail than necessary. Bullet points are helpful.",
        "Detailed": "Provide a thorough explanation with examples.",
        "Judge-Level": "Give a comprehensive, judge-quality answer with exact rule citations (e.g. CR 702.12b) and step-by-step breakdowns."
    }.get(response_style, "Provide a detailed explanation.")

    system_prompt = f"""You are an expert Magic: The Gathering judge assistant.

**Format Context:** {format_context}
**Response Style:** {style_instruction}

**Rules Priority (strictly follow this order):**
1. FIRST, search and cite the official Magic: The Gathering Comprehensive Rules at https://magic.wizards.com/en/rules — this is the authoritative source. Reference rule numbers directly (e.g., "CR 116.3c").
2. Only supplement with other sources (community rulings, judge blogs, etc.) if the official rules do not cover the question.

**When answering:**
1. Search the official Comprehensive Rules first and cite the governing rule number(s)
2. Use precise MTG terminology
3. Explain the reasoning behind the ruling
4. If card text is provided, reference it
5. If the question involves the stack or timing, explain the sequence using priority rules (CR 116)
6. For complex interactions, break down each step with rule citations

**Format your answers with:**
- Governing rule number(s) upfront when applicable
- Step-by-step breakdown for complex interactions
- A clear ruling summary"""

    user_message = "**Question:** " + question
    if card_context:
        user_message += f"\n\n**Relevant card text:**\n{card_context}"

    # Token budget by response style
    max_output_tokens = {
        "Extra-Concise": 300,
        "Concise": 600,
        "Detailed": 1200,
        "Judge-Level": 2000,
    }.get(response_style, 2000)

    try:
        # Judge-Level uses live web search to fetch the latest official Comprehensive
        # Rules from magic.wizards.com/en/rules. All other styles use training data,
        # which is faster, cheaper, and sufficient for common rules questions.
        tools = [{"type": "web_search_preview"}] if response_style == "Judge-Level" else []

        response = client.responses.create(
            model="gpt-4o",
            tools=tools,
            instructions=system_prompt,
            input=user_message,
            max_output_tokens=max_output_tokens,
        )
        # Extract text output from the response
        for block in response.output:
            if block.type == "message":
                for content in block.content:
                    if content.type == "output_text":
                        return content.text
        return "❌ **No response returned.** Please try again."
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "429" in error_msg:
            return "❌ **API Quota Exceeded**\n\nYou've reached your OpenAI API usage limit. Please:\n\n1. **Check your billing** at https://platform.openai.com/account/billing\n2. **Add credits** to your account\n3. **Try again later**"
        elif "invalid_api_key" in error_msg:
            return "❌ **Invalid API Key**\n\nPlease check your OpenAI API key in the .env file."
        else:
            return f"❌ **Error getting response:** {error_msg}"


def main():
    """Main Streamlit app function."""

    # Initialize OpenAI client
    client = init_openai_client()

    # Header
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
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Question input — pre-populated when an example button is clicked
        question = st.text_area(
            "Enter your MTG rules question:",
            value=st.session_state.pop("example_question", ""),
            placeholder="e.g., How does indestructible interact with -1/-1 counters?",
            height=120
        )
        
        # Submit button
        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("Looking up the rules..."):
                    card_context = None

                    # Get GPT response with settings
                    response = get_gpt_response(client, question, card_context, format_type, response_style)

                    # Display response
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