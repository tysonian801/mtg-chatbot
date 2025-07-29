"""
MTG Rules Assistant - A Streamlit app for Magic: The Gathering rules questions.

This app provides judge-like responses to MTG rules questions using GPT-4o and
can optionally fetch card information from the Scryfall API.

Features:
- GPT-4o powered MTG rules assistance
- Format-specific responses (Standard, Modern, Legacy, Commander, Limited)
- Adjustable response detail levels
- Password protection for controlled access
- Card name detection (future: Scryfall integration)
- Mobile-optimized interface

Author: Tyson Clegg
Version: 1.0
"""

import streamlit as st
import openai
import os
from typing import Optional
import re

# Page configuration
st.set_page_config(
    page_title="MTG Rules Assistant",
    page_icon="🃏",
    layout="wide"
)

# Initialize OpenAI client
def init_openai_client():
    """
    Initialize OpenAI client with API key from environment variables or Streamlit secrets.
    
    This function handles both local development (using .env file) and deployment
    (using Streamlit secrets). It validates the API key format and provides
    helpful error messages if the key is missing or invalid.
    
    Returns:
        openai.OpenAI: Configured OpenAI client instance
        
    Raises:
        st.stop(): If API key is missing or invalid
    """
    # Try to get API key from Streamlit secrets first (for deployment)
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        api_key = None
    
    # Fall back to environment variable (for local development)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("❌ OPENAI_API_KEY not found. Please set it in your environment or Streamlit secrets.")
        st.stop()
    
    # Validate API key format (OpenAI keys start with 'sk-' or 'sk-svcacct-')
    if api_key and (api_key.startswith("sk-") or api_key.startswith("sk-svcacct-")):
        return openai.OpenAI(api_key=api_key)
    else:
        st.error(f"❌ Invalid API key format. Got: {api_key[:20] if api_key else 'None'}...")
        st.stop()

def get_gpt_response(
        client: openai.OpenAI, 
        question: str, 
        card_context: Optional[str] = None, 
        format_type: str = "Any Format", 
        response_style: str = "Detailed"
    ) -> str:
    """
    Get response from GPT-4o for MTG rules questions.
    
    This function constructs a comprehensive prompt for GPT-4o that includes:
    - MTG-specific system instructions
    - Format context (Standard, Modern, etc.)
    - Response style preferences
    - Card context (if available)
    - MTG rules context for better accuracy
    
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
    # Build the system prompt with settings
    format_context = f"Focus on {format_type} format rules and interactions." if format_type != "Any Format" else "Consider all formats when answering."
    
    # Define response style instructions for different detail levels
    style_instruction = {
        "Extra-Concise": "Be extremely concise. Unless absolutely necessary to answer the question accurately, answer in one or two sentences. Bullet points are helpful.",
        "Concise": "Keep your answer brief and to the point. Don't include more detail than necessary. Bullet points are helpful.",
        "Detailed": "Provide a thorough explanation with examples.",
        "Judge-Level": "Give a comprehensive answer with rule citations and step-by-step breakdowns."
    }.get(response_style, "Provide a detailed explanation.")
    
    system_prompt = f"""You are an expert Magic: The Gathering judge assistant with deep knowledge of MTG rules, card interactions, and tournament rulings. 

**Format Context:** {format_context}
**Response Style:** {style_instruction}

**Your expertise includes:**
- Comprehensive understanding of MTG Comprehensive Rules
- Knowledge of card interactions and edge cases
- Familiarity with tournament rulings and judge decisions
- Understanding of different formats (Standard, Modern, Legacy, Commander, etc.)

**When answering:**
1. Use precise MTG terminology and rule citations
2. Explain the reasoning behind your answer
3. Reference specific rules when possible (e.g., "Rule 702.12b states...")
4. If card text is provided, reference it in your explanation
5. Be concise but thorough
6. Use a friendly, helpful tone
7. If the question involves timing or the stack, explain the sequence clearly
8. For complex interactions, break down the steps

**Format your answers with:**
- Clear explanations
- Rule citations when relevant
- Step-by-step breakdowns for complex interactions
- Examples when helpful"""

    # Build the user message with MTG context
    user_message = question
    
    # Core MTG rules context to help the AI provide accurate responses
    mtg_context = """
**Key MTG Concepts to Consider:**
- State-based actions happen before any player gets priority
- The stack resolves last-in, first-out (LIFO)
- Protection prevents damage, enchanting, blocking, and targeting
- Indestructible prevents destruction but not other ways of leaving the battlefield
- Replacement effects can modify how events occur
- Triggered abilities use "when," "whenever," or "at"
- Activated abilities have costs and effects separated by colons
"""
    
    # Construct the complete user message with context and question
    user_message = mtg_context + "\n\n**Question:** " + question
    
    # Add card context if available (for future Scryfall integration)
    if card_context:
        user_message += f"\n\n**Relevant card text:**\n{card_context}"

    try:
        # Dynamically adjust OpenAI parameters based on response style
        # This optimizes cost and response quality for different detail levels
        if response_style == "Extra-Concise":
            max_tokens = 300  # Very short responses
            temperature = 0.1  # Very consistent
        elif response_style == "Concise":
            max_tokens = 600  # Brief but complete
            temperature = 0.2  # Consistent with some variety
        elif response_style == "Detailed":
            max_tokens = 1200  # Thorough explanations
            temperature = 0.3  # Balanced consistency and creativity
        else:  # Judge-Level
            max_tokens = 2000  # Comprehensive with citations
            temperature = 0.2  # Consistent for accuracy
        
        # Make the API call to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        
        # Handle specific OpenAI API errors with user-friendly messages
        if "insufficient_quota" in error_msg or "429" in error_msg:
            return "❌ **API Quota Exceeded**\n\nYou've reached your OpenAI API usage limit. Please:\n\n1. **Check your billing** at https://platform.openai.com/account/billing\n2. **Add credits** to your account\n3. **Try again later**\n\nThis is a billing issue, not a problem with the app."
        elif "invalid_api_key" in error_msg:
            return "❌ **Invalid API Key**\n\nPlease check your OpenAI API key in the .env file."
        else:
            return f"❌ **Error getting response:** {error_msg}"

def extract_card_names(text: str) -> list[str]:
    """
    Extract potential card names from text using simple heuristics.
    
    This function uses regex patterns to identify potential MTG card names
    in user questions. It's designed to work with the Scryfall API for
    future card lookup functionality.
    
    Note: This is a basic implementation. For production use, consider:
    - Using a proper MTG card database
    - Implementing fuzzy matching
    - Adding context-aware detection
    
    Args:
        text: The input text to search for card names
        
    Returns:
        list[str]: List of potential card names (limited to 5)
    """
    # Regex pattern to find capitalized words/phrases that might be card names
    # Matches: "Lightning Bolt", "Black Lotus", "Force of Will", etc.
    words = re.findall(r'\b[A-Z][a-zA-Z\s\-\']+\b', text)
    
    # Filter out common words and short names to reduce false positives
    common_words = ['the', 'and', 'or', 'but', 'for', 'with', 'from', 'into', 
                   'during', 'including', 'until', 'against', 'among', 
                   'throughout', 'despite', 'towards', 'upon']
    
    filtered = [word.strip() for word in words 
               if len(word.strip()) > 2 and word.strip().lower() not in common_words]
    
    return filtered[:5]  # Limit results to prevent API spam

# AUTHENTICATION

def check_password():
    """
    Simple password protection for controlled access to the app.
    
    This function implements basic password authentication using Streamlit's
    session state and secrets management. The password is checked against
    Streamlit secrets (for deployment) or a default value (for development).
    
    Security features:
    - Password input is hidden (type="password")
    - Password is deleted from session state after verification
    - Uses Streamlit secrets for secure storage in production
    
    Returns:
        bool: True if password is correct, False otherwise
    """
    def password_entered():
        """Check if the entered password matches the stored secret."""
        # Get password from Streamlit secrets (deployment) or use default (development)
        correct_password = st.secrets.get("password", "mtg2024")
        
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Security: Don't store password in session
        else:
            st.session_state["password_correct"] = False

    # Show password input on first visit
    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Show password input with error message for incorrect attempts
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password is correct, allow access
        return True

def main():
    """Main Streamlit app function."""
    
    # Check authentication first
    # if not check_password():
    #     st.stop()
    
    # Initialize OpenAI client
    client = init_openai_client()
    
    # Header
    st.title("🃏 MTG Rules Assistant")
    st.markdown("Ask any Magic: The Gathering rules question and get a judge-like response powered by GPT-4o!")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("This assistant uses GPT-4o to provide accurate MTG rules guidance.")
        
        # Format selection
        st.subheader("🎮 Format")
        format_type = st.selectbox(
            "Select format for context:",
            ["Any Format", "Standard", "Modern", "Legacy", "Commander", "Limited"]
        )
        
        # Response style
        st.subheader("📝 Response Style")
        response_style = st.selectbox(
            "How detailed should the response be?",
            ["Extra-Concise", "Concise", "Detailed", "Judge-Level"]
        )
        
        st.markdown("---")
        st.markdown("**Future Features:**")
        st.markdown("- Card text lookup via Scryfall")
        st.markdown("- Rule citation links")
        st.markdown("- Question history")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Question input
        st.subheader("📝 Ask Your Question")
        question = st.text_area(
            "Enter your MTG rules question:",
            placeholder="e.g., How does indestructible interact with -1/-1 counters?",
            height=120
        )
        
        # Submit button
        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("🤔 Thinking like a judge..."):
                    # Reinitialize client for each request to avoid caching issues
                    fresh_client = init_openai_client()
                    
                    # For now, we'll skip card lookup but leave the structure
                    card_context = None
                    
                    # Get GPT response with settings
                    response = get_gpt_response(fresh_client, question, card_context, format_type, response_style)
                    
                    # Display response
                    st.subheader("📋 Answer")
                    st.markdown(response)
                    
                    # Optional: Show detected card names (for future enhancement)
                    detected_cards = extract_card_names(question)
                    if detected_cards:
                        st.info(f"💡 Detected potential card names: {', '.join(detected_cards)}")
                        st.caption("Card lookup feature coming soon!")
            else:
                st.warning("Please enter a question.")
    
    with col2:
        # Example questions
        st.subheader("💡 Example Questions")
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
        
        # Handle example selection
        if 'example_question' in st.session_state:
            question = st.session_state.example_question
            del st.session_state.example_question
    
    # Footer
    st.markdown("---")
    st.caption("⚖️ This assistant provides guidance but is not a substitute for official MTG rules or judge rulings.")

if __name__ == "__main__":
    main() 