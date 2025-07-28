# 🃏 MTG Rules Assistant

A web-based Magic: The Gathering rules assistant powered by GPT-4o and Streamlit. Get judge-like responses to your MTG rules questions instantly!

## ✨ Features

- **GPT-4o Powered**: Get accurate, judge-like responses to MTG rules questions
- **Clean UI**: Simple, intuitive Streamlit interface
- **Example Questions**: Quick access to common rules questions
- **Future-Ready**: Code structure ready for Scryfall API integration

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Virtual environment (recommended)

### Installation

1. **Clone or download this project**
   ```bash
   cd mtg_chatbot
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key**
   
   **Option A: Environment variable (recommended)**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   **Option B: Create a .env file**
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 🎯 Usage

1. **Ask a Question**: Type your MTG rules question in the text area
2. **Get Answer**: Click "Get Answer" to receive a GPT-4o powered response
3. **Try Examples**: Use the example questions in the sidebar for quick testing

### Example Questions

- "How does indestructible interact with -1/-1 counters?"
- "Can I counter a spell that can't be countered?"
- "What happens when a creature with protection from red is targeted by a red spell?"
- "How do multiple instances of the same ability work?"

## 🏗️ Project Structure

```
mtg_chatbot/
├── app.py                 # Main Streamlit application
├── scryfall_helper.py     # Scryfall API integration (future use)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore           # Git ignore file
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Customization

You can modify the following in `app.py`:

- **System Prompt**: Change the judge assistant prompt in `get_gpt_response()`
- **Model Settings**: Adjust `max_tokens` and `temperature` in the OpenAI call
- **UI Layout**: Modify the Streamlit layout and styling

## 🚧 Future Enhancements

The code is structured to support these upcoming features:

### Phase 2: Scryfall Integration
- Automatic card name detection in questions
- Fetch card rules text from Scryfall API
- Include card context in GPT responses

### Phase 3: Advanced Features
- Question history and favorites
- Rule citation links
- Card image display
- Multi-language support

## 🛠️ Development

### Adding Scryfall Integration

To enable card lookup functionality:

1. **Import the helper module** in `app.py`:
   ```python
   from scryfall_helper import get_relevant_cards_context
   ```

2. **Modify the main function** to use card context:
   ```python
   # Extract card names from question
   detected_cards = extract_card_names(question)
   
   # Get card context if cards are detected
   card_context = None
   if detected_cards:
       card_context = get_relevant_cards_context(detected_cards)
   
   # Pass card context to GPT
   response = get_gpt_response(client, question, card_context)
   ```

### Testing

```bash
# Run with debug mode
streamlit run app.py --logger.level debug

# Run on different port
streamlit run app.py --server.port 8502
```

## ⚠️ Important Notes

### API Costs
- GPT-4o API calls incur costs based on usage
- Monitor your OpenAI usage in the OpenAI dashboard
- Consider implementing rate limiting for production use

### Rate Limits
- Scryfall API has rate limits (10 requests per second)
- The `scryfall_helper.py` includes built-in delays to respect limits

### Legal Disclaimer
This assistant provides guidance but is not a substitute for official MTG rules or judge rulings. Always consult official sources for tournament rulings.

## 🐛 Troubleshooting

### Common Issues

**"OPENAI_API_KEY environment variable not found"**
- Ensure your API key is set correctly
- Restart your terminal after setting the environment variable

**"Error getting response"**
- Check your OpenAI API key is valid
- Verify you have sufficient API credits
- Check your internet connection

**Import errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate your virtual environment

### Getting Help

1. Check the [Streamlit documentation](https://docs.streamlit.io/)
2. Review [OpenAI API documentation](https://platform.openai.com/docs)
3. Check [Scryfall API documentation](https://scryfall.com/docs/api)

## 📄 License

This project is for educational and personal use. Please respect OpenAI's and Scryfall's terms of service.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Happy gaming! 🎮** 