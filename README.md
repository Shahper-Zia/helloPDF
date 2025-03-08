# PDF Chat Application

CHAT WITH GIVEN PDF USING LLM'S

## Project Structure
```
pdf-chat-app/
├── src/
│   ├── core/
│   │   ├── pdf_processor.py   # PDF processing and extraction
│   │   └── query_handler.py   # Query handling and response generation
│   ├── utils/
│   │   └── helpers.py         # Utility functions
│   ├── config/
│   │   └── settings.py        # Configuration settings
│   └── main.py               # Main application entry point
├── tests/                    # Unit tests
├── data/
│   └── sample_pdfs/         # Sample PDF storage
└── requirements.txt         # Project dependencies
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd helloPDF
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your API keys and configurations:
```env
GOOGLE_API_KEY=your_api_key_here
```

5. Run the application:
```bash
chainlit run src/main.py -w
```

The application will be available at `http://localhost:8000`