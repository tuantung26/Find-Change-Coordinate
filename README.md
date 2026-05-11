# Geometry Assistant with LangChain and Gemini

This repository contains `run.py`, a Python script that leverages LangChain, Google's Gemini 2.5 Flash model, and Pydantic to solve geometric coordinate problems and execute multi-step movement instructions using AI tool calling.

## Features

- **Structured Data Parsing:** Uses Gemini's structured output capabilities (via Pydantic schemas) to break down complex, multi-part natural language queries into logical steps.
- **Geometry Solving:** Automatically solves geometry problems to find point coordinates while calculating mathematical properties like the Manhattan norm and Euclidean norm.
- **Tool Calling:** Employs custom LangChain tools (`move_x`, `move_y`) to let the LLM dynamically calculate distances and execute physical "movements" along coordinate axes.

## Prerequisites

- Python 3.8+
- A Google Gemini API Key

## Installation

1. **Install the required Python dependencies:**
   Run the following command to install LangChain, the Google GenAI integration, Pydantic, and python-dotenv.
   
   ```bash
   pip install langchain-core langchain-google-genai pydantic python-dotenv
   ```

2. **Environment Setup:**
   Create a `.env` file in the same directory as `run.py` and add your Google API key:
   
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

Execute the script from your terminal:

```bash
python run.py
```

### Example Workflow

Currently, the script processes a hardcoded complex user query:
> *"what is the coordinate of the point D satisfying that the quadrilateral ABCD is a rectangle where A(0, 0), B(0, 5), C(10, 5); and how to move the point D to the origin then move it to the point I which is the middle of A and B"*

When run, the script performs the following steps:

1. **Query Classification:** Splits the prompt into two distinct instructions (coordinate calculation vs. movement directions).
2. **Coordinate Resolution:** Calculates that point D must be `(10, 0)` to complete the rectangle, and extracts its norms.
3. **Execution:** Translates the movement intent into exact numeric values and calls the `move_x` and `move_y` tools to simulate translating the point to the origin `(0, 0)`, and then to the midpoint of A and B `(0, 2.5)`.

## Customization

To try out different geometry problems, you can modify the `user_query` variable inside `run.py` with your own coordinates and movement instructions.

## Built With

- LangChain - LLM framework
- Google Gemini API - Generative AI model
- Pydantic - Data validation
