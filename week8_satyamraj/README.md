# Agentic AI Pipeline Project

This project includes a short quiz about agentic AI and a small multi-agent AI pipeline built with LangChain.

## Project Structure
```
celebal/
├── requirements.txt    # Python dependencies
├── quiz.md             # Short quiz about agentic AI
├── agents.py           # Multi-agent system (Planner, Executor, Reviewer)
├── main.py             # Main script to run the pipeline
├── .env.example        # Example environment variables file
└── README.md           # This file
```

## Setup Instructions

1. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key**
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to `.env`

3. **Run the pipeline**
   ```powershell
   python main.py
   ```

## Agents in the Pipeline
1. **Planner Agent**: Creates a step-by-step plan to achieve your goal
2. **Executor Agent**: Executes the plan and provides results
3. **Reviewer Agent**: Reviews the results and suggests improvements
