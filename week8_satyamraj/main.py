from agents import PlannerAgent, ExecutorAgent, ReviewerAgent
from dotenv import load_dotenv
import os


def main():
    # Load environment variables
    load_dotenv()
    
    print("=" * 50)
    print("AGENTIC AI PIPELINE")
    print("=" * 50)
    
    # Get user input
    goal = input("\nEnter your goal: ")
    
    # Initialize agents
    planner = PlannerAgent()
    executor = ExecutorAgent()
    reviewer = ReviewerAgent()
    
    # Run the pipeline
    print("\n--- Step 1: Planning ---")
    plan = planner.plan(goal)
    print(plan)
    
    print("\n--- Step 2: Executing ---")
    execution_result = executor.execute(plan)
    print(execution_result)
    
    print("\n--- Step 3: Reviewing ---")
    review = reviewer.review(goal, execution_result)
    print(review)
    
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    main()
