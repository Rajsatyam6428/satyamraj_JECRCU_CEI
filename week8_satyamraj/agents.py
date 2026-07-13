from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
import os


class PlannerAgent:
    def __init__(self, api_key=None):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a planner agent. Your task is to create a clear, step-by-step plan to achieve the user's goal. Make sure the plan is actionable and specific."),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def plan(self, goal):
        messages = [
            HumanMessage(content=f"Create a step-by-step plan to achieve this goal: {goal}")
        ]
        chain = self.prompt | self.llm
        return chain.invoke({"messages": messages}).content


class ExecutorAgent:
    def __init__(self, api_key=None):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an executor agent. Your task is to execute the given plan step by step and provide a summary of the results."),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def execute(self, plan):
        messages = [
            HumanMessage(content=f"Execute this plan and provide results: {plan}")
        ]
        chain = self.prompt | self.llm
        return chain.invoke({"messages": messages}).content


class ReviewerAgent:
    def __init__(self, api_key=None):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a reviewer agent. Your task is to review the executed results, check if the goal is achieved, and suggest improvements if needed."),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def review(self, goal, execution_result):
        messages = [
            HumanMessage(content=f"Review these results for the goal: '{goal}'. Results: {execution_result}")
        ]
        chain = self.prompt | self.llm
        return chain.invoke({"messages": messages}).content
