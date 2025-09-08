import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, function_tool, OpenAIChatCompletionsModel

import planning_agent

load_dotenv(find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Error: GEMINI_API_KEY not found in .env file. Please set it in your .env file.")

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def run_multi_agent_system():
    while True:
        user_query = input("💬 Enter your query for the agent team (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("👋 Exiting Multi-Agent Team. Goodbye!")
            break

        if user_query.strip() == "":
            print("⚠️ Please enter a valid query.")
            continue

        try:
            # 1. Run the Planner Agent
            print("------- Planning to response -------\n")
            planning_agent.run_planning_agent(external_client, user_query)

        except Exception as e:
            print(f"❌ [Agent Error] An unexpected error occurred during the agent's execution: {e}")


if __name__ == "__main__":
    run_multi_agent_system()
