import os
from dotenv import load_dotenv, find_dotenv
from tavily import TavilyClient
from agents import Agent, Runner, AsyncOpenAI, function_tool, OpenAIChatCompletionsModel


load_dotenv(find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


if not GEMINI_API_KEY:
    raise ValueError("Error: GEMINI_API_KEY not found in .env file. Please set it in your .env file.")

if not TAVILY_API_KEY:
    raise ValueError("Error: TAVILY_API_KEY not found in .env file. Please set it in your .env file.")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

@function_tool
def tavily_deep_search(query: str) -> str:
    """
    Performs a deep web search using Tavily and returns a summary of the results.
    This tool provides up-to-date information by searching the web comprehensively.
    """
    print(f"\n🔎 [Agent Step: Tool Call] Initiating deep search for: '{query}'...")
    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=10)

        if not response or not response.get('results'):
            """ [Agent Step: Tool Output] No relevant results found for this query.")"""
            return "No information found."

        search_results_summary = ""
        for i, result in enumerate(response['results']):
            search_results_summary += f"--- Source {i + 1}: {result.get('title', 'N/A')} ---\n"
            search_results_summary += f"URL: {result.get('url', 'N/A')}\n"
            search_results_summary += f"Snippet: {result.get('content', 'No snippet available.')}\n\n"

        return search_results_summary

    except Exception as e:
        print(f"❌ [Agent Step: Tool Error] An error occurred during Tavily deep search: {e}")
        return f"Error encountered during search: {str(e)}"


def run_deep_search_agent():
    """Main function to run the deep search agent using OpenAI Agents SDK."""
    print("🚀 Deep Search Agent Initialized and ready!")
    print("-----------------------------------------------------\n")

    llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=external_client
    )

    agent = Agent(
        name="Deep Search AI Assistant",
        instructions=(
            "You are an intelligent deep search assistant. Your primary role is to accurately answer user queries "
            "by leveraging the `tavily_deep_search` tool to get the most up-to-date and comprehensive information. "
            "Always use the `tavily_deep_search` tool when information beyond your training data is needed or when "
            "the user specifically asks for the latest information. "
            "After performing the search, synthesize the information into a clear, concise, and informative answer. "
            "Always provide the most relevant and latest information available from the search results."
        ),
        tools=[tavily_deep_search],
        model=llm_model
    )


    while True:
        user_query = input("💬 Enter your query for the agent (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("👋 Exiting Deep Search Agent. Goodbye!")
            break

        try:
            result = Runner.run_sync(agent, user_query)
            print(f"[{agent.name} Response] ")
            print(result.final_output)
            print("--------------------------------\n")

        except Exception as e:
            print(f"❌ [Agent Error] An unexpected error occurred during the agent's execution: {e}")

if __name__ == "__main__":
    run_deep_search_agent()
