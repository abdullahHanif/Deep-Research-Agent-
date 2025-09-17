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
            """ [Agent Step: Tool Output] No relevant results found for this query."""
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


def run_multi_agent_team():
    """Main function to run the multi-agent team using OpenAI Agents SDK."""
    print("🚀 Multi-Agent Team Initialized and ready!")
    print("-----------------------------------------------------\n")

    llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",
        openai_client=external_client
    )

    researcher_agent = Agent(
        name="Researcher Agent",
        instructions=(
            "You are a researcher agent. Your job is to use the `tavily_deep_search` tool to find the most relevant and up-to-date information on a given topic. "
            "Provide a detailed summary of your findings."
        ),
        tools=[tavily_deep_search],
        model=llm_model
    )

    writer_agent = Agent(
        name="Writer Agent",
        instructions=(
            "You are a writer agent. Your job is to take the research findings from the Researcher Agent and write a clear, concise, and easy-to-understand summary for the user. "
            "Do not add any new information, only use what is provided in the research findings."
        ),
        model=llm_model
    )


    while True:
        user_query = input("💬 Enter your query for the agent team (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("👋 Exiting Multi-Agent Team. Goodbye!")
            break

        try:
            # 1. Run the Researcher Agent
            research_result = Runner.run_sync(researcher_agent, user_query)
            print(f"[{researcher_agent.name} Response] ")
            print(research_result.final_output)
            print("--------------------------------\n")

            # 2. Run the Writer Agent
            writer_prompt = f"Based on the following research, please write a summary for the user:\n\n{research_result.final_output}"
            final_result = Runner.run_sync(writer_agent, writer_prompt)
            print(f"[{writer_agent.name} Response] ")
            print(final_result.final_output)
            print("--------------------------------\n")


        except Exception as e:
            print(f"❌ [Agent Error] An unexpected error occurred during the agent's execution: {e}")

if __name__ == "__deep_research_system__":
    run_multi_agent_team()