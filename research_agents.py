import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from typing import Optional, Any
from dotenv import load_dotenv, find_dotenv
from tavily import TavilyClient

load_dotenv(find_dotenv())
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("Error: TAVILY_API_KEY not found in .env file. Please set it in your .env file.")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


@function_tool
def citation_tool(self, query: str) -> str:
    # Placeholder for a citation tool implementation
    return f"Citations for {query}: [1] Example Source A, [2] Example Source B"


@function_tool
def synthesis_tool(self, query: str) -> str:
    # Placeholder for a citation tool implementation
    return f"Citations for {query}: [1] Example Source A, [2] Example Source B"


@function_tool
def reporter_tool(self, query: str) -> str:
    # Placeholder for a citation tool implementation
    return f"Citations for {query}: [1] Example Source A, [2] Example Source B"


@function_tool
def web_research_tool(self, query: str) -> str:
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


class ResearchAgent:
    def __init__(self, external_client: AsyncOpenAI):
        self._external_client = external_client
        self._agent: Optional[Agent] = None
        self.create_web_research_agent()

    def create_web_research_agent(self) -> Agent:
        if self._agent is None:
            llm_model = OpenAIChatCompletionsModel(
                model="gemini-1.5-flash",
                openai_client=self._external_client
            )

            self._agent = Agent(
                name="Deep Research Agent",
                instructions=(
                    "You are a researcher agent. Your job is to use the `tavily_deep_search` tool to find the most relevant and up-to-date information on a given topic. "
                    "Provide a detailed summary of your findings."
                ),
                tools=[web_research_tool],
                model=llm_model
            )
        return self._agent

    def run_main_research_agent(self, planning_output: Any):
        print("🚀 Research Agent has started!")
        print("-----------------------------------------------------\n")
        try:
            # 1. Run the Researcher Agent
            research_result = Runner.run_sync(self._agent, planning_output)
            print(f"[{self._agent.name} Response] ")
            print(research_result.final_output)
            print("--------------------------------\n")

        except Exception as e:
            print(f"❌ [Agent Error] An unexpected error occurred during the {self._agent} agent's execution: {e}")