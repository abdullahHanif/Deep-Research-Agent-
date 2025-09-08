from agents import Agent, Runner, AsyncOpenAI, function_tool, OpenAIChatCompletionsModel
from research_agents import ResearchAgent

def run_planning_agent(external_client: AsyncOpenAI , user_query: str):
    print("🚀 Planning Agent Initialized and ready!")
    print("-----------------------------------------------------\n")

    llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",
        openai_client=external_client
    )

    planning_agent = Agent(
        name="Deep search planner",
        instructions=(
            "You are planning agent in an advanced Research Agent designed to perform deep, structured research like a professional analyst, Your task is to break the user’s question into smaller, concrete research tasks. For Example: “Compare renewable energy policies in 3 countries” → Tasks: “Country A energy policy,” “Country B energy policy,” “Country C energy policy.”"
        ),
        model=llm_model
    )

    try:
        # 1. Run the Planning Agent
        planning_result = Runner.run_sync(planning_agent, user_query)
        print(f"[{planning_agent.name} Response] ")
        print(planning_result.final_output)
        print("--------------------------------\n")
        # 2. Run the Research Agent with the output from the Planning Agent
        research_agents = ResearchAgent(external_client)
        research_agents.run_main_research_agent(planning_result.final_output)


    except Exception as e:
        print(f"❌ [Agent Error] An unexpected error occurred during the {planning_agent.name} agent's execution: {e}")