from agents import Agent, Runner, AsyncOpenAI, function_tool, OpenAIChatCompletionsModel

def run_reporter_agent(external_client: AsyncOpenAI):
    print("🚀 Planning Agent Initialized and ready!")
    print("-----------------------------------------------------\n")

    llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",
        openai_client=external_client
    )

    reporter_agent = Agent(
        name="Deep search planner",
        instructions=(
            "You are a researcher agent. Your job is to use the `tavily_deep_search` tool to find the most relevant and up-to-date information on a given topic. "
            "Provide a detailed summary of your findings."
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
            research_result = Runner.run_sync(reporter_agent, user_query)
            print(f"[{reporter_agent.name} Response] ")
            print(research_result.final_output)
            print("--------------------------------\n")

        except Exception as e:
            print(f"❌ [Agent Error] An unexpected error occurred during the agent's execution: {e}")