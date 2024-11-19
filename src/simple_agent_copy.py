# ----------------------------------------
# Model
# ----------------------------------------
from models import create_model

model = create_model("qwen2.5-7b-instruct-mlx")

# ----------------------------------------
# Graph Builder (Build Time)
# ----------------------------------------
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph

graph_builder = StateGraph(MessagesState)

# ----------------------------------------
# Node
# ----------------------------------------
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig


# async def call_final_llm(state: MessagesState) -> MessagesState:
#     return {"messages": [model.invoke(state["messages"])]}


async def call_first_llm(state: MessagesState, config: RunnableConfig) -> MessagesState:
    messages = state["messages"]

    response = model.invoke(
        [
            SystemMessage(
                f"""
                
                """
            ),
            state["messages"][-1],
            HumanMessage("Translate the above context into Japanese"),
        ],
        config,
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Add node (agent)
graph_builder.add_node("first_llm", call_first_llm)  # Add node (agent)


async def call_second_llm(
    state: MessagesState, config: RunnableConfig
) -> MessagesState:
    messages = state["messages"]

    response = model.invoke(
        [
            SystemMessage(
                """
                
                        """
            ),
            *messages,  # flatten messages here by unpacking the list with *
            HumanMessage("Translate the above context into English"),
        ],
        config,
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Add node (agent)
graph_builder.add_node("second_llm", call_second_llm)  # Add node (agent)


async def call_third_llm(state: MessagesState, config: RunnableConfig) -> MessagesState:
    messages = state["messages"]
    response = model.invoke(
        [
            SystemMessage(
                """
                
                """
            ),
            *messages,  # flatten messages here by unpacking the list with *
            HumanMessage("Translate the above context into Chinese"),
        ],
        config,
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Add node (agent)
graph_builder.add_node("third_llm", call_third_llm)  # Add node (agent)
# ----------------------------------------
# Edge
# ----------------------------------------
from langgraph.graph import START
from langgraph.graph import END

graph_builder.set_entry_point("first_llm")
graph_builder.add_edge("first_llm", "second_llm")
graph_builder.add_edge("second_llm", "third_llm")
graph_builder.add_edge("third_llm", END)

# ----------------------------------------
# Graph (Compiled, ready to run)
# ----------------------------------------
graph = graph_builder.compile()

# ----------------------------------------
# Generate Graph PNG
# ----------------------------------------
# # with open("graph.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())
# print("PNG image saved as graph.png")
