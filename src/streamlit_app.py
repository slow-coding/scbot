import asyncio
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from simple_agent_copy import graph
from models import create_model

from langchain.schema.runnable.config import RunnableConfig

# llm = create_model("qwen2.5-32b-instruct")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Please enter your question")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):

        st_callback = StreamlitCallbackHandler(parent_container=st.container())

        async def on_message():

            progress_text = "Operation in progress. Please wait."
            progress_complete_text = "Operation complete."
            progress = 0

            chain_step = len(graph.nodes) - 1   # 0: __start__
            my_bar = st.progress(progress, text=progress_text)

            overall_output = ""

            async for event in graph.astream_events(
                input={"messages": st.session_state.messages},
                version="v2",
            ):

                statusContainer: st.StatusContainer
                if (
                    event["event"] == "on_chain_start"
                    and event["name"] != ("LangGraph" and "__start__")
                    and "langgraph_node" in event["metadata"]
                    and event["name"] == event["metadata"]["langgraph_node"]
                ):
                    statusContainer = st.status(
                        label=f"{event["name"]}: start thinking",
                        state="running",
                        expanded=False,
                    )

                if (
                    event["event"] == "on_chain_end"
                    and event["name"] != ("LangGraph" and "__start__")
                    and "langgraph_node" in event["metadata"]
                    and event["name"] == event["metadata"]["langgraph_node"]
                ):
                    with statusContainer as status:
                        if "content" in event["data"]["output"]["messages"][-1]:
                            st.markdown(
                                f"""{event["data"]["output"]['messages'][-1]["content"]}"""
                            )
                        else:
                            st.markdown(
                                f"""{event["data"]["output"]['messages'][-1].content}"""
                            )
                        status.update(
                            label=f"""{event["name"]} is done""",
                            state="complete",
                        )
                    progress += int(1 / chain_step * 100)
                    my_bar.progress(progress, text=progress_text)

                if event["event"] == "on_chain_end":
                    if not event["parent_ids"]:
                        overall_output = (
                            f"{event['data']['output']['messages'][-1].content}"
                        )
                        st.markdown(overall_output)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": overall_output}
                        )
                        my_bar.progress(100, text=progress_complete_text)

        asyncio.run(on_message())
