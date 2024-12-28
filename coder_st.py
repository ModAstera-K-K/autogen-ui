import asyncio
import os

import autogen
import streamlit as st
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

load_dotenv()

st.write("""# AutoGen Chat Agents""")

# Create a container for messages at the start
message_container = st.empty()

# Create a session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Update environment variables for Azure
azure_api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

config_list = [
    {
        "model": azure_deployment,
        "api_type": "azure",
        "azure_endpoint": azure_api_base,
        "api_key": azure_api_key,
        "api_version": "2023-07-01-preview",  # Update this as needed
    }
]

llm_config = {"config_list": config_list}


# After the initial imports, add this helper function
def scroll_to_bottom():
    js = """
        <script>
            function scroll() {
                var chatContainer = parent.document.querySelector('[data-testid="stChatMessageContainer"]');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }
            scroll();
        </script>
    """
    st.markdown(js, unsafe_allow_html=True)


class TrackableAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        # Add message to session state and update display
        st.session_state.messages.append(
            {"role": sender.name, "content": message}
        )
        with message_container.container():
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            scroll_to_bottom()
        return super()._process_received_message(message, sender, silent)


class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        # Add message to session state and update display
        st.session_state.messages.append(
            {"role": sender.name, "content": message}
        )
        with message_container.container():
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            scroll_to_bottom()
        return super()._process_received_message(message, sender, silent)


planner = TrackableAssistantAgent(
    name="planner",
    llm_config=llm_config,
    # the default system message of the AssistantAgent is overwritten here
    system_message="You are a helpful AI assistant. You suggest coding and reasoning steps for another AI assistant to accomplish a task. Do not suggest concrete code. For any action beyond writing code or reasoning, convert it to a step that can be implemented by writing code. For example, browsing the web can be implemented by writing code that reads and prints the content of a web page. Finally, inspect the execution result. If the plan is not good, suggest a better plan. If the execution is wrong, analyze the error and suggest a fix.",
)

planner_user = TrackableUserProxyAgent(
    name="planner_user",
    max_consecutive_auto_reply=0,  # terminate without auto-reply
    human_input_mode="NEVER",
    code_execution_config={
        "use_docker": True
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)


def ask_planner(message):
    planner_user.initiate_chat(planner, message=message)
    # return the last message received from the planner
    return planner_user.last_message()["content"]


# create an AssistantAgent instance named "assistant"
assistant = TrackableAssistantAgent(
    name="assistant",
    llm_config={
        "temperature": 0,
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
        "functions": [
            {
                "name": "ask_planner",
                "description": "ask planner to: 1. get a plan for finishing a task, 2. verify the execution result of the plan and potentially suggest new plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "question to ask planner. Make sure the question include enough context, such as the code and the execution result. The planner does not know the conversation between you and the user, unless you share the conversation with the planner.",
                        },
                    },
                    "required": ["message"],
                },
            },
        ],
    },
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = TrackableUserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",  # or "NEVER"
    # human_input_mode="NEVER",  # or "NEVER"
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: "content" in x
    and x["content"] is not None
    and x["content"].rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "planning__local",
        "use_docker": True,
    },
    function_map={"ask_planner": ask_planner},
)

with st.container():
    # Display existing messages
    with message_container.container():
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Type something...")
    if user_input:
        # Add user input to messages
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        if not azure_api_key or not azure_api_base or not azure_deployment:
            st.warning(
                "You must provide valid Azure OpenAI credentials (API base, key, and deployment name)",
                icon="⚠️",
            )
            st.stop()

        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Define an asynchronous function
        async def initiate_chat():
            await user_proxy.a_initiate_chat(
                assistant,
                message=user_input,
            )

        # Run the asynchronous function within the event loop
        try:
            loop.run_until_complete(initiate_chat())
        finally:
            loop.close()
