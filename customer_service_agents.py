# Import necessary libraries
import os
import logging
import json
import pandas as pd
import traceback # Keep traceback for potential errors
from fuzzywuzzy import fuzz
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, GroupChat, GroupChatManager

# Disable Docker usage for local testing
os.environ["AUTOGEN_USE_DOCKER"] = "False"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load knowledge base CSV files
general_knowledge = pd.read_csv("general_agent.csv")
senior_knowledge = pd.read_csv("senior_agent.csv")

# Define Azure OpenAI API deployment details and credentials
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY") # Use env var if available
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://YOUR_AZURE_OPENAI_ENDPOINT.openai.azure.com") # Use env var if available
azure_deployment_general = "YOUR_GPT4O_DEPLOYMENT_NAME"
azure_deployment_senior = "YOUR_O3MINIM_DEPLOYMENT_NAME"
api_version = "2024-12-01-preview"

# Define LLM configurations for general agent
llm_config_general = {
    "model": azure_deployment_general,
    "api_type": "azure",
    "api_version": api_version,
    "api_key": AZURE_OPENAI_API_KEY,
    "base_url": AZURE_OPENAI_ENDPOINT,
}

# Define LLM configurations for senior agent
llm_config_senior = {
    "model": azure_deployment_senior,
    "api_type": "azure",
    "api_version": api_version,
    "api_key": AZURE_OPENAI_API_KEY,
    "base_url": AZURE_OPENAI_ENDPOINT,
}

#  Deifine the function to search the knowledge base
def search_kb(query, kb_dataframe):
    """Search a knowledge base dataframe for matching questions."""
    # Ensure inputs are strings
    query_str = str(query).lower()
    # Ensure 'Question' column exists and is string type
    if 'Question' not in kb_dataframe.columns:
        logging.error("Knowledge base CSV must contain a 'Question' column.")
        return None, 0
    kb_dataframe['Question'] = kb_dataframe['Question'].astype(str)

    similar_questions = kb_dataframe['Question'].apply(
        lambda x: fuzz.ratio(x.lower(), query_str)
    )
    # Handle case where similar_questions might be empty or all NaN
    if similar_questions.empty or similar_questions.isnull().all():
        return None, 0

    # Get the maximum score and the corresponding answer
    max_score = similar_questions.max()
    if pd.notna(max_score) and max_score > 75:
        # Ensure 'Answer' column exists
        if 'Answer' not in kb_dataframe.columns:
             logging.error("Knowledge base CSV must contain an 'Answer' column.")
             return None, max_score
        answer = kb_dataframe.loc[similar_questions.idxmax(), 'Answer']
        return answer, max_score
    return None, max_score if pd.notna(max_score) else 0

# ======= TOOL FUNCTIONS =======

# Define a class-based approach for function handling
class CustomerServiceTools:
    """Tools for the customer service agents to use."""

    # Retrieve information from the GENERAL knowledge base for common questions
    @staticmethod
    def retrieve_from_general_kb(query):
        """Retrieve information from the GENERAL knowledge base for common questions."""
        logging.info(f"Looking up in GENERAL KB: {query}")
        if isinstance(query, dict) and 'query' in query:
            query = query['query']

        answer, score = search_kb(query, general_knowledge)
        if answer:
            logging.info(f"Found general KB answer with score {score}")
            return str(answer)
        else:
            logging.info(f"No match found in general KB (best score: {score})")
            return "No answer found in general knowledge base."

    # Retrieve information from the SENIOR knowledge base for complex/escalated issues
    @staticmethod
    def retrieve_from_senior_kb(query):
        """Retrieve information from the SENIOR knowledge base for complex/escalated issues."""
        logging.info(f"Looking up in SENIOR KB: {query}")
        if isinstance(query, dict) and 'query' in query:
            query = query['query']

        answer, score = search_kb(query, senior_knowledge)
        if answer:
            logging.info(f"Found senior KB answer with score {score}")
            return str(answer)
        else:
            logging.info(f"No match found in senior KB (best score: {score})")
            # Important: Return a specific string so the senior agent knows the lookup failed
            return "No answer found in senior knowledge base."

# Initialize the tools
tools = CustomerServiceTools()

# ======= AGENT CONFIGURATION =======

# Define the user proxy agent that will simulate the customer
user_proxy = UserProxyAgent(
    name="customer",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=0,
    code_execution_config=False,
    # User proxy needs access to *both* KB search functions
    function_map={
        "retrieve_from_general_kb": tools.retrieve_from_general_kb,
        "retrieve_from_senior_kb": tools.retrieve_from_senior_kb,
    }
)

# Define function schema for the GENERAL KB tool
tool_schema_general_kb = {
    "type": "function",
    "function": {
        "name": "retrieve_from_general_kb",
        "description": "Search the GENERAL knowledge base for answers to common customer questions (store hours, basic returns, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's question to look up in the GENERAL knowledge base."
                }
            },
            "required": ["query"]
        }
    }
}

# Define function schema for the SENIOR KB tool
tool_schema_senior_kb = {
    "type": "function",
    "function": {
        "name": "retrieve_from_senior_kb",
        "description": "Search the SENIOR knowledge base for answers to complex or escalated issues (complaints, safety, policy exceptions, technical details).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's complex/escalated question to look up in the SENIOR knowledge base."
                }
            },
            "required": ["query"]
        }
    }
}

# Initialize the general agent with the LLM configuration
# Define the general agent with a specific system message
general_agent = AssistantAgent(
    name="general_agent",
    llm_config={
        **llm_config_general,
        "tools": [tool_schema_general_kb] # General agent only uses general KB tool
    },
    system_message="""
You are a general customer service agent for a retail store.

For EVERY customer question:
1. FIRST, use the `retrieve_from_general_kb` function to search the general knowledge base.
2. If you get a relevant answer (NOT 'No answer found...'), provide it to the customer.
3. If the general KB search returns 'No answer found...' OR the question involves sensitive topics (foreign objects, safety, complaints, disputes, policy exceptions, complex technical issues), state CLEARLY: 'I need to escalate this to our senior team.' and STOP your response.

Do NOT attempt to answer sensitive topics yourself. Only answer directly if the general KB provides a clear, relevant answer AND the topic is NOT sensitive.
""",
    function_map={
        "retrieve_from_general_kb": tools.retrieve_from_general_kb,
    }
)

# Initialize the senior agent with the LLM configuration
# Define the senior agent with a specific system message
# This agent handles complex issues escalated by the general agent
senior_agent = AssistantAgent(
    name="senior_agent",
    llm_config={
        **llm_config_senior,
        "tools": [tool_schema_senior_kb] # Senior agent uses senior KB tool
    },
    system_message="""
You are a senior customer service agent handling escalated issues.

For EVERY escalated question you receive:
1. FIRST, use the `retrieve_from_senior_kb` function to search the senior knowledge base for specific policies or procedures related to the complex issue.
2. If you find a relevant answer in the senior KB, provide that information to the customer.
3. If the senior KB search returns 'No answer found...', THEN use your expertise to analyze the situation and provide a comprehensive response, resolution, or clear next steps to the customer directly. Address complaints, disputes, technical matters, safety concerns, foreign objects, or policy exceptions with professionalism.
""",
    function_map={
        "retrieve_from_senior_kb": tools.retrieve_from_senior_kb,
    }
)

# ======= GROUP CHAT SETUP =======

# Create the GroupChat
groupchat = GroupChat(
    agents=[user_proxy, general_agent, senior_agent],
    messages=[],
    max_round=12
)

# Create the GroupChatManager
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config_general
)

# ======= TEST SCENARIOS =======

print("\n\n----- TESTING BASIC FUNCTIONALITY 1 (via GroupChat) -----\n")
user_proxy.initiate_chat(
    manager,
    message="What are your store hours?"
)

# print("\n\n----- TESTING BASIC FUNCTIONALITY 2 (via GroupChat) -----\n")
# user_proxy.initiate_chat(
#     manager,
#     message="How to sign up for loyalty program?"
# )

# print("\n\n----- TESTING ESCALATION SCENARIO 1 (via GroupChat) -----\n")
# user_proxy.initiate_chat(
#     manager,
#     message="Your store has expired product on the shelf. This is insane!"
# )

# print("\n\n----- TESTING ESCALATION SCENARIO 2 (via GroupChat) -----\n")
# user_proxy.initiate_chat(
#     manager,
#     message="what are my options for replacement or refund?"
# )

# print("\n\n----- TESTING ESCALATION SCENARIO 3 (via GroupChat) -----\n")
# user_proxy.initiate_chat(
#     manager,
#     message="I found a piece of plastic in my cereal box! This is unacceptable."
# )

# print("\n\n----- TESTING ESCALATION SCENARIO 4 (via GroupChat) -----\n")
# user_proxy.initiate_chat(
#     manager,
#     message="Can you tell me the exact quantum entanglement state of the manager's socks?" # KB miss for both agents
# )