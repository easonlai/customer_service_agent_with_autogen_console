# Customer Service Agent with AutoGen

## Overview
This project implements a customer service agent system using AutoGen. It consists of two agents:
1. **General Agent**: Handles common customer queries using a general knowledge base.
2. **Senior Agent**: Handles escalated or complex issues using a senior knowledge base.

The agents are powered by Azure OpenAI GPT models and utilize fuzzy matching to search knowledge bases stored in CSV files.

## Features
- **Knowledge Base Search**: Uses fuzzy matching to find relevant answers in the knowledge bases.
- **Escalation Mechanism**: Automatically escalates complex issues to the senior agent.
- **Group Chat Simulation**: Simulates interactions between the customer, general agent, and senior agent.
- **Customizable Configurations**: Easily configure API keys, endpoints, and deployment names.
- **Test Scenarios**: Predefined test cases to validate the functionality of the agents.

## Project Structure
- `customer_service_agents.py`: Main script containing the implementation of the agents and their functionalities.
- `general_agent.csv`: Knowledge base for the general agent.
- `senior_agent.csv`: Knowledge base for the senior agent.
- `requirements.txt`: List of dependencies required for the project.
- `readme.md`: Documentation for the project.

## Setup

### Prerequisites
- Python 3.8 or higher
- Azure OpenAI API credentials

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd customer_service_agent_with_autogen_public
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables for Azure OpenAI API:
   ```bash
   export AZURE_OPENAI_API_KEY="<your-api-key>"
   export AZURE_OPENAI_ENDPOINT="<your-endpoint>"
   ```

## Usage
1. Run the main script to test the functionality:
   ```bash
   python customer_service_agents.py
   ```
2. Modify the test scenarios in the script to simulate different customer queries.

## Knowledge Base Format
The knowledge bases (`general_agent.csv` and `senior_agent.csv`) should have the following columns:
- `Question`: The customer query.
- `Answer`: The corresponding answer.

## Dependencies
- `pandas`
- `fuzzywuzzy`
- `autogen`

## Logging
Logs are generated to track the system's behavior and errors. You can customize the logging level in the script.

## Agent Logic

### General Agent
The **General Agent** is designed to handle common customer queries using the general knowledge base. Its logic is as follows:
1. When a customer asks a question, the General Agent first uses the `retrieve_from_general_kb` function to search the general knowledge base for a relevant answer.
2. If a relevant answer is found (with a similarity score above 75), the agent provides the answer to the customer.
3. If no relevant answer is found or the question involves sensitive topics (e.g., safety, complaints, disputes, policy exceptions, or technical issues), the agent escalates the query to the Senior Agent by stating: "I need to escalate this to our senior team."
4. The General Agent does not attempt to answer sensitive topics directly.

### Senior Agent
The **Senior Agent** is responsible for handling escalated or complex customer queries. Its logic is as follows:
1. When a query is escalated, the Senior Agent first uses the `retrieve_from_senior_kb` function to search the senior knowledge base for a relevant answer.
2. If a relevant answer is found, the agent provides the answer to the customer.
3. If no relevant answer is found, the Senior Agent uses its expertise to analyze the situation and provide a comprehensive response. This includes addressing complaints, disputes, technical matters, safety concerns, or policy exceptions with professionalism.

### Escalation Mechanism
The escalation mechanism ensures that:
- The General Agent handles straightforward queries efficiently.
- Complex or sensitive queries are escalated to the Senior Agent for specialized handling.

### Group Chat Simulation
The system simulates a group chat environment where:
- The **User Proxy Agent** represents the customer and initiates queries.
- The **General Agent** and **Senior Agent** collaborate to provide answers based on their respective knowledge bases and expertise.
- The interaction is managed by a `GroupChatManager` to ensure smooth communication between agents.

## Test Scenarios
The script includes predefined test scenarios to validate the functionality of the agents. These scenarios simulate various customer queries, including:
1. Basic queries like store hours or loyalty program sign-up.
2. Escalation scenarios such as complaints about expired products or safety concerns.
3. Edge cases where neither knowledge base contains a relevant answer.

To test a scenario, uncomment the corresponding section in the `customer_service_agents.py` script and run it.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

