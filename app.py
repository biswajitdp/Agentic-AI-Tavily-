import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from dotenv import load_dotenv
import logging
import sys
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info(f"Application started at {datetime.now()}")
logger.info("="*80)

load_dotenv()
logger.info("Environment variables loaded")

# Configure Streamlit page
logger.info("Configuring Streamlit page settings")
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)
logger.info("Streamlit page configuration completed")

# Custom CSS
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    .main {
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        min-height: 100vh;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.75rem;
    }
    .title-section {
        text-align: center;
        margin-bottom: 3rem;
        color: white;
    }
    .input-section {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .response-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 2rem;
        color: white;
        margin-top: 2rem;
    }
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border-radius: 0.75rem !important;
        padding: 1rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
    }
    @keyframes blink {
        0%, 49% { opacity: 1; }
        50%, 100% { opacity: 0; }
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize LLM
logger.info("Initializing ChatOpenAI LLM")
llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
logger.info("ChatOpenAI LLM initialized successfully")

# Create the Tavily search tool using the decorator
logger.info("Creating Tavily search tool")
@tool
def tavily_search(query: str) -> str:
    """Search the internet for current information about a topic using Tavily Search. 
    Use this to find the latest news, events, and information about any topic."""
    logger.info(f"Tavily search initiated with query: {query}")
    try:
        tavily_tool = TavilySearchResults(max_results=5)
        logger.debug("Tavily tool instantiated")
        results = tavily_tool.run(query)
        logger.info(f"Tavily search completed successfully, results length: {len(str(results))}")
        return results
    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        logger.error(f"Tavily search failed: {error_msg}", exc_info=True)
        return error_msg
logger.info("Tavily search tool created")

# List of tools
tools = [tavily_search]
logger.info(f"Tools registered: {len(tools)} tool(s)")
for tool_obj in tools:
    logger.debug(f"  - Tool: {tool_obj.name}")

# Bind tools to the LLM
logger.info("Binding tools to LLM")
llm_with_tools = llm.bind_tools(tools)
logger.info("Tools successfully bound to LLM")

# Streaming display function - ChatGPT style
def stream_response(text: str, speed: float = 0.02):
    """Stream text response character by character like ChatGPT"""
    logger.info(f"Starting streaming response with speed: {speed}s per character")
    response_placeholder = st.empty()
    displayed_text = ""
    
    for char in text:
        displayed_text += char
        response_placeholder.markdown(f"""
        <div class="response-box">
            {displayed_text}<span style="animation: blink 1s infinite;">▌</span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(speed)
    
    # Final display without cursor
    response_placeholder.markdown(f"""
    <div class="response-box">
        {displayed_text}
    </div>
    """, unsafe_allow_html=True)
    logger.info("Streaming response completed")

def run_agent(user_input: str, status_placeholder):
    """Run the agentic AI loop with Streamlit integration"""
    logger.info(f"Agent started with user input: {user_input[:100]}...")
    
    messages = [
        SystemMessage(content="You are a helpful AI assistant. When asked questions, search for the latest information and provide comprehensive answers based on what you find."),
        HumanMessage(content=user_input)
    ]
    logger.debug(f"Initial messages count: {len(messages)}")
    
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Starting iteration {iteration}/{max_iterations}")
        status_placeholder.write(f"Processing... Iteration {iteration}")
        
        try:
            # Get response from LLM
            logger.debug(f"Invoking LLM with {len(messages)} messages")
            response = llm_with_tools.invoke(messages)
            logger.debug(f"LLM response received, tool_calls count: {len(response.tool_calls) if response.tool_calls else 0}")
            
            # Check if we need to use tools
            if not response.tool_calls:
                logger.info(f"Agent completed successfully at iteration {iteration}")
                logger.info(f"Final answer length: {len(response.content)} characters")
                status_placeholder.empty()
                return response.content
            
            # Add the assistant response to messages
            messages.append(response)
            logger.debug(f"Assistant message added, total messages: {len(messages)}")
            
            # Process tool calls
            for idx, tool_call in enumerate(response.tool_calls, 1):
                logger.info(f"Processing tool call {idx}: {tool_call['name']}")
                logger.debug(f"Tool call arguments: {tool_call['args']}")
                
                # Execute the tool
                if tool_call['name'] == 'tavily_search':
                    logger.debug("Executing Tavily search tool")
                    tool_result = tavily_search.invoke(tool_call['args'])
                else:
                    tool_result = f"Unknown tool: {tool_call['name']}"
                    logger.warning(f"Unknown tool encountered: {tool_call['name']}")
                
                logger.debug(f"Tool result length: {len(str(tool_result))} characters")
                
                # Add tool result to messages
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call['id']
                ))
                logger.debug(f"Tool result message added, total messages: {len(messages)}")
        
        except Exception as e:
            logger.error(f"Error during iteration {iteration}: {str(e)}", exc_info=True)
            status_placeholder.empty()
            return f"Error during processing: {str(e)}"
    
    logger.warning(f"Agent reached max iterations ({max_iterations}) without completion")
    status_placeholder.empty()
    return "Max iterations reached without completing the agent task"

# Main UI - Centered layout
st.markdown("""
<div class="title-section">
    <h1>🤖 AI Assistant</h1>
</div>
""", unsafe_allow_html=True)

# Input section
user_question = st.text_input(
    "Ask a question",
    placeholder="What would you like to know?",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    search_button = st.button("🔍 Search", type="primary")

# Response section
if search_button and user_question:
    logger.info(f"Search button clicked with question: {user_question[:100]}...")
    status_placeholder = st.empty()
    
    try:
        with st.spinner(""):
            logger.debug("Running agent")
            final_answer = run_agent(user_question, status_placeholder)
        
        logger.info("Agent execution completed successfully")
        logger.debug(f"Response displayed, answer length: {len(final_answer)} characters")
        
        # Stream the response like ChatGPT
        stream_response(final_answer, speed=0.01)
    
    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}", exc_info=True)
        st.error(f"Error: {str(e)}")

elif search_button and not user_question:
    logger.warning("Search button clicked but no question entered")
    st.warning("Please enter a question")

logger.debug("Streamlit UI rendering completed")
