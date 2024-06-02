from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_astradb import AstraDBVectorStore
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_core.tools import tool
from github_assistant import fetch_github_issues
from langchain_community.vectorstores import FAISS
from note import note_tool

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


add_to_vectorstore = input("Do you want to update issues? {y/N}: ").lower() in ["yes","y"]

if add_to_vectorstore:
    owner = "Dynamo-Dream"
    repo = "A_Star_ALgo"
    issues = fetch_github_issues(owner,repo)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = ["FAISS is an important library", "LangChain supports FAISS"]
    vector_store = FAISS.from_documents(issues,embedding=embeddings)
    vector_store.save_local("faiss_index")
    vector_store = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    results = vector_store.similarity_search("Flash message")


retreiver = vector_store.as_retriever(kwargs={"k":3})
retreiver_tool = create_retriever_tool(
    retreiver,
    "github_search",
    "Search for information about github issues. For any questions about github issues you must use this tool"
)

prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatGoogleGenerativeAI(model="gemini-pro")
tools = [retreiver_tool, note_tool]
llm = llm.bind_tools(tools)
agent = create_tool_calling_agent(llm,tools,prompt)
agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

while (question := "Ask a question about issue (q to quit) " != "q"):
    result = agent_executor.invoke({"input":question})
    print(result["output"])






    
    