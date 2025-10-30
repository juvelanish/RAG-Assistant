import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
# from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
import os 
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import AIMessage,HumanMessage,SystemMessage
from colorama import Fore,init
from PyPDF2 import PdfReader
# Load environment variables
load_dotenv()


def load_documents() -> List[str]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """
    results = []
    dir = Path("data")
    for docs in dir.glob("*.*"):
         r_path = f"data/{docs.name}"
         if ".txt" in docs.name:
             
             with open(r_path,"r") as doc:
                 content = doc.read()
         
         elif ".pdf" in docs.name:
                reader =PdfReader(r_path)   
                content = []
                for i, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text:  # only include pages with text
                     content.append({
                             "page_number": i,
                            "text": text.strip()
                        })
         else:
             init(autoreset=True)
             print("\n" + Fore.RED + "Unsupported file found " + docs.name)
         results.append({"content":content,"name":docs.name})
    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )
        #memory for general chats outside knowledge base
        self.messages = [SystemMessage("You are an AI assistant designed to be helpful by answering questions from your own knowledge base. Be simple and direct")]



        # Initialize vector database
        self.vector_db = VectorDB()
        with open("./prompt.txt","r") as f:
            prompt = f.read().replace("$query","{query}").replace("$content$","{content}")
        self.prompt_template = ChatPromptTemplate.from_template(prompt)
        
        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()
      
        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
        context = self.vector_db.search(input)
        llm_answer = ""
        
    
        for inx in range(len(context["documents"][0][0])):
            data = llm_answer + context["documents"][0][0][inx] + '\n' + "metadata:" + str(context["metadatas"][0][0][inx])

        llm_answer = self.chain.invoke({"query":input,"content":data})   
        return llm_answer
    
        
    def chat(self,query):
        """For general questions
        Args:
           query: The question
        returns the llm's response from its own knowldge base"""
        reponse = ''
        self.messages.append(HumanMessage(query))
        reponse = self.llm.invoke(self.messages)
        self.messages.append(AIMessage(reponse.content))
        return reponse


def main():
      """Main function to demonstrate the RAG assistant."""
      try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        done = False
        init(autoreset=True)
        print("\n\n"+ Fore.YELLOW + "For general questions(Outside your custom knowledge base) use '/chat <question>'  ")
        while not done:
            print('\n \n')
            question = input(Fore.CYAN+"Enter a question or 'quit' to exit: ")
            if question.lower().strip() == "quit":
                done = True
            elif '/chat' in question.lower():
                print("\n")
                result = assistant.chat(question.replace("/chat",""))
                print(Fore.YELLOW + f"AI:{result.content}")
                print("\n")
                print("-"*134)

            else:
                print('\n')
                result = assistant.invoke(question)
                print(Fore.BLUE + f"AI:{result}")
                print("\n")
                print("-"*134)

      except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")

if __name__ == "__main__":
  main()