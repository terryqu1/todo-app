from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
import os
import openai
import csv
from langchain.document_loaders import WebBaseLoader, OutlookMessageLoader, UnstructuredEmailLoader
from langchain.indexes import VectorstoreIndexCreator
from flask import Flask, render_template

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('index.html', message="Sample message")

@app.route('/run-script')
def run_script():
    # Your Python code here
    # Retrieve the API key from the environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set!")

    # Set the API key for the OpenAI library
    openai.api_key = openai_api_key

    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.chat_models import ChatOpenAI
    from langchain.chains.summarize import load_summarize_chain
    import nltk

    # Define prompt
    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    #loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

    loader = UnstructuredEmailLoader(r"C:\Users\terry\Downloads\[Hinge] Re_ Date of Birth Change Request.eml", mode = "elements")
    docs = loader.load()

    # Define LLM
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.5)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Set up stuff chain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
    )

    #print(stuff_chain.run(docs))

    from langchain.chains.mapreduce import MapReduceChain
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain

    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes 
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of summaries:
    {doc_summaries}
    Take these and distill it into a final, consolidated summary of the main themes. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    s = map_reduce_chain.run(split_docs)
    print(s)
    return render_template('index.html', message=s)

if __name__ == "__main__":
    app.run()








# tools = load_tools(["serpapi", "llm-math"], llm=llm)
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# agent.run("Summarize the wikipedia article on learning")