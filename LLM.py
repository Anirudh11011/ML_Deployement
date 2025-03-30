import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
import pandas as pd
import uuid
import chromadb



from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name="llama3-70b-8192"  # updated model name
)

# Load Portfolio
@st.cache_resource
def load_portfolio():
    df = pd.read_csv("my_portfolio.csv")
    client = chromadb.PersistentClient('vectorstore')
    collection = client.get_or_create_collection(name="portfolio")

    if not collection.count():
        for _, row in df.iterrows():
            collection.add(
                documents=row["Techstack"],
                metadatas={"links": row["Links"]},
                ids=[str(uuid.uuid4())]
            )
    return collection

# Define prompt templates
prompt_extract = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTION:
    The scraped text is from the career's page of a website.
    Your job is to extract the job postings and return them in JSON format containing the
    following keys: `role`, `experience`, `skills` and `description`.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE):
    """
)

prompt_email = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}

    ### INSTRUCTION:
    You are Virat, a business development executive at York. York is an AI & Software Consulting company dedicated to facilitating
    the seamless integration of business processes through automated tools.
    Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability,
    process optimization, cost reduction, and heightened overall efficiency.
    Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ
    in fulfilling their needs.
    Also add the most relevant ones from the following links to showcase York's portfolio: {link_list}
    Remember you are Mohan, BDE at AtliQ.
    Do not provide a preamble.
    ### EMAIL (NO PREAMBLE):
    """
)

# Streamlit UI
st.title("🚀 AI-Powered Job Email Generator")
job_url = st.text_input("Paste the job URL here:")
generate_button = st.button("Generate Email")

if generate_button and job_url:
    with st.spinner("Scraping and processing..."):
        # Load job content
        loader = WebBaseLoader(job_url)
        page_data = loader.load().pop().page_content

        # Extract job info
        chain_extract = prompt_extract | llm
        res = chain_extract.invoke(input={'page_data': page_data})
        job = JsonOutputParser().parse(res.content)

        # Query portfolio
        collection = load_portfolio()
        links = collection.query(query_texts=job['skills'], n_results=2).get('metadatas', [])
        link_list = links[0][0]["links"] if links and links[0] else ""

        # Generate email
        chain_email = prompt_email | llm
        email_response = chain_email.invoke({
            "job_description": str(job),
            "link_list": link_list
        })

    st.success("✅ Email generated!")
    st.text_area("Generated Email", email_response.content, height=300)
