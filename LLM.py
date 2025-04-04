import streamlit as st
import pandas as pd
import uuid
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, OutputParserException
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from bs4 import BeautifulSoup

# Initialize the LLM (update with your API key and model as required)
llm = ChatGroq(
    temperature=0,
    groq_api_key='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',  # Replace with your key
    model_name="llama-3.3-70b-versatile"
)

# Function to clean HTML by removing image tags
def remove_images(html):
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.find_all("img"):
        img.decompose()
    return soup.get_text()

# Cache the portfolio load to avoid reloading every time
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

# Define prompt templates for extraction and email generation
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

st.title("🚀 AI-Powered Job Email Generator")

job_url = st.text_input("Paste the job URL here:")

if st.button("Generate Email") and job_url:
    with st.spinner("Scraping and processing..."):
        # Instantiate the loader and load page data
        try:
            loader = WebBaseLoader(job_url)
            page_data = loader.load().pop().page_content
        except Exception as e:
            st.error("Error loading the webpage: " + str(e))
            st.stop()

        # Clean the page data to remove image tags
        cleaned_page_data = remove_images(page_data)
        
        # Check if cleaned_page_data is not empty
        if not cleaned_page_data.strip():
            st.error("No text content could be extracted from the webpage.")
            st.stop()
        
        # Extract job details using the extraction prompt
        chain_extract = prompt_extract | llm
        res = chain_extract.invoke(input={'page_data': cleaned_page_data})
        
        try:
            job = JsonOutputParser().parse(res.content)
        except OutputParserException as e:
            st.error("Error parsing job details from LLM output.")
            st.write("LLM output was:", res.content)
            st.stop()
        
        # Query portfolio and retrieve a single link
        collection = load_portfolio()
        links = collection.query(query_texts=job.get('skills', ''), n_results=2).get('metadatas', [])
        # Get the first link if available
        link_list = links[0][0]["links"] if links and links[0] else ""
        
        # Generate the email using the email prompt
        chain_email = prompt_email | llm
        email_response = chain_email.invoke({
            "job_description": str(job),
            "link_list": link_list
        })
    
    st.success("✅ Email generated!")
    st.text_area("Generated Email", email_response.content, height=300)
