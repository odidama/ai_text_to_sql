import os
import streamlit as st
import mysql.connector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="My App", page_icon="🪼")

st.title("Hi! I'm geovac, happy to help...")
st.markdown("Quick info: Only sample Customers and Persons data are available for now. Some sample questions: Are "
            "there any customers from Canada? Do we have any customers named Emeka from Malaysia? etc.")


def connect_to_db():
    try:
        conn_string = os.getenv("SUPABASE_URI")
        engine = SQLDatabase.from_uri(conn_string, sample_rows_in_table_info=0)
        return engine
    except Exception as e:
        print(f"Error connecting to DB: {e}")
        return None


# def connect_to_db():
#     try:
#         host = '127.0.0.1'
#         port = st.secrets["DB_PORT"]
#         user = st.secrets["MYSQL_USER"]
#         password = st.secrets["MYSQL_PASSWORD"]
#         database = st.secrets["MYSQL_DATABASE"]
#         db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
#         # engine = create_engine(db_uri, echo=False)
#         engine = SQLDatabase.from_uri(db_uri, schema="geovac")
#         return engine
#     except mysql.connector.Error as err:
#         print(f"Error connecting to MySQL: {err}")
#     return None

db = connect_to_db()



def get_db_schema(_):
    return db.get_table_info()


def run_sql_query(query):
    print(f"Running SQL Query: {query} \n\n")
    return db.run(query)


def initiate_llm(load_from_hugging_face=False):
    if load_from_hugging_face:
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
            task="text-generation",
            temperature=0.7,
            max_new_tokens=200
        )
        return ChatHuggingFace(llm=llm)
    # return ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=st.secrets["GOOGLE_API_KEY"], temperature=0)


def write_sql_query(llm):
    template = """You are an expert in SQL query generation. Based on the table schema below, write an SQL query that 
    would answer the user's question: {schema}
    
    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given an input question, convert it to SQL query. No preamble"
                       "Do not return anything else apart from the SQL query. No prefix or suffix, no SQL keywords"
             ),
            ("user", template),
        ]

    )
    return (
            RunnablePassthrough.assign(schema=get_db_schema)
            | prompt
            | llm
            | StrOutputParser()
    )


def process_user_query(query, llm):
    template = """
    Based on the table schema below, question, sql query and sql response, write a natural language response:
    {schema}
    
    Question: {question}
    SQL Query: {query}
    SQL Response: {response}
    """

    prompt_response = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given an input question and SQL response, convert it to a natural language response.",
            ),
            ("user", template)

        ]
    )
    full_chain = (
            RunnablePassthrough.assign(query=write_sql_query(llm))
            | RunnablePassthrough.assign(schema=get_db_schema, response=lambda x: run_sql_query(x["query"]), )
            | prompt_response
            | llm
    )
    return full_chain.invoke({"question": query})


if "chart_history" not in st.session_state:
    st.session_state.chart_history = [
        AIMessage(content="Hello! I'm geovac, your SQL assistant."),
    ]

with st.sidebar:
    default_db = st.secrets["NEON_DATABASE_URL"]
    st.subheader("Settings")
    st.write("This is an ai chat application that interacts with a cloud DB and analyzes data through Natural "
             "Language. Login and experiment! ")
    st.text_input("Host:", value="MyDB", disabled=True, key="Host")
    st.text_input("Port:", value="1234", disabled=True, key="Port")
    st.text_input("User:", value="You", disabled=True, key="User")
    st.text_input("Password:", value="********", disabled=True, key="Password")
    st.text_input("Database:", value="Sample_DataBase", disabled=True, key="Database")

    if st.button("Login"):
        with st.spinner("Connecting to DB..."):
            # db = connect_to_db()
            st.session_state.db = db
            st.success("Connected to NeonDB!")

for message in st.session_state.chart_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

query = st.chat_input("Type your message here...")
if query is not None and query != "":
    st.session_state.chart_history.append(HumanMessage(content=query))

    with st.chat_message("Human"):
        st.markdown(query)

    with st.chat_message("AI"):
        response = process_user_query(query, llm=initiate_llm(load_from_hugging_face=False))
        # response = process_user_query(query, llm=initiate_llm(load_from_hugging_face=True))
        st.markdown(response.content)

    st.session_state.chart_history.append(AIMessage(content=response.content))
