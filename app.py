%%writefile app.py
#for streamlit
import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
import chromadb
import openai
import streamlit as st

# Define the conversation prompt template
CONDENSE_QUESTION_PROMPT = """
Assist Radeema's customers and prospects.
You are RadeeBot, should respond in the language the user uses.
Analyze the provided PDF document and use relevant information to advise and assist the user.
Keep it formal, and professional.
Prioritize understanding the user's intent, and ask for clarifications or additional details if needed.

If a question pertains to the documents required for specific connections (such as social connection or isolated building connection), refer to the specific section of the provided PDF that pertains to these procedures. Provide a detailed list of required documents for each type of connection based on the information available in the PDF.

If a question is unclear or complex, engage in multi-turn interactions to gather more information.

Offer alternative solutions or tips if the user's issue cannot be resolved directly.
If you cannot answer a question, recommend the user to contact support via email or phone 080 2000 123 mail adress: service.client@radeema.ma.
When referring to the procedures guide or required documents, provide the link to the 'guide des procédures commerciales' at https://urlzs.com/oMaKu


Avoid discussions not related to Radeema. You should not tell jokes, or engage in random discussions.

You are made for Radeema, a company based in Marrakech that manages water and electricity distribution. It was launched in July 1964 and took over the management of liquid sanitation services in January 1998.

Stay focused on Radeema's mission, values, products, history, culture, and staff.
If needed, suggest the Commercial Procedures Guide document via this link https://urlzs.com/oMaKu
{chat_history}
{question}
"""

api_key = os.getenv('OPENAI_API_KEY')

client = chromadb.Client()

class ChatVectorDBChain:
    def __init__(self, index, prompt, llm):
        self.index = index
        self.prompt = prompt
        self.llm = llm
        self.chat_history = []

    def __call__(self, user_input):
        # Update chat history
        self.chat_history.append({"role": "user", "content": user_input})

        # Generate the prompt
        prompt = self.prompt.template_str.format(chat_history=self.chat_history, question=user_input)

        # Use the OpenAI language model to generate a response
        response = self.llm.generate([prompt])

        # Extract the text from the response
        response_text = response.generations[0][0].text.strip()  # Update this line based on the actual structure of the response

        # Update chat history
        self.chat_history.append({"role": "assistant", "content": response_text})

        return response_text

class PromptTemplate:
    def __init__(self, template_str, index=None):
        self.template_str = template_str
        self.index = index

client = chromadb.Client()

collection = client.create_collection("conversation_context")

def create_pdf_loaders(pdf_folder_name):
    loaders = []
    for i in os.listdir(pdf_folder_name):
        file_path = os.path.join(pdf_folder_name, i)
        if os.path.isfile(file_path):
            loader = UnstructuredPDFLoader(file_path)
            loaders.append(loader)
    return loaders

pdf_folder_name = '/content/PDF'
loaders = create_pdf_loaders(pdf_folder_name)

index_creator = VectorstoreIndexCreator()
index = index_creator.from_loaders(loaders)

prompt = PromptTemplate(template_str=CONDENSE_QUESTION_PROMPT)
chain = ChatVectorDBChain(index, prompt, OpenAI(temperature=0))

# Streamlit application starts here
st.title("Radeema Chatbot")
user_input = st.text_input("Your question:")

if user_input:
    if user_input.lower() in ['exit', 'quit', 'stop']:
        st.write("Ending the conversation.")
    else:
        response = chain(user_input)
        st.write(f"Chatbot: {response}")
