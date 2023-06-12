import os
import pandas as pd
from typing import List, Dict, Any
import json
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import DataFrameLoader
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

class KvadratProfilesGPT():
    def __init__(self, openai_api_key, persist_directory) -> None:
        self.persist_directory = persist_directory
        self.openai_api_key = openai_api_key
        self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectordb = None
        self.qa = None

        try:
            self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
            self.qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type="stuff", vectorstore=self.vectordb)
        except Exception as e:
            print(f'failed loading existing profiles: {e}')


    def load_profiles(self, profiles_df):
        profiles_df = profiles_df.drop(columns=['preamble', 'article', 'competence_list', 'cv_list', 'employment_list', 'education_list'])
        loader = DataFrameLoader(profiles_df, page_content_column='content')
        documents = loader.load()

        texts_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = texts_splitter.split_documents(documents)

        self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding, persist_directory=self.persist_directory)
        self.vectordb.persist()

        self.qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=self.openai_api_key), chain_type="stuff", vectorstore=self.vectordb)

    def ask(self, query):
        return self.qa.run(query)

    def clean(self):
        self.vectordb.delete_collection()
        self.vectordb.persist()

if __name__ == '__main__':
    pwd = os.getcwd()
    profile_location = os.environ['KGPT_PROFILE_LOCATION']
    openai_api_key = os.environ['KGPT_OPENAI_API_KEY']
    chromadb_location = os.environ['KGPT_CHROMADB_LOCATION']

    kvadratProfilesGpt = KvadratProfilesGPT(openai_api_key, chromadb_location)

    # prep
    with open(profile_location, "r") as profile_json:
        profiles = json.load(profile_json)

    def build_profiles_content(profiles):
        for profile in profiles['data']:
            profile['content'] = json.dumps(profile)

        return profiles

    profiles = build_profiles_content(profiles)
    profiles_df = pd.DataFrame.from_dict(profiles['data'])

    # load
    #kvadratProfilesGpt.load_profiles(profiles_df)

    # ask
    answer = kvadratProfilesGpt.ask('role search')
    print(answer)

    # clean
    #kvadratProfilesGpt.clean()
