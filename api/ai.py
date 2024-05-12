from langchain.document_loaders import CSVLoader
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.sql_database import SQLDatabase
from langchain.chains import SQLDatabaseSequentialChain, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from prompt import CHAT_PROMPT, COMBINED_TEMPLATE

import os
import pymysql
import pandas as pd

class CustomCallback(BaseCallbackHandler):
    def __init__(self, callback) -> None:
        self.callback = callback
        super().__init__()
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.callback(token)
    
    def on_llm_end(self, *args, **kwargs) -> None:
        self.callback(None)


class LocalRetrievalQAWithSQLDatabase:
    def __init__(
        self,
        openai_api_key: str,
        host: str,
        user: str,
        password: str,
        database: str,
        table_names: list[str],
        tmp_dir: str = "tmp",
        faiss_dir: str = "faiss_dir",
        index: str = "table_data",
        preload: bool = True,
    ):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table_names = table_names
        self.tmp_dir = tmp_dir
        self.faiss_dir = faiss_dir
        self.openai_api_key = openai_api_key
        self.index = index
        self.load_vector_store()

    def load_vector_store(self):
        self.faiss = FAISS.load_local(
            self.faiss_dir,
            OpenAIEmbeddings(openai_api_key=self.openai_api_key),
            index_name=self.index,
        )

    def get_docs_and_injest(self):
        docs = self.db_to_docs()
        self.injest_docs_to_faiss(docs)

    def add_docs_to_vectorstore(self, text: str, source: str):
        doc = Document(page_content=text, metadata={"source": source})
        doc = CharacterTextSplitter(chunk_size=2000).split_documents([doc])
        faiss = FAISS.load_local(
            self.faiss_dir,
            OpenAIEmbeddings(openai_api_key=self.openai_api_key),
            index_name=self.index,
        )
        faiss.add_documents(doc)
        faiss.save_local(self.faiss_dir, self.index)
        self.load_vector_store()

    def db_to_docs(self) -> list[Document]:
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
        )
        documents = []
        for table in self.table_names:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            os.makedirs(self.tmp_dir, exist_ok=True)
            file_path = os.path.join(self.tmp_dir, table)
            df.to_csv(file_path)
            docs = CSVLoader(file_path=file_path).load_and_split()
            documents.extend(docs)
        return documents

    def injest_docs_to_faiss(
        self,
        docs: list[Document],
    ):
        faiss = FAISS.from_documents(
            documents=docs,
            embedding=OpenAIEmbeddings(openai_api_key=self.openai_api_key),
        )
        faiss.save_local(self.faiss_dir, self.index)

    def make_sql_chain(
        self,
    ):
        db = SQLDatabase.from_uri(
            f"mysql+pymysql://{self.user}:{self.password}@{self.host}/{self.database}",
            include_tables=self.table_names,
        )
        llm = ChatOpenAI(
            temperature=0,
            verbose=True,
            openai_api_key=self.openai_api_key,
            request_timeout=20,
        )
        return SQLDatabaseSequentialChain.from_llm(llm, database=db, verbose=True)


    def chat(
        self,
        prompt: str,
        sql: bool = False,
        chat_history: list = None,
        mmr: bool = False,
        stream: bool = False,
        callback_func: callable = print
    ):
        if chat_history is None:
            chat_history = []

        convo = self.format_messages(chat_history, 700)
        combined = convo + "\n" + "Human: " + prompt
        combo_chain = LLMChain(
            llm=ChatOpenAI(
                temperature=0,
                openai_api_key=self.openai_api_key,
                request_timeout=20,
                model="gpt-3.5-turbo-16k",
            ),
            prompt=COMBINED_TEMPLATE,
            verbose=True,
        )
        combined = combo_chain.run(convo=combined)

        if not mmr:
            docs = self._reduce_tokens_below_limit(
                self.faiss.similarity_search(combined), 3500
            )
        else:
            docs = self._reduce_tokens_below_limit(
                self.faiss.max_marginal_relevance_search(combined), 3500
            )

        if sql:
            try:
                sql_chain = self.make_sql_chain()
                sql_result = sql_chain.run(combined)
            except Exception:
                sql_result = ""
        else:
            sql_result = ""


        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=self.openai_api_key,
            request_timeout=20,
            model="gpt-3.5-turbo-16k",
            streaming=stream,
            callbacks=[CustomCallback(callback_func)]
        )
        chain = load_qa_with_sources_chain(
            llm, chain_type="stuff", verbose=True, prompt=CHAT_PROMPT
        )
        return chain(
            {
                "input_documents": docs,
                "question": prompt,
                "sql": sql_result,
                "convo": convo,
            }
        )["output_text"]
        
        

    def _reduce_tokens_below_limit(self, docs: list, docs_token_limit: int):
        num_docs = len(docs)
        tokens = [len(doc.page_content) for doc in docs]
        token_count = sum(tokens[:num_docs])
        while token_count > docs_token_limit:
            num_docs -= 1
            token_count -= tokens[num_docs]
        return docs[:num_docs]

    def format_messages(
        self,
        chat_history: list[tuple[str, str]],
        tokens_limit: int,
        human_only: bool = False,
    ) -> str:
        chat_history = [
            (f"Human: {history[0]}", f"NettoChatBot: {history[1]}")
            for history in chat_history
        ]
        tokens_used = 0
        cleaned_msgs = []
        for history in reversed(chat_history):
            if not human_only:
                tokens_used += len(history[0]) + len(history[1])
            else:
                tokens_used += len(history[0])

            if tokens_used > tokens_limit:
                break

            if human_only:
                cleaned_msgs.append(history[0])
            else:
                cleaned_msgs.append((history[0], history[1]))

        if not human_only:
            return "\n\n".join(
                reversed(
                    [clean_msg[0] + "\n\n" + clean_msg[1] for clean_msg in cleaned_msgs]
                )
            )
        else:
            return "\n\n".join(reversed(cleaned_msgs))

