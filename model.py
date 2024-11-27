from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

class RAGPDFBot:

    def __init__(self):
        load_dotenv()
        self.file_path=""
        self.user_input=""
        self.sec_id=os.getenv("HUGGINGFACE_ACCESS_TOKEN")
        self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    def build_vectordb(self,chunk_size,overlap,file_path):
        loader = PyPDFLoader(file_path=file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=overlap)
        self.index = VectorstoreIndexCreator(embedding=HuggingFaceEmbeddings(),text_splitter=text_splitter).from_loaders([loader])

    def load_model(self,n_threads,max_tokens,repeat_penalty,n_batch,top_k,temp):
        callbacks = [StreamingStdOutCallbackHandler()]

        self.llm = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            max_length=128,
            temperature=0.5,
            huggingfacehub_api_token=self.sec_id,
            callbacks=callbacks,
            verbose=True
        )
        
    def retrieval(self,user_input,top_k=1,context_verbosity = False,rag_off=False):
        self.user_input = user_input
        self.context_verbosity = context_verbosity
        result = self.index.vectorstore.similarity_search(self.user_input,k=top_k)
        context = "\n".join([document.page_content for document in result])

        if self.context_verbosity:
            print(f"Retrieving information related to your question...")
            print(f"Found this content which is most similar to your question:{context}")



        template="""Context:{context}

            Instructions for the LLM:
                1.Based on the provided context, answer the question in a precise and accurate manner.
                2.If the question is directly related to the context, provide a clear and concise response.
                3.Ensure the answer does not exceed 3 lines.
                4.If the question is unrelated to the context, respond with "I don't know."
                5.Avoid any irrelevant or nonsensical information in the answer.

                Question:
                {question}
            """
        
        
        # """
        #         You are an assistant for question-answering tasks. Use the following context to answer the question and the answer should not be outside the context or with your knowledge. If the question is not realted to the context, just say that you don't know as the response. Use three sentences maximum and keep the answer concise.
        #         {context}
        #         Answer the following question:  
        #         {question}"""
            # """Dont't just repeat  the following context, use it in conbination with your knowledge to improve your answer to the question and don't deviate from the question: {context}
            # Question: {question}
            # """
            
        self.prompt = PromptTemplate(template=template,input_variables=["context","question"]).partial(context=context)

    def inference(self):
        if self.context_verbosity:
            print(f"Your Query: {self.prompt}")
        
        llm_chain = self.prompt | self.llm
        print(f"Processing the information...\n")
        response =llm_chain.invoke({"question": self.user_input})
        print(response)
        return response