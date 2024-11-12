from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

class RAGPDFBot:

    def __init__(self):
        self.file_path=""
        self.user_input=""
        self.sec_id="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"
        self.repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
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
            callbacks=callbacks
        )
        
    def retrieval(self,user_input,top_k=1,context_verbosity = False,rag_off=False):
        self.user_input = user_input
        self.context_verbosity = context_verbosity
        result = self.index.vectorstore.similarity_search(self.user_input,k=top_k)
        context = "\n".join([document.page_content for document in result])

        if self.context_verbosity:
            print(f"Retrieving information related to your question...")
            print(f"Found this content which is most similar to your question:{context}")

        if rag_off:
            template = """Question: {question}
            Answer: This is the response:
            """
            self.prompt = PromptTemplate(template=template,input_variables=["question"])
        else:
            template="""Dont't just repeat  the following context, use it in combination with your knowledge to improve just your answer to the question without any extra stuff and give a short answer if its a one word just give one word answer,dont give any extra question n answers just stick to the point: {context}
            Question: {question}
            """
            self.prompt = PromptTemplate(template=template,input_variables=["context","question"]).partial(context=context)

    def inference(self):
        if self.context_verbosity:
            print(f"Your Query: {self.prompt}")
        
        llm_chain = self.prompt | self.llm
        print(f"Processing the information...\n")
        response =llm_chain.invoke({"question": self.user_input})

        return response

