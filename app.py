import os
from flask import Flask, jsonify, request, render_template
from langchain.document_loaders import DirectoryLoader
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from langchain.chains import RetrievalQA


os.environ["OPENAI_API_KEY"] = "your open AI key"

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

docsearch = None
chain = None
rqa = None
directory = './data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  print(loader)
  documents = loader.load()
  return documents

def loadPDF():
    print("LOAD PDF")

    raw_text = ''
    documents = load_docs(directory)
    for document in documents:
        if document.page_content:
            raw_text +=  document.page_content
    print("raw_text : ", raw_text)

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200, #striding over the text
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)


    embeddings = OpenAIEmbeddings()

    global docsearch 
    docsearch = FAISS.from_texts(texts, embeddings)
    docsearch.embedding_function
    global chain 
    chain = load_qa_chain(OpenAI(), chain_type="stuff") # we are going to stuff all the docs in at once

    chain.llm_chain.prompt.template

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})

    global rqa 
    rqa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

    print("PDF LOADED")

def get_result(query):
    docs = docsearch.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    print("result : ", result)
    return result

@app.route('/api/data', methods=['GET'])
def get_data():
    text_param = request.args.get('text')
    data = {'Result': get_result(text_param)}
    return jsonify(data)

if __name__ == '__main__':
    loadPDF()
    app.run(host='0.0.0.0',debug=False, port=3003)
