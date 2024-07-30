import streamlit as st
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import pickle

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

from transformers import GPT2LMHeadModel, GPT2Tokenizer

#Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def get_natural_language_answer(question, answer):
    # Prepare the input text
    input_text = f"Rephrase the following answer naturally for the given question.\nQuestion: {question}\nAnswer: {answer}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model.generate(input_ids, max_new_tokens=70, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    natural_language_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    index_of_last_fullstop=len(natural_language_answer)-1
    if(natural_language_answer[-1]!='.'):
      for i in range(len(natural_language_answer)):
        if(natural_language_answer[i] == '.'):
          index_of_last_fullstop = i
    natural_language_answer = natural_language_answer[:index_of_last_fullstop+1]

    natural_language_answer=natural_language_answer.split()
    for i in range(len(natural_language_answer)):
        if(natural_language_answer[i] == 'Answer:'):
            natural_language_answer=natural_language_answer[i+1:]
            break
    natural_language_answer_str=''
    for i in natural_language_answer:
        natural_language_answer_str+=i+" "
    natural_language_answer_str=natural_language_answer_str.strip()

    return natural_language_answer_str

# Define paths for loading
pipeline_path = "qa_pipeline.pkl"
index_path = "faiss_index.bin"
knowledge_base_path = "knowledge_base_vectors.npy"
dataset_path = "question_answer_dataset.pkl"

# Load the dataset
df = pd.read_pickle(dataset_path)

# Initialize sentence transformer model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer("average_word_embeddings_glove.6B.300d", device=device)

# Load the QA pipeline
with open(pipeline_path, 'rb') as f:
    qa_pipeline_data = pickle.load(f)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_pipeline_data['model'])
qa_tokenizer = AutoTokenizer.from_pretrained(qa_pipeline_data['tokenizer'])
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

# Load the FAISS index
index = faiss.read_index(index_path)

# Load the knowledge base vectors
knowledge_base_vectors = np.load(knowledge_base_path)

def search_vectors(index, query_vectors, k=5):
    D, I = index.search(query_vectors, k)
    return D, I

def get_relevant_documents(question, k=5):
    query_vector = embedding_model.encode([question]).astype('float32')
    distances, indices = search_vectors(index, query_vector, k)
    return df.iloc[indices[0]]

def generate_answer(question, context):
    result = qa_pipeline({'question': question, 'context': context})
    return result['answer']

print("All components loaded successfully!")

st.markdown("<h1 style='text-align: center; color: blue;'>Indigo</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: skyblue;'>Question-Answering Model</h4>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: skyblue;'>Hack-to-Hire 2024</h6>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: skyblue;'></h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: skyblue;'></h4>", unsafe_allow_html=True)

st.markdown("<h6 style='color: skyblue;'>Description: </h6>", unsafe_allow_html=True)
st.write("I, Rakshit Walia, developed a state-of-the-art question-answering model leveraging the Quora Question" 
         "Answer Dataset with the goal of creating an AI system capable of understanding and generating accurate responses to" 
         "a wide variety of user queries. By employing BERT, I fine-tuned the model to enhance contextual comprehension" 
         "and generate human-like interactions. The implementation involved meticulous" 
         "data processing using the `pandas` library, and extensive training to handle diverse question formats and maintain" 
         "conversational coherence. The resulting system offers a robust tool and has many applications")

st.markdown("<h1 style='text-align: center; color: yellow ;'></h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: yellow ;'></h1>", unsafe_allow_html=True)

text_input = st.text_input(
    "Ask a Question 👇",
)

if text_input:

    st.write("Question : ", text_input)

    #Retrieve relevant documents
    relevant_docs = get_relevant_documents(text_input)
    context = " ".join(relevant_docs['answer'].tolist())

    #Generate answer using the retrieved context
    med_answer = generate_answer(text_input, context)
    answer = get_natural_language_answer(text_input, med_answer)
    
    st.write("Model Generated Answer: ")
    st.write(answer)

    rating = st.slider("Was the response helpful? Please Rate the Quality of the response", 1, 5, 3)
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("Submit", key=1, use_container_width=True) :
            if(rating==1):
                st.write("You Gave : ⭐")
                st.write("Thank you for your feedback; we're sorry for any inconvenience.")
            elif(rating==2):
                st.write("You Gave : ⭐⭐")
                st.write("Thanks for your input; we're working on improvements.")
            elif(rating==3):
                st.write("You Gave : ⭐⭐⭐")
                st.write("Appreciate the feedback; we'll keep working to get better.")
            
            elif(rating==4):
                st.write("You Gave : ⭐⭐⭐⭐")
                st.write("Thanks for the positive feedback!")
            
                df = pd.read_csv('data_updated.csv', encoding='ISO-8859-1')
                new_row = {
                    'question': text_input,
                    'answer': answer}
                new_row_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_row_df], ignore_index=True)
                df.to_csv('data_updated.csv', index=False)
                print("Row added and CSV saved successfully.")
            
            elif(rating==5):
                st.write("You Gave : ⭐⭐⭐⭐⭐")
                st.write("Thank you for the 5 stars! We're glad you had a great experience!")

                df = pd.read_csv('data_updated.csv', encoding='ISO-8859-1')               
                new_row = {
                    'question': text_input,
                    'answer': answer}
                new_row_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_row_df], ignore_index=True)
                df.to_csv('data_updated.csv', index=False)
                print("Row added and CSV saved successfully.")
                