import streamlit as st
import PyPDF2
from transformers import pipeline
import torch
import tensorflow as tf
from tensorflow import keras


# Streamlit app
def main():
    st.title("Cutting-Edge PDF Summarizer")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        try:
            # Read the PDF file
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ''

            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

            # Display extracted text (optional)
            st.subheader("Extracted Text")
            st.text_area("Text from PDF", text, height=300)

            # Load the summarization pipeline
            st.subheader("Processing the Summary")
            summarizer = pipeline("summarization", model="t5-base")

            # Customize the summary length
            max_length = st.slider("Maximum length of the summary (words)", 50, 500, 150)
            min_length = st.slider("Minimum length of the summary (words)", 20, 100, 50)

            # Generate the summary
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

            # Display the summary
            st.subheader("Summary")
            st.write(summary[0]['summary_text'])

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
