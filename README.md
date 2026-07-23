# Green Card Questionnaire Assistant

An AI-powered Streamlit application that helps users answer **marriage-based Green Card application questions** by searching through immigration reference documents and generating grounded responses.

The application provides a conversational interface where users can ask questions related to the Green Card application process and receive answers based on uploaded immigration documents.

> **Scope:** This application focuses on marriage-based Green Card scenarios where the applicant's spouse is a Green Card holder.

---

# Project Overview

The Green Card application process requires understanding eligibility requirements, documentation, forms, and procedural steps. Immigration documents can be lengthy and difficult to navigate.

This project simplifies access to immigration information by providing an AI assistant that allows users to:

- Ask Green Card-related questions using natural language.
- Quickly find relevant information from immigration documents.
- Receive summarized answers instead of manually searching through PDFs.
- View source documents used to generate responses.

---

# Key Features

## AI-Powered Green Card Q&A

Users can ask questions in plain English and receive AI-generated answers based on uploaded immigration reference documents.

Example questions:
Can a Green Card holder sponsor their spouse?
What documents are required for a marriage-based Green Card?
What evidence proves a bona fide marriage?
What happens after filing Form I-130?


---

## Document-Based Responses

The assistant uses uploaded immigration documents as its knowledge source.

Benefits:

- Provides answers based on reference materials.
- Reduces manual document searching.
- Improves consistency and traceability.
- Allows users to verify information.

---

## Source Document Visibility

Every answer includes the source documents used to generate the response.

Example:
Answer:
A spouse of a Green Card holder may qualify under the family-based
immigration category depending on eligibility requirements.

Source Documents:

USCIS_Marriage_Green_Card_Guide.pdf
Spouse_Sponsorship_Process.pdf


---

# Application Workflow

Immigration Reference PDFs
|
v
Document Processing
|
v
Searchable Knowledge Base
|
v
User Question
|
v
Relevant Information Retrieval
|
v
AI Response Generation
|
v
Answer + Source Documents


---

# Technology Stack

## Frontend

### Streamlit

Used to build the interactive web interface.

Responsibilities:

- User question input
- Displaying AI responses
- Showing document references
- Managing application interaction

---

## AI Models

### Google Gemini Flash

Model:
gemini-embedding-001


Used for:

- Converting document content into searchable representations.
- Finding relevant sections based on user questions.

---

## Vector Search

### FAISS

FAISS provides fast similarity search over document information and helps identify the most relevant sections for each user query.

---

## AI Framework

### LangChain

LangChain connects the different components:

- PDF document loading
- Text processing
- Document search
- AI response generation

