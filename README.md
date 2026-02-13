# Data-Analyst-Projrct-3

📘 PROJECT REPORT

1️⃣ Project Overview

The Dynamic AI Chatbot is an intelligent conversational system designed to understand natural language queries and respond intelligently. It uses Natural Language Processing (NLP) and Machine Learning techniques to recognize user intent, extract important information, analyze sentiment, and generate contextual responses.
The chatbot can be integrated into web platforms, customer support systems, and virtual assistant applications.



2️⃣ Objectives
Develop an NLP-based conversational AI system
Implement Machine Learning–based intent recognition
Perform Named Entity Recognition (NER)
Analyze user sentiment
Maintain contextual memory
Provide real-time API-based communication



3️⃣ System Architecture

User → FastAPI Backend → NLP Processing → Intent Classifier → Entity Extraction → Response Generator → SQLite Database → Analytics



4️⃣ Key Features



🔹 Intent Recognition
Uses TF-IDF and Logistic Regression to classify user queries into predefined intents.
🔹 Named Entity Recognition
Extracts important information such as:
Email addresses
Phone numbers
Order IDs
Dates

🔹 Sentiment Analysis
Detects whether user tone is:
Positive
Negative
Neutral

🔹 Contextual Memory
Maintains conversation history to ensure smooth interaction.

🔹 Fallback Mechanism
Handles low-confidence predictions with clarification prompts.

🔹 Analytics Tracking
Tracks:
Intent distribution
Fallback rate
Average response latency
User feedback



5️⃣ Technologies Used
Technology	Purpose
🐍 Python	Core development
⚡ FastAPI	Backend API
🤖 Scikit-learn	Intent classification
📚 NLTK (VADER)	Sentiment analysis
🗄 SQLite	Database storage
🔌 WebSockets	Real-time communication



6️⃣ Database Structure
The system stores:
Session details
Messages (user & bot)
Intent and sentiment results
Feedback records
Tool execution logs



7️⃣ Applications

Customer Suppor Automation
E-commerce Chat Assistant
Banking Virtual Assistant
IT Helpdesk Support
FAQ Automation



8️⃣ Conclusion

The Dynamic AI Chatbot demonstrates how NLP and Machine Learning can be combined to build an intelligent, scalable conversational system. With intent recognition, entity extraction, sentiment analysis, contextual memory, and analytics tracking, the chatbot provides a strong foundation for real-world AI-driven communication systems.
