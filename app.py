from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Initialize your model and prompt
template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Define the Flask route for the home page
@app.route('/')
def index():
    return render_template('index.html')  # Renders the HTML page

# Route to handle chat requests
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']  # Get user input
    context = request.form.get('context', '')  # Get the conversation context (empty on first message)

    # Process the input using the Ollama model
    result = chain.invoke({"context": context, "question": user_input})

    # Update the context with the conversation history
    updated_context = f"{context}\nUser: {user_input}\nAI: {result}"

    return jsonify({'response': result, 'context': updated_context})  # Return bot response and updated context

if __name__ == "__main__":
    app.run(debug=True)
