from flask import Flask, request, jsonify, send_file
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

app = Flask(__name__)

# ------------------------------
# NLP Setup
# ------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ------------------------------
# FAQ Data
# ------------------------------
faq_data = {
    
}

preprocessed_questions = [preprocess(q) for q in faq_data.keys()]

# ------------------------------
# Small Talk
# ------------------------------
small_talk = {

    "hi": "Hey there! 👋 How can I help you today?",
    "hello": "Hello! 😊 How are you doing?",
    "hey": "Hi there! Ready to explore something?",
    "good morning": "Good morning 🌞 Hope your day’s off to a great start!",
    "good afternoon": "Good afternoon ☀️ How can I assist you today?",
    "good evening": "Good evening 🌆 What can I do for you?",
    "how are you": "I’m just a bot, but I’m doing great! Thanks for asking 😄",
    "thank you": "You’re most welcome! 💙",
    "thanks": "Happy to help! 🙌",
    "bye": "Goodbye 👋 Stay safe and take care!",
    
    # About CodeAlpha
    "code": "CodeAlpha is a technology-driven platform focused on empowering learners and professionals through internships, training programs, and project-based learning in tech and management fields.",
    "services": "CodeAlpha offers virtual internships, hands-on training, workshops, and real-world project experience across domains like web development, data science, AI/ML, and cybersecurity.",
    # Additional small talk
    "what's your name": "I'm your friendly FAQ bot created by Gynorrj! What's yours?",
    "who made you": "I was created by Gynorrj.",
    "can you help me": "Absolutely! What do you need help with?",
    "what can you do": "I can answer your questions and provide helpful info.",
    
    # FAQ style Q&A
    "how do i reset my password": "To reset your password, click on 'Forgot Password' at login and follow the instructions.",
    "what is your return policy": "Our return policy allows returns within 30 days of purchase with a receipt.",
    "how can i track my order": "You can track your order using the tracking link sent to your email.",
    "do you offer international shipping": "Yes, we ship internationally. Shipping fees vary by location.",
    "what payment methods do you accept": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.",
    "what are your business hours": "Our support team is available Monday to Friday, 9 AM to 6 PM.",
    "can i change my order": "Orders can be changed within 2 hours of placing them. Contact support quickly!",
    "how do i cancel my subscription": "To cancel your subscription, go to account settings and select 'Cancel Subscription'.",
    "is my data secure": "Yes, we use industry-standard security measures to protect your data.",
   
    # Ending conversations
    "thanks for your help": "Anytime! Feel free to ask if you need anything else.",
    "i'm done for now": "Okay, have a great day! Come back anytime.",
    "talk later": "Looking forward to it! Bye for now 👋",
    "you are awesome": "Thanks! You’re pretty awesome too 😊",
    "see you soon": "See you soon! Take care!",
}

# ------------------------------
# Chatbot Logic
# ------------------------------
def chatbot_response(user_input):
    for key in small_talk:
        if key in user_input.lower():
            return small_talk[key]
    
    user_processed = preprocess(user_input)
    all_questions = preprocessed_questions + [user_processed]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_questions)
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    idx = similarity.argmax()
    score = similarity[0, idx]
    
    if score > 0.3:
        question = list(faq_data.keys())[idx]
        return faq_data[question]
    else:
        return "🤔 I'm not sure about that. Could you rephrase?"

# ------------------------------
# Flask Routes
# ------------------------------
@app.route("/")
def home():
    return send_file("FAQ_Chatbot.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_msg = request.json['message']
    bot_reply = chatbot_response(user_msg)
    return jsonify({"reply": bot_reply})

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
