import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
from collections import Counter
import re
import requests
import os




app = Flask(__name__)
CORS(app)

import nltk
nltk.data.path.append('./nltk_data')  # look for data in the repo

from nltk.corpus import stopwords
stop_words = stopwords.words('english')



# Define the preprocessing function
def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

#  Load model and vectorizer locally (no MLflow)
model_path = "models/logistic_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print(" Model and vectorizer loaded locally.")

@app.route('/')
def home():
    return "Welcome to our flask API"

# @app.route('/predict_with_timestamps', methods=['POST'])
# def predict_with_timestamps():
#     data = request.json
#     comments_data = data.get('comments')
    
#     if not comments_data:
#         return jsonify({"error": "No comments provided"}), 400

#     try:
#         comments = [item['text'] for item in comments_data]
#         timestamps = [item['timestamp'] for item in comments_data]
#         preprocessed_comments = [preprocess_comment(comment) for comment in comments]
#         transformed_comments = vectorizer.transform(preprocessed_comments)
#         predictions = [str(pred) for pred in model.predict(transformed_comments)]
#     except Exception as e:
#         return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
#     response = [{"comment": c, "sentiment": s, "timestamp": t}
#                 for c, s, t in zip(comments, predictions, timestamps)]
#     return jsonify(response)

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        plt.figure(figsize=(6, 6))
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=None,   # keep percentages inside pie only
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Add legend above, centered, horizontal
        plt.legend(
            wedges, labels,
            title="Sentiments",
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05),  # push legend above chart
            ncol=3                       # horizontal layout
        )


        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500
    

# class AdvancedHFCommentAnalyzer:
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.base_url = "https://api-inference.huggingface.co/models"
#         self.headers = {"Authorization": f"Bearer {api_key}"}
        
#         # Advanced model endpoints for better analysis
#         self.models = {
#             # Summarization with better understanding
#             'summarization': f"{self.base_url}/facebook/bart-large-cnn",
#             'abstractive_summary': f"{self.base_url}/google/pegasus-cnn_dailymail",
            
#             # Advanced sentiment analysis
#             'sentiment': f"{self.base_url}/nlptown/bert-base-multilingual-uncased-sentiment",
#             'emotion': f"{self.base_url}/j-hartmann/emotion-english-distilroberta-base",
            
#             # Topic and aspect extraction
#             'zero_shot': f"{self.base_url}/facebook/bart-large-mnli",
#             'aspect_extraction': f"{self.base_url}/yangheng/deberta-v3-base-absa-v1.1",
            
#             # Named Entity Recognition for better themes
#             'ner': f"{self.base_url}/dbmdz/bert-large-cased-finetuned-conll03-english",
            
#             # Question Answering for specific insights
#             'qa': f"{self.base_url}/deepset/roberta-base-squad2"
#         }
    
#     def call_hf_api(self, model_url, payload, retries=3):
#         """Enhanced API call with better error handling"""
#         import time
        
#         for attempt in range(retries):
#             try:
#                 response = requests.post(model_url, headers=self.headers, json=payload, timeout=45)
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     if isinstance(result, dict) and 'error' in result:
#                         if 'loading' in result['error'].lower():
#                             time.sleep(15)  # Wait for model to load
#                             continue
#                     return result
#                 elif response.status_code == 503:
#                     time.sleep(10 * (attempt + 1))  # Progressive backoff
#                     continue
                    
#             except Exception as e:
#                 if attempt < retries - 1:
#                     time.sleep(5)
#                     continue
        
#         return {"error": "All API attempts failed"}
    
#     def extract_intelligent_themes(self, comments):
#         """Use zero-shot classification to find actual themes"""
        
#         # Define potential themes that are relevant for YouTube videos
#         candidate_themes = [
#             "video quality and production",
#             "educational content and learning",
#             "entertainment and humor", 
#             "technical explanations",
#             "personal opinions and reactions",
#             "audio and sound quality",
#             "visual effects and editing",
#             "presenter style and delivery",
#             "accuracy and fact-checking",
#             "comparison with other content",
#             "beginner vs advanced level",
#             "practical applications",
#             "community and interaction",
#             "monetization and sponsorship",
#             "platform and accessibility issues"
#         ]
        
#         combined_text = " ".join(comments[:30])  # Sample for analysis
        
#         payload = {
#             "inputs": combined_text,
#             "parameters": {"candidate_labels": candidate_themes}
#         }
        
#         result = self.call_hf_api(self.models['zero_shot'], payload)
        
#         if 'error' in result:
#             return self.fallback_theme_extraction(comments)
        
#         # Get themes with confidence > 0.3
#         themes = []
#         labels = result.get('labels', [])
#         scores = result.get('scores', [])
        
#         for label, score in zip(labels, scores):
#             if score > 0.3:  # Only confident predictions
#                 themes.append(label.title())
        
#         return themes[:5] if themes else self.fallback_theme_extraction(comments)
    
#     def extract_intelligent_concerns(self, comments):
#         """Use aspect-based sentiment analysis to find real concerns"""
        
#         # Define specific concern areas for YouTube content
#         concern_aspects = [
#             "audio quality problems",
#             "video clarity issues", 
#             "misinformation or errors",
#             "boring or unengaging content",
#             "inappropriate content",
#             "technical difficulties",
#             "poor explanation quality",
#             "offensive language or behavior",
#             "clickbait or misleading titles",
#             "excessive advertising",
#             "accessibility issues",
#             "outdated information"
#         ]
        
#         concerns = []
        
#         # Analyze negative comments for specific concerns
#         negative_comments = self.identify_negative_comments(comments[:20])
        
#         for comment in negative_comments:
#             # Use zero-shot to classify what type of concern this is
#             payload = {
#                 "inputs": comment,
#                 "parameters": {"candidate_labels": concern_aspects}
#             }
            
#             result = self.call_hf_api(self.models['zero_shot'], payload)
            
#             if 'error' not in result:
#                 labels = result.get('labels', [])
#                 scores = result.get('scores', [])
                
#                 # Get the most likely concern if confidence is high
#                 if scores and scores[0] > 0.5:
#                     concerns.append({
#                         'concern': labels[0],
#                         'example': comment[:100] + "..." if len(comment) > 100 else comment,
#                         'confidence': scores[0]
#                     })
        
#         # Deduplicate and format concerns
#         concern_summary = {}
#         for concern_data in concerns:
#             concern_type = concern_data['concern']
#             if concern_type not in concern_summary:
#                 concern_summary[concern_type] = {
#                     'examples': [],
#                     'total_confidence': 0,
#                     'count': 0
#                 }
            
#             concern_summary[concern_type]['examples'].append(concern_data['example'])
#             concern_summary[concern_type]['total_confidence'] += concern_data['confidence']
#             concern_summary[concern_type]['count'] += 1
        
#         # Format final concerns
#         final_concerns = []
#         for concern_type, data in sorted(concern_summary.items(), 
#                                        key=lambda x: x[1]['total_confidence'], 
#                                        reverse=True)[:5]:
#             avg_confidence = data['total_confidence'] / data['count']
#             final_concerns.append(f"{concern_type.title()} (mentioned {data['count']} times)")
        
#         return final_concerns if final_concerns else ["No specific concerns identified through AI analysis"]
    
#     def extract_intelligent_highlights(self, comments):
#         """Extract meaningful positive highlights using emotion analysis"""
        
#         highlight_emotions = [
#             "appreciation and gratitude",
#             "excitement and enthusiasm", 
#             "learning and discovery",
#             "inspiration and motivation",
#             "entertainment and joy",
#             "helpfulness and utility",
#             "clarity and understanding",
#             "uniqueness and creativity",
#             "professionalism and quality",
#             "community and connection"
#         ]
        
#         highlights = []
        
#         # Analyze positive comments for specific highlights
#         positive_comments = self.identify_positive_comments(comments[:20])
        
#         for comment in positive_comments:
#             # Use emotion classification first
#             emotion_payload = {"inputs": comment}
#             emotion_result = self.call_hf_api(self.models['emotion'], emotion_payload)
            
#             if 'error' not in emotion_result:
#                 # Then classify the type of positive sentiment
#                 highlight_payload = {
#                     "inputs": comment,
#                     "parameters": {"candidate_labels": highlight_emotions}
#                 }
                
#                 highlight_result = self.call_hf_api(self.models['zero_shot'], highlight_payload)
                
#                 if 'error' not in highlight_result:
#                     labels = highlight_result.get('labels', [])
#                     scores = highlight_result.get('scores', [])
                    
#                     if scores and scores[0] > 0.4:
#                         highlights.append({
#                             'highlight': labels[0],
#                             'example': comment[:100] + "..." if len(comment) > 100 else comment,
#                             'confidence': scores[0]
#                         })
        
#         # Process and deduplicate highlights
#         highlight_summary = {}
#         for highlight_data in highlights:
#             highlight_type = highlight_data['highlight']
#             if highlight_type not in highlight_summary:
#                 highlight_summary[highlight_type] = {
#                     'examples': [],
#                     'total_confidence': 0,
#                     'count': 0
#                 }
            
#             highlight_summary[highlight_type]['examples'].append(highlight_data['example'])
#             highlight_summary[highlight_type]['total_confidence'] += highlight_data['confidence']
#             highlight_summary[highlight_type]['count'] += 1
        
#         # Format final highlights
#         final_highlights = []
#         for highlight_type, data in sorted(highlight_summary.items(), 
#                                          key=lambda x: x[1]['total_confidence'], 
#                                          reverse=True)[:5]:
#             avg_confidence = data['total_confidence'] / data['count']
#             final_highlights.append(f"{highlight_type.title()} (expressed {data['count']} times)")
        
#         return final_highlights if final_highlights else ["General positive reception noted"]
    
#     def identify_negative_comments(self, comments):
#         """Identify truly negative comments using advanced sentiment analysis"""
#         negative_comments = []
        
#         for comment in comments:
#             payload = {"inputs": comment}
#             result = self.call_hf_api(self.models['sentiment'], payload)
            
#             if 'error' not in result and result:
#                 # Handle different response formats
#                 sentiment_data = result[0] if isinstance(result, list) else result
                
#                 # Check if it's negative with high confidence
#                 if hasattr(sentiment_data, 'get'):
#                     label = sentiment_data.get('label', '').upper()
#                     score = sentiment_data.get('score', 0)
                    
#                     if ('NEGATIVE' in label or '1 STAR' in label or '2 STAR' in label) and score > 0.7:
#                         negative_comments.append(comment)
        
#         return negative_comments
    
#     def identify_positive_comments(self, comments):
#         """Identify genuinely positive comments"""
#         positive_comments = []
        
#         for comment in comments:
#             payload = {"inputs": comment}
#             result = self.call_hf_api(self.models['sentiment'], payload)
            
#             if 'error' not in result and result:
#                 sentiment_data = result[0] if isinstance(result, list) else result
                
#                 if hasattr(sentiment_data, 'get'):
#                     label = sentiment_data.get('label', '').upper()
#                     score = sentiment_data.get('score', 0)
                    
#                     if ('POSITIVE' in label or '4 STAR' in label or '5 STAR' in label) and score > 0.7:
#                         positive_comments.append(comment)
        
#         return positive_comments
    
#     def generate_intelligent_summary(self, comments, themes):
#         """Generate contextual summary using advanced NLP"""
        
#         # Create context-aware summary prompt
#         context_text = f"Video comments analysis: {len(comments)} total comments. "
#         context_text += f"Main themes: {', '.join(themes[:3])}. "
#         # context_text += f"Key concerns: {len(concerns)} identified. "
#         # context_text += f"Positive aspects: {len(highlights)} highlighted."
        
#         # Sample representative comments for summary generation
#         sample_comments = comments[:15]  # Limit for API efficiency
#         combined_text = f"{context_text}\n\nRepresentative comments:\n" + "\n".join(sample_comments)
        
#         # Use advanced summarization
#         payload = {
#             "inputs": combined_text,
#             "parameters": {
#                 "max_length": 150,
#                 "min_length": 60,
#                 "do_sample": True,
#                 "temperature": 0.3
#             }
#         }
        
#         result = self.call_hf_api(self.models['abstractive_summary'], payload)
        
#         if 'error' in result:
#             # Fallback to BART
#             result = self.call_hf_api(self.models['summarization'], payload)
        
#         if 'error' in result:
#             return self.generate_fallback_summary(comments, themes)
        
#         summary = result[0].get('summary_text', '') if isinstance(result, list) else result.get('summary_text', '')
        
#         # Enhance summary with specific insights
#         enhanced_summary = self.enhance_summary_with_insights(summary, themes, len(comments))
        
#         return enhanced_summary
    
#     def enhance_summary_with_insights(self, base_summary, themes, comment_count):
#         """Add intelligent context to the summary"""
        
#         # Analyze engagement level
#         engagement_level = "high" if comment_count > 100 else "moderate" if comment_count > 30 else "limited"
        
#         # Analyze sentiment balance
#         # concern_ratio = len([c for c in concerns if 'no' not in c.lower()]) / max(len(concerns), 1)
#         # highlight_ratio = len([h for h in highlights if 'general' not in h.lower()]) / max(len(highlights), 1)
        
#         # if highlight_ratio > concern_ratio * 2:
#         #     sentiment_assessment = "overwhelmingly positive"
#         # elif concern_ratio > highlight_ratio * 2:
#         #     sentiment_assessment = "notably critical"
#         # else:
#         #     sentiment_assessment = "balanced"
        
#         # Create intelligent summary
#         enhanced = f"{base_summary} "
#         # enhanced += f"Analysis of {comment_count} comments reveals {engagement_level} audience engagement with {sentiment_assessment} reception. "
        
#         if themes:
#             enhanced += f"Primary discussion centers on {themes[0].lower()}"
#             if len(themes) > 1:
#                 enhanced += f" and {themes[1].lower()}"
#             enhanced += ". "
        
#         # if concern_ratio > 0.3:
#         #     enhanced += f"Notable viewer concerns require attention. "
        
#         # if highlight_ratio > 0.3:
#         #     enhanced += f"Strong positive feedback indicates successful content delivery."
        
#         return enhanced
    
#     # Fallback methods for when APIs fail
#     def fallback_theme_extraction(self, comments):
#         """Simple theme extraction when API fails"""
#         themes = ["Technical Discussion", "General Feedback", "Content Quality"]
#         return themes
    
#     def generate_fallback_summary(self, comments, themes, concerns, highlights, comment_count):
#         """Generate basic summary when APIs fail"""
#         return f"Analysis of {len(comments)} comments shows mixed audience reception with various themes including {', '.join(themes[:2])} and general feedback patterns."
    
#     def analyze_comments(self, comments):
#         """Main analysis method with intelligent processing"""
#         try:
#             if len(comments) < 2:
#                 return {
#                     "key_themes": ["Insufficient data for intelligent analysis"],
#                     "main_concerns": ["Need more comments for AI analysis"],
#                     "positive_highlights": ["Need more comments for AI analysis"],
#                     "executive_summary": "Insufficient comment data for advanced AI analysis."
#                 }
            
#             # Clean comments
#             clean_comments = [self.clean_comment(c) for c in comments if len(str(c).strip()) > 10]
#             analysis_comments = clean_comments[:60]  # Limit for API efficiency
            
#             # Parallel analysis using advanced AI
#             print("🧠 Extracting intelligent themes...")
#             key_themes = self.extract_intelligent_themes(analysis_comments)
            
#             # print("⚠️  Analyzing specific concerns...")
#             # main_concerns = self.extract_intelligent_concerns(analysis_comments)
            
#             # print("✨ Identifying positive highlights...")
#             # positive_highlights = self.extract_intelligent_highlights(analysis_comments)
            
#             # main_concerns, positive_highlights

#             print("📝 Generating contextual summary...")
#             executive_summary = self.generate_intelligent_summary(analysis_comments, key_themes)
            
#             return {
#                 "key_themes": key_themes,
#                 # "main_concerns": main_concerns,
#                 # "positive_highlights": positive_highlights,
#                 "executive_summary": executive_summary,
#                 "analysis_metadata": {
#                     "total_comments_analyzed": len(analysis_comments),
#                     "original_comment_count": len(comments),
#                     "processing_method": "advanced_ai_analysis",
#                     "models_used": ["BART-MNLI", "DeBERTa-ABSA", "DistilRoBERTa-Emotion", "PEGASUS"],
#                     "intelligence_level": "high"
#                 }
#             }
            
#         except Exception as e:
#             return {
#                 "key_themes": ["Advanced AI analysis encountered limitations"],
#                 "main_concerns": ["Technical processing constraints with AI models"],
#                 "positive_highlights": ["Attempted advanced sentiment understanding"],
#                 "executive_summary": f"Processed {len(comments)} comments using advanced AI with some limitations due to model availability.",
#                 "error": str(e)
#             }
    
#     def clean_comment(self, comment):
#         """Enhanced comment cleaning"""
#         if not comment:
#             return ""
#         # Remove URLs, excessive punctuation, normalize whitespace
#         clean = re.sub(r'http[s]?://\S+', '', str(comment))
#         clean = re.sub(r'[^\w\s.,!?-]', ' ', clean)
#         clean = re.sub(r'\s+', ' ', clean.strip())
#         return clean

# class MinimalCommentAnalyzer:
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.headers = {"Authorization": f"Bearer {api_key}"}
        
#         # Using only the fastest, most reliable models
#         self.models = {
#             # Fast theme classification - very quick response
#             'themes': "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
#             # Fast summarization - lightweight model
#             'summary': "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-6-6"
#         }
    
#     def quick_api_call(self, model_url, payload, timeout=15):
#         """Ultra-fast API call with minimal retries"""
#         try:
#             response = requests.post(
#                 model_url, 
#                 headers=self.headers, 
#                 json=payload, 
#                 timeout=timeout
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
#                 if isinstance(result, dict) and 'error' in result:
#                     return {"error": result['error']}
#                 return result
#             else:
#                 return {"error": f"API error: {response.status_code}"}
                
#         except Exception as e:
#             return {"error": str(e)}
    
#     def extract_quick_themes(self, comments):
#         """Fast theme extraction using predefined categories"""
        
#         # Minimal set of most common YouTube video themes
#         quick_themes = [
#             "video quality",
#             "educational content", 
#             "entertainment",
#             "audio issues",
#             "helpful tutorial",
#             "boring content",
#             "great explanation",
#             "technical problems"
#         ]
        
#         # Take only first 10 comments and combine for speed
#         sample_text = " ".join(comments[:10])[:1000]  # Limit text length
        
#         payload = {
#             "inputs": sample_text,
#             "parameters": {"candidate_labels": quick_themes}
#         }
        
#         result = self.quick_api_call(self.models['themes'], payload)
        
#         if 'error' in result:
#             return ["General Discussion", "Mixed Feedback"]
        
#         # Return top 3 themes only
#         labels = result.get('labels', [])[:3]
#         return [label.title() for label in labels] if labels else ["General Feedback"]
    
#     def generate_quick_summary(self, comments):
#         """Super fast summary generation"""
        
#         # Take only first 8 comments to keep it lightning fast
#         sample_comments = comments[:8]
#         combined_text = " ".join(sample_comments)
        
#         # Truncate to 500 chars for speed
#         if len(combined_text) > 500:
#             combined_text = combined_text[:500] + "..."
        
#         payload = {
#             "inputs": combined_text,
#             "parameters": {
#                 "max_length": 80,  # Very short summary
#                 "min_length": 30,
#                 "do_sample": False  # Faster processing
#             }
#         }
        
#         result = self.quick_api_call(self.models['summary'], payload, timeout=10)
        
#         if 'error' in result:
#             return f"Quick analysis of {len(comments)} comments shows mixed audience engagement with various feedback patterns."
        
#         summary_text = ""
#         if isinstance(result, list) and len(result) > 0:
#             summary_text = result[0].get('summary_text', '')
#         elif isinstance(result, dict):
#             summary_text = result.get('summary_text', '')
        
#         if not summary_text:
#             return f"Analysis of {len(comments)} comments reveals general audience discussion."
            
#         return summary_text
    
#     def clean_comment_fast(self, comment):
#         """Minimal comment cleaning for speed"""
#         if not comment:
#             return ""
#         # Only basic cleaning
#         clean = str(comment).strip()
#         clean = re.sub(r'http\S+', '', clean)  # Remove URLs
#         clean = re.sub(r'\s+', ' ', clean)     # Normalize spaces
#         return clean[:200]  # Truncate long comments
    
#     def analyze_comments_fast(self, comments):
#         """Ultra-fast analysis - themes and summary only"""
#         try:
#             if len(comments) < 1:
#                 return {
#                     "key_themes": ["No Comments"],
#                     "executive_summary": "No comments available for analysis."
#                 }
            
#             # Clean only what we need
#             clean_comments = [
#                 self.clean_comment_fast(c) for c in comments[:15]  # Max 15 comments
#                 if len(str(c).strip()) > 5
#             ]
            
#             if len(clean_comments) == 0:
#                 return {
#                     "key_themes": ["Empty Comments"],
#                     "executive_summary": "Comments were too short or empty for analysis."
#                 }
            
#             print(f"⚡ Fast analysis of {len(clean_comments)} comments...")
            
#             # Parallel-ish processing - themes first (usually faster)
#             key_themes = self.extract_quick_themes(clean_comments)
#             executive_summary = self.generate_quick_summary(clean_comments)
            
#             return {
#                 "key_themes": key_themes,
#                 "executive_summary": executive_summary,
#                 "processed_comments": len(clean_comments),
#                 "total_comments": len(comments),
#                 "processing_mode": "ultra_fast"
#             }
            
#         except Exception as e:
#             return {
#                 "key_themes": ["Analysis Error"],
#                 "executive_summary": f"Quick analysis of {len(comments)} comments completed with some limitations.",
#                 "error": str(e)
#             }

    
# # Initialize with your HF API key
# HF_API_KEY = os.getenv("HF_API_KEY")
# fast_analyzer = MinimalCommentAnalyzer(HF_API_KEY)


# @app.route('/ai_comment_summary', methods=['POST'])
# def fast_comment_analysis():
#     """
#     Ultra-fast comment analysis - themes and summary only
#     Optimized for speed with minimal processing
#     """
#     try:
#         data = request.get_json()
        
#         if not data or 'comments' not in data:
#             return jsonify({
#                 'error': 'Missing comments data',
#                 'required_format': {'comments': ['comment1', 'comment2']}
#             }), 400
        
#         comments = data['comments']
        
#         if not isinstance(comments, list):
#             return jsonify({'error': 'Comments must be a list'}), 400
        
#         # Super fast analysis
#         result = fast_analyzer.analyze_comments_fast(comments)
        
#         return jsonify(result)
    
#     except Exception as e:
#         return jsonify({
#             'error': 'Fast analysis failed',
#             'message': str(e)
#         }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
