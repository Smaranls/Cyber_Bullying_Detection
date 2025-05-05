# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import re
import os
import requests
import time
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import uuid
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
import pytz
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

load_dotenv()
email = os.getenv('EMAIL')
password = os.getenv('EMAIL_PASSWORD')

# OCR setup
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract is not installed. OCR functionality will be limited.")

# NLTK data download
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
app.config['REPORTS_FOLDER'] = 'static/reports'
app.config['EMAIL_USERNAME'] = email
app.config['EMAIL_PASSWORD'] = password
db = SQLAlchemy(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    city = db.Column(db.String(120), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    parent_email = db.Column(db.String(120), nullable=True)
    activities = db.relationship('UserActivity', backref='user', lazy=True)

# UserActivity Model
class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    activity_type = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=True)
    result = db.Column(db.Boolean, nullable=False)
    reason = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

with app.app_context():
    db.create_all()

sia = SentimentIntensityAnalyzer()

model = MobileNetV2(weights='imagenet')

negative_patterns = [
    r"no\s+one\s+(?:will\s+)?(?:ever\s+)?(?:like|love)s?\s+you",
    r"(?:you\s+are|you're)\s+(?:ugly|stupid|dumb|fat|idiot|pathetic|loser|worthless)",
    r"(?:hate|hating)\s+you",
    r"kill\s+(?:your\s*self|yourself)",
    r"(?:go|just|why\s+don't\s+you)\s+die",
    r"should\s+(?:just\s+)?die",
    r"nobody\s+(?:likes|cares\s+about)\s+you",
    r"everyone\s+hates\s+you",
    r"(?:you\s+are|you're)\s+(?:better\s+off|worthless)\s+dead",
]

bullying_keywords = [
    'stupid', 'ugly', 'fat', 'dumb', 'idiot', 'loser', 'worthless', 'kill yourself', 
    'hate you', 'pathetic', 'nobody loves', 'no one loves', 'die', 'hate yourself',
    'freak', 'end your life', 'end yourself', 'everyone hates', 'good for nothing', 'reject', 'unwanted'
]

harmful_objects = [
    'knife', 'dagger', 'sword', 'axe', 'revolver', 'gun', 'pistol', 'rifle',
    'shotgun', 'weapon', 'assault_rifle', 'missile', 'projectile', 'blade'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_cyberbullying(text):
    text_lower = text.lower()
    for pattern in negative_patterns:
        if re.search(pattern, text_lower):
            return True, "Contains harmful language pattern"
    for keyword in bullying_keywords:
        if keyword in text_lower:
            return True, f"Contains harmful keyword: '{keyword}'"
    sentiment = sia.polarity_scores(text)
    compound_score = sentiment['compound']
    if compound_score < -0.3:
        return True, f"Very negative sentiment: {compound_score}"
    return False, f"Sentiment score: {compound_score}"

def extract_text_from_image(image_path):
    if not TESSERACT_AVAILABLE:
        return "OCR module (pytesseract) not installed. Cannot extract text from image."
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        if len(text.strip()) < 5:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=1)
            text = pytesseract.image_to_string(dilated)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def process_image_analysis(image_path):
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        detected_harmful_object = None
        max_confidence = 0.0
        for _, class_name, confidence in decoded_predictions:
            for harmful_object in harmful_objects:
                if harmful_object in class_name.lower():
                    if confidence > max_confidence:
                        detected_harmful_object = class_name
                        max_confidence = confidence
        
        object_is_harmful = detected_harmful_object is not None and max_confidence > 0.15
        extracted_text = extract_text_from_image(image_path)
        
        if extracted_text and len(extracted_text.strip()) > 0:
            text_is_bullying, text_reason = is_cyberbullying(extracted_text)
        else:
            text_is_bullying = False
            text_reason = "No text detected in image or text could not be analyzed"
        
        is_bullying = object_is_harmful or text_is_bullying
        sentiment_value = -0.5 if is_bullying else 0.2
        
        if object_is_harmful and text_is_bullying:
            detection_reason = f"Detected harmful object ({detected_harmful_object}, conf: {max_confidence:.2f}) AND harmful text in image: {text_reason}"
        elif object_is_harmful:
            detection_reason = f"Detected harmful object: {detected_harmful_object} (confidence: {max_confidence:.2f})"
        elif text_is_bullying:
            detection_reason = f"Detected harmful text in image: {text_reason}"
        else:
            detection_reason = "No harmful content detected in image or text"
        
        return is_bullying, sentiment_value, detection_reason, extracted_text
    except Exception as e:
        print(f"Error in image analysis: {str(e)}")
        return False, 0.0, f"Error in analysis: {str(e)}", ""

def generate_activity_report(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return None
        activities = UserActivity.query.filter_by(user_id=user_id).order_by(UserActivity.timestamp.desc()).all()
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=colors.darkgreen, spaceAfter=12)
        subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Heading2'], fontSize=14, textColor=colors.blue, spaceAfter=10)
        normal_style = styles['Normal']
        elements = []
        elements.append(Paragraph("Cyberbullying Detection - User Activity Report", title_style))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Username: {user.username}", normal_style))
        elements.append(Paragraph(f"Age: {user.age}", normal_style))
        elements.append(Paragraph(f"City: {user.city}", normal_style))
        elements.append(Paragraph(f"Gender: {user.gender.capitalize()}", normal_style))
        elements.append(Spacer(1, 12))
        current_time = datetime.now(pytz.timezone('UTC')).strftime("%Y-%m-%d %H:%M:%S UTC")
        elements.append(Paragraph(f"Report Generated: {current_time}", normal_style))
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Recent Activities:", subtitle_style))
        elements.append(Spacer(1, 6))
        if not activities:
            elements.append(Paragraph("No activities recorded yet.", normal_style))
        else:
            for i, activity in enumerate(activities):
                activity_time = activity.timestamp.strftime("%Y-%m-d %H:%M:%S")
                elements.append(Paragraph(f"Activity #{i+1} - {activity_time}", normal_style))
                elements.append(Paragraph(f"Type: {'Text' if activity.activity_type == 'text' else 'Image'} Analysis", normal_style))
                elements.append(Paragraph(f"Result: {'Bullying Content Detected' if activity.result else 'Non-Bullying Content'}", normal_style))
                if activity.reason:
                    elements.append(Paragraph(f"Reason: {activity.reason}", normal_style))
                if activity.activity_type == 'text' and activity.content:
                    elements.append(Paragraph(f"Analyzed Content: \"{activity.content}\"", normal_style))
                if activity.activity_type == 'image' and activity.content:
                    elements.append(Paragraph(f"Image was analyzed (stored at {activity.content})", normal_style))
                elements.append(Spacer(1, 10))
            bullying_count = sum(1 for a in activities if a.result)
            total_count = len(activities)
            if total_count > 0:
                bullying_percentage = (bullying_count / total_count) * 100
                elements.append(Spacer(1, 10))
                elements.append(Paragraph("Summary Statistics:", subtitle_style))
                elements.append(Paragraph(f"Total Activities: {total_count}", normal_style))
                elements.append(Paragraph(f"Bullying Content Detected: {bullying_count} ({bullying_percentage:.1f}%)", normal_style))
                elements.append(Paragraph(f"Non-Bullying Content: {total_count - bullying_count} ({100 - bullying_percentage:.1f}%)", normal_style))
        doc.build(elements)
        pdf_data = buffer.getvalue()
        buffer.close()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{user.username}_{timestamp}.pdf"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
        with open(report_path, 'wb') as f:
            f.write(pdf_data)
        return report_path
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None

def send_report_email(user_id, report_path):
    try:
        user = User.query.get(user_id)
        if not user or not user.parent_email:
            return False
        msg = MIMEMultipart()
        msg['From'] = app.config['EMAIL_USERNAME']
        msg['To'] = user.parent_email
        msg['Subject'] = f"Cyberbullying Detection Report for {user.username}"
        body = f"""
        Dear Parent/Guardian,
        
        This is an automated report from the Cyberbullying Detection System.
        
        The report contains information about the activities of {user.username} on our platform.
        Please review the attached PDF file for details.
        
        Thank you for helping us create a safer online environment for young users.
        
        Regards,
        Cyberbullying Detection Team
        """
        msg.attach(MIMEText(body, 'plain'))
        with open(report_path, 'rb') as f:
            attachment = MIMEApplication(f.read(), _subtype="pdf")
            filename = os.path.basename(report_path)
            attachment.add_header('Content-Disposition', 'attachment', filename=filename)
            msg.attach(attachment)
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(app.config['EMAIL_USERNAME'], app.config['EMAIL_PASSWORD'])
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def user_requires_reporting(username):
    user = User.query.filter_by(username=username).first()
    return user and user.age <= 16 and user.parent_email

def record_user_activity(username, activity_type, content, result, reason):
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return
        activity = UserActivity(
            user_id=user.id,
            activity_type=activity_type,
            content=content,
            result=result,
            reason=reason,
            timestamp=datetime.utcnow()
        )
        db.session.add(activity)
        db.session.commit()
    except Exception as e:
        print(f"Error recording activity: {str(e)}")

# Nominatim Geocoding with Retry Mechanism
def geocode_location(country, city):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{city}, {country}",
        "format": "json",
        "limit": 1,
        "addressdetails": 1
    }
    headers = {"User-Agent": "CyberbullyingDetectionApp/1.0"}
    for attempt in range(3):  # Retry up to 3 times
        try:
            response = requests.get(url, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = data[0]
                    lat, lon = float(result["lat"]), float(result["lon"])
                    print(f"Geocoded {city}, {country} to ({lat}, {lon})")
                    return lat, lon
                else:
                    print(f"No geocoding results for {city}, {country}")
                    return None, None
            else:
                print(f"Geocoding failed with status {response.status_code}: {response.text}")
                return None, None
        except requests.exceptions.RequestException as e:
            print(f"Geocoding error on attempt {attempt + 1}: {str(e)}")
            if attempt < 2:
                time.sleep(1)  # Respect Nominatim's rate limit (1 request/second)
            continue
    return None, None

# Overpass API to fetch police stations with Retry Mechanism
def get_police_stations(lat, lon, radius_km=10):  # Increased radius to 10 km
    overpass_url = "https://overpass-api.de/api/interpreter"
    lat_delta = radius_km / 111  # 1 degree latitude ~ 111 km
    lon_delta = radius_km / (111 * abs(lat)) if lat != 0 else radius_km / 111  # Avoid division by zero
    bbox = (lat - lat_delta, lon - lon_delta, lat + lat_delta, lon + lon_delta)
    query = f"""
    [out:json];
    node["amenity"="police"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    out body;
    """
    for attempt in range(3):  # Retry up to 3 times
        try:
            response = requests.post(overpass_url, data={"data": query}, headers={"User-Agent": "CyberbullyingDetectionApp/1.0"}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "elements" in data and data["elements"]:
                    stations = []
                    for element in data["elements"]:
                        stations.append({
                            "name": element.get("tags", {}).get("name", "Unnamed Police Station"),
                            "lat": element["lat"],
                            "lon": element["lon"],
                            "phone": element.get("tags", {}).get("phone", "N/A")
                        })
                    print(f"Found {len(stations)} police stations near ({lat}, {lon})")
                    return stations
                else:
                    print(f"No police stations found near ({lat}, {lon})")
                    return []
            else:
                print(f"Overpass API failed with status {response.status_code}: {response.text}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Overpass API error on attempt {attempt + 1}: {str(e)}")
            if attempt < 2:
                time.sleep(1)  # Respect Overpass API rate limits
            continue
    return []

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        age = request.form.get('age', type=int)
        city = request.form['city']
        gender = request.form['gender']
        parent_email = request.form.get('parent_email', '')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if not age or age < 5 or age > 100:
            flash('Please enter a valid age between 5 and 100')
            return redirect(url_for('register'))
        
        if age <= 16 and not parent_email:
            flash('Parent email is required for users under 17 years old')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(
            username=username, 
            password=hashed_password,
            age=age,
            city=city,
            gender=gender,
            parent_email=parent_email if age <= 16 else None
        )
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    return render_template('dashboard.html', user_age=user.age)

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    if 'username' not in session:
        return redirect(url_for('login'))
    text_prediction = None
    text_reason = None
    if request.method == 'POST':
        user_input = request.form['text']
        if user_input.strip():
            is_bullying, detection_reason = is_cyberbullying(user_input)
            text_prediction = 1 if is_bullying else 0
            text_reason = detection_reason
            record_user_activity(
                username=session['username'],
                activity_type='text',
                content=user_input,
                result=is_bullying,
                reason=detection_reason
            )
    user = User.query.filter_by(username=session['username']).first()
    return render_template('dashboard.html', 
                          text_prediction=text_prediction, 
                          text_reason=text_reason,
                          active_tab='text',
                          user_age=user.age)

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'username' not in session:
        return redirect(url_for('login'))
    image_prediction = None
    image_sentiment = None
    uploaded_image_path = None
    detection_reason = None
    extracted_text = None
    if 'image' not in request.files:
        flash('No image part')
        return redirect(url_for('dashboard'))
    file = request.files['image']
    if file.filename == '':
        flash('No image selected')
        return redirect(url_for('dashboard'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        is_bullying, sentiment_value, detection_reason, extracted_text = process_image_analysis(file_path)
        image_prediction = 1 if is_bullying else 0
        image_sentiment = round(sentiment_value, 2)
        uploaded_image_path = url_for('static', filename=f'uploads/{unique_filename}')
        record_user_activity(
            username=session['username'],
            activity_type='image',
            content=file_path,
            result=is_bullying,
            reason=detection_reason
        )
    user = User.query.filter_by(username=session['username']).first()
    return render_template('dashboard.html', 
                          image_prediction=image_prediction, 
                          image_sentiment=image_sentiment, 
                          uploaded_image_path=uploaded_image_path,
                          detection_reason=detection_reason,
                          extracted_text=extracted_text,
                          active_tab='image',
                          user_age=user.age)

@app.route('/get_nearby_stations')
def get_nearby_stations():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user = User.query.filter_by(username=session['username']).first()
    if user.age <= 16:
        return jsonify({'error': 'Feature not available'}), 403
    
    lat, lon = geocode_location("India", user.city)  # Assuming India; adjust as needed
    if lat is None or lon is None:
        return jsonify({'error': f'Could not geocode location: {user.city}, India'}), 500
    
    stations = get_police_stations(lat, lon)
    if not stations:
        return jsonify({'error': f'No police stations found near {user.city}, India'}), 404
    return jsonify({'stations': stations})

@app.route('/open_map', methods=['GET', 'POST'])
def open_map():
    if 'username' not in session:
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if user.age <= 16:
        return "Feature not available for users under 17", 403
    
    if request.method == 'POST':
        country = request.form['country']
        city = request.form['city']
        lat, lon = geocode_location(country, city)
        if lat is None or lon is None:
            flash(f'Could not find location: {city}, {country}')
            return render_template('map.html', error=f'Could not find location: {city}, {country}')
        stations = get_police_stations(lat, lon)
        if not stations:
            flash(f'No police stations found near {city}, {country}')
        return render_template('map.html', lat=lat, lon=lon, stations=stations)
    return render_template('map.html')

@app.route('/logout')
def logout():
    username = session.get('username')
    if username and user_requires_reporting(username):
        user = User.query.filter_by(username=username).first()
        if user:
            report_path = generate_activity_report(user.id)
            if report_path:
                send_report_email(user.id, report_path)
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)