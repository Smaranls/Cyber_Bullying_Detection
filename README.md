Cyberbullying Detection System

A Flask-based web application that detects cyberbullying in text and images, tracks user activity, and sends reports to parents for users under 17. It also helps users over 16 find nearby police stations using geolocation APIs.

Features





Text Analysis: Checks text for bullying using sentiment analysis and harmful keyword detection.



Image Analysis: Identifies harmful objects (e.g., weapons) and extracts text from images using OCR and deep learning (MobileNetV2).



User Management: Supports user registration, login, and activity tracking with a SQLite database.



Reporting: Generates PDF reports of user activities and emails them to parents for underage users.



Geolocation: Finds nearby police stations using Nominatim and Overpass APIs.



Secure: Uses environment variables for sensitive data and restricts file uploads to safe formats.

Prerequisites

Before setting up the project, ensure you have:





Python: Version 3.8 or higher (Download).



Tesseract OCR: For extracting text from images.





Windows: Download and install from Tesseract Releases. Note the installation path (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe).



Linux: Install with sudo apt-get install tesseract-ocr.



macOS: Install with brew install tesseract.



Gmail Account: For sending email reports. If 2-factor authentication (2FA) is enabled, you’ll need an App Password.



Git: To clone the repository (Download).



A Code Editor: Like VS Code or Notepad (optional but helpful).