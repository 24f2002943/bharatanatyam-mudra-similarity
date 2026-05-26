# Bharatanatyam Mudra Similarity & Feedback System

A web-based application designed to help users learn, practice, and perfect Bharatanatyam hand gestures (Mudras). The system uses machine learning and computer vision to identify hand gestures in real-time, providing instant feedback and confidence scores to the practitioner.

## Features

- **Real-Time Mudra Recognition**: Uses webcam input to detect and classify hand gestures instantly.
- **Mudra Library**: A comprehensive visual dictionary of traditional Bharatanatyam mudras along with their cultural meanings and applications.
- **Practice Mode**: Interactive mode where users can perform mudras in front of the camera and receive real-time classification and accuracy feedback.
- **Machine Learning Powered**: Utilizes a trained classification model to accurately map 3D hand landmarks (21 points) to specific mudras.

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, Vanilla JavaScript
- **Computer Vision**: Google MediaPipe (Hand Landmark Detection)
- **Machine Learning**: Scikit-Learn (Classification Model), NumPy

## Project Structure

- `app.py`: Main Flask application handling routes and prediction API.
- `data/`: Contains the trained machine learning model (`mudra_classifier.pkl`) and label encoders. Note: large models may not be tracked in version control due to file size limits.
- `scripts/`: Python utility scripts for model training, evaluation, and landmark extraction.
- `static/`: Contains static assets (CSS styles, JS logic, and mudra reference images).
- `templates/`: HTML templates for the web interface.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/24f2002943/bharatanatyam-mudra-similarity.git
   cd bharatanatyam-mudra-similarity
   ```

2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate
   ```

3. **Install Dependencies:**
   Ensure you have the required Python libraries installed:
   ```bash
   pip install flask numpy scikit-learn
   ```

4. **Add Model Files:**
   Ensure that `mudra_classifier.pkl` and `label_encoder.pkl` are present in the `data/` directory.

5. **Run the Application:**
   ```bash
   python app.py
   ```

6. **Access the App:**
   Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage

- **Home**: Welcome page introducing the application.
- **Library**: Browse through various mudras and learn their meanings (e.g., *Pathaka*, *Anjali*, *Alapadmam*).
- **Practice**: Allow camera access, perform a mudra, and the system will try to recognize it and display the predicted mudra and its meaning.

## License

This project is open-source and available for educational purposes.
