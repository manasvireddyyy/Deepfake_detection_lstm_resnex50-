# Deepfake Detection using Deep Learning (ResNext and LSTM)

## Project Overview
This project aims to detect deepfake videos using a Deep Learning approach combining **ResNext50** for feature extraction and **LSTM (Long Short-Term Memory)** for sequence analysis. The system processes video frames to identify potential manipulations and classifies videos as "REAL" or "FAKE".

## System Architecture
![System Architecture](System%20Architecture.png)

## Prerequisites
- Python 3.8+
- Django 5.0+
- PyTorch
- CUDA (optional, for GPU support)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Deepfake-detection-using-Deep-Learning-ResNext-and-LSTM-
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Model Setup

> [!IMPORTANT]
> You must download the pre-trained model before running the application.

1.  **Download the Model:**
    [Download Model Here](https://drive.google.com/file/d/13j383gCWVPXcf50iwdxzaxWwpD6cSpsT/view?usp=sharing)

2.  **Place the Model:**
    -   Create a folder named `models` in the root directory of the project.
    -   Place the downloaded `.pt` file inside the `models` folder.

    Structure should look like this:
    ```
    Deepfake-detection-using-Deep-Learning-ResNext-and-LSTM-/
    ├── models/
    │   └── <model_filename>.pt
    ├── ml_app/
    ├── manage.py
    ├── ...
    ```

## Usage

1.  **Run the Django development server:**
    ```bash
    python manage.py runserver
    ```

2.  **Access the application:**
    Open your web browser and go to `http://127.0.0.1:8000/`.

3.  **Detect Deepfakes:**
    -   Upload a video file through the web interface.
    -   Enter the sequence length (default suggested: 100).
    -   Wait for the processing to complete.
    -   View the result ("REAL" or "FAKE") along with the confidence score and heatmaps.

## Features
-   **Video Preprocessing**: Splits video into frames and crops faces.
-   **Deep Learning Model**: Utilizes ResNext50 for spatial features and LSTM for temporal features.
-   **Heatmap Generation**: Visualizes the focus areas of the model.
-   **Web Interface**: User-friendly Django-based web application.

## Technologies Used
-   **Backend**: Django, Python
-   **Machine Learning**: PyTorch, torchvision, face_recognition, OpenCV, NumPy
-   **Frontend**: HTML, CSS, JavaScript

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
# Deepfake Detection Project
