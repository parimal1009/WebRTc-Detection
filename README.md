# WebRTC Real-Time Object Detection Demo

## Overview

This project showcases an **Ultra-Advanced Real-Time Object Detection** system that operates with exceptional speed and precision. The backend is built using **FastAPI**, a modern, high-performance Python web framework, which handles the server logic and API endpoints. For live video streaming, the system leverages **WebRTC (Web Real-Time Communication)**, a powerful technology that enables peer-to-peer video and data transfer directly in the browser, minimizing latency. At its core, the system utilizes an **ensemble of state-of-the-art models**, including the renowned YOLO family and advanced Transformer-based detectors, to ensure robust and accurate object detection. This combination of technologies delivers maximum detection accuracy with perfectly refined bounding boxes.

### Demo Photo

![Screenshot 2025-08-21 201729](https://github.com/parimal1009/WebRTc-Detection/blob/main/image%20and%20video/Screenshot%202025-08-21%20201729.png?raw=true)

## Features

* **Real-time object detection via browser WebRTC streaming:** Streams live video from a webcam directly to the server for processing, enabling instant visual feedback without the need for additional software.

* **Ensemble of multiple SOTA detection models for robust results:** Combines the strengths of different models (e.g., YOLO's speed and Transformer's accuracy) to achieve superior performance across a wide range of detection tasks.

* **Advanced image enhancement and multi-scale pyramid processing:** Before analysis, the system applies techniques like adaptive contrast and sharpening to improve image quality, ensuring that objects are clearly visible even in poor lighting. Multi-scale processing allows the models to detect objects of various sizes efficiently.

* **Dynamic object recognition and label refinement using AI models:** Integrates a Large Language Model (LLM) to perform contextual analysis on detected objects, refining labels and providing more detailed descriptions of the scene.

* **Comprehensive performance metrics and visualization in real-time:** Displays key performance indicators such as frames per second (FPS), processing time, and model-specific metrics on the user interface, providing a clear view of the system's efficiency.

* **WebSocket communication for live updates and control:** A WebSocket connection maintains a persistent link between the browser and the server, allowing for real-time updates on detections and enabling users to change model settings instantly.

* **Easy switching of detection models and confidence thresholds:** The web interface provides dropdown menus to quickly switch between available models and adjust the confidence threshold, giving users control over the detection sensitivity.

* **Health checks and extended scene analysis support:** Includes API endpoints for monitoring the server's health and provides advanced analysis capabilities that can interpret the relationships between detected objects in a scene.

## Demo Video

Watch the full demonstration of the real-time detection in action: [Demo Video on Google Drive](https://drive.google.com/file/d/1hnklfXK3nU7j4T79jqWOOn4d0UXAl8Ka/view?usp=sharing)

## Prerequisites

* **Python 3.8+:** The project is developed and tested with Python 3.8 and above.

* **CUDA-enabled GPU recommended for best performance (optional):** A NVIDIA GPU with CUDA support is highly recommended to leverage GPU acceleration, which significantly improves the detection speed. The application will still function on a CPU, but performance will be limited.

* **Required Python packages listed in `requirements.txt`:** All necessary dependencies, including `ultralytics`, `transformers`, `fastapi`, and `uvicorn`, are provided in this file for easy installation.

## Installation

1. Clone the repository:

   ```
   git clone [https://github.com/parimal1009/WebRTc-Detection.git](https://github.com/parimal1009/WebRTc-Detection.git)
   cd WebRTc-Detection
   ```

2. Create and activate a virtual environment:

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Set environment variables for API keys if using Groq or Hugging Face APIs:

   ```
   export GROQ_API_KEY="your_groq_api_key"
   export HF_TOKEN="your_huggingface_token"
   ```


````markdown
# WebRTC Real-Time Object Detection Demo

## Overview

This project showcases an **Ultra-Advanced Real-Time Object Detection** system that operates with exceptional speed and precision. The backend is built using **FastAPI**, a modern, high-performance Python web framework, which handles the server logic and API endpoints. For live video streaming, the system leverages **WebRTC (Web Real-Time Communication)**, a powerful technology that enables peer-to-peer video and data transfer directly in the browser, minimizing latency. At its core, the system utilizes an **ensemble of state-of-the-art models**, including the renowned YOLO family and advanced Transformer-based detectors, to ensure robust and accurate object detection. This combination of technologies delivers maximum detection accuracy with perfectly refined bounding boxes.

### Demo Photo

![Screenshot 2025-08-21 201729](https://github.com/parimal1009/WebRTc-Detection/blob/main/image%20and%20video/Screenshot%202025-08-21%20201729.png?raw=true)

## Features

* **Real-time object detection via browser WebRTC streaming:** Streams live video from a webcam directly to the server for processing, enabling instant visual feedback without the need for additional software.

* **Ensemble of multiple SOTA detection models for robust results:** Combines the strengths of different models (e.g., YOLO's speed and Transformer's accuracy) to achieve superior performance across a wide range of detection tasks.

* **Advanced image enhancement and multi-scale pyramid processing:** Before analysis, the system applies techniques like adaptive contrast and sharpening to improve image quality, ensuring that objects are clearly visible even in poor lighting. Multi-scale processing allows the models to detect objects of various sizes efficiently.

* **Dynamic object recognition and label refinement using AI models:** Integrates a Large Language Model (LLM) to perform contextual analysis on detected objects, refining labels and providing more detailed descriptions of the scene.

* **Comprehensive performance metrics and visualization in real-time:** Displays key performance indicators such as frames per second (FPS), processing time, and model-specific metrics on the user interface, providing a clear view of the system's efficiency.

* **WebSocket communication for live updates and control:** A WebSocket connection maintains a persistent link between the browser and the server, allowing for real-time updates on detections and enabling users to change model settings instantly.

* **Easy switching of detection models and confidence thresholds:** The web interface provides dropdown menus to quickly switch between available models and adjust the confidence threshold, giving users control over the detection sensitivity.

* **Health checks and extended scene analysis support:** Includes API endpoints for monitoring the server's health and provides advanced analysis capabilities that can interpret the relationships between detected objects in a scene.

## Demo Video

Watch the full demonstration of the real-time detection in action: [Demo Video on Google Drive](https://drive.google.com/file/d/1hnklfXK3nU7j4T79jqWOOn4d0UXAl8Ka/view?usp=sharing)

## Prerequisites

* **Python 3.8+:** The project is developed and tested with Python 3.8 and above.

* **CUDA-enabled GPU recommended for best performance (optional):** A NVIDIA GPU with CUDA support is highly recommended to leverage GPU acceleration, which significantly improves the detection speed. The application will still function on a CPU, but performance will be limited.

* **Required Python packages listed in `requirements.txt`:** All necessary dependencies, including `ultralytics`, `transformers`, `fastapi`, and `uvicorn`, are provided in this file for easy installation.

## Installation

1. Clone the repository:

   ```
   git clone [https://github.com/parimal1009/WebRTc-Detection.git](https://github.com/parimal1009/WebRTc-Detection.git)
   cd WebRTc-Detection
   ```

2. Create and activate a virtual environment:

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Set environment variables for API keys if using Groq or Hugging Face APIs:

   ```
   export GROQ_API_KEY="your_groq_api_key"
   export HF_TOKEN="your_huggingface_token"
   ```

## Running the Application

Start the server locally with the following command:

````

./start.sh

```

Then open your browser and navigate to [http://localhost:8000](https://www.google.com/search?q=http://localhost:8000) to access the live object detection interface.

## Usage

* **Stream your webcam or upload images via the WebRTC interface:** Simply allow webcam access in your browser to start the live stream, or use the file upload feature to analyze static images.

* **Switch detection models and adjust confidence thresholds in real-time:** Use the dropdown menus in the UI to select a different model or fine-tune the minimum confidence score required for a detection to be displayed.

* **View detection results with bounding boxes and live performance metrics:** The system will draw a bounding box around each detected object and display its label and confidence score. Performance metrics are updated in real-time.

* **Analyze scenes with enhanced AI-powered recognition:** For more complex scenes, the integrated LLM can provide a narrative description of the objects and their relationships.

## Troubleshooting

* **Ensure dependencies installed match your Python version:** Incompatibilities between package versions and your Python interpreter can cause issues.

* **If GPU is available but not detected, verify CUDA drivers:** Make sure your NVIDIA drivers and CUDA toolkit are correctly installed and configured.

* **API-based models require valid API keys and network connectivity:** Ensure your `GROQ_API_KEY` or `HF_TOKEN` environment variables are set correctly and that you have an active internet connection.

## License

This project is licensed under the MIT License.

Built by Parimal, leveraging `ultralytics`, `transformers`, `Groq`, and `FastAPI`.
```

## Running the Application

Start the server locally with the following command:
