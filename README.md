# Agricultural AI Suite: Crop, Fertilizer, and Disease Prediction Models

This repository contains a suite of three machine learning models designed to assist in modern farming. Each model is developed and deployed as a web application with a simple user interface.

1. **Crop Recommendation Model**: Recommends the best crop to grow based on soil and weather conditions.

2. **Fertilizer Recommendation Model**: Suggests the optimal fertilizer for a given crop and soil profile.

3. **Plant Disease Prediction Model**: Identifies diseases in plants from leaf images.

---
## ⚙️ Model Development Techniques

Each model was built using specific techniques tailored to its task.

### 1. Crop & Fertilizer Recommendation Models (Classical ML)

These two models solve tabular data problems and were built using the `scikit-learn` library.

* **Algorithm**: Both models use a **Random Forest Classifier**. This is an ensemble learning method that operates by constructing a multitude of decision trees at training time. For a prediction, it outputs the class that is the mode of the classes of the individual trees, making it robust and accurate.

* **Data Preprocessing**: A `scikit-learn` **Pipeline** was used to streamline the preprocessing steps. This ensures that the same transformations are applied consistently during both training and prediction. The pipeline consists of:

  * **`StandardScaler`**: Applied to all numerical features (like Temperature, Moisture, Nitrogen, etc.) to scale the data, giving it a mean of 0 and a standard deviation of 1. This is crucial for the performance of many ML algorithms.

  * **`OneHotEncoder`**: Applied to all categorical features (like Soil Type, Crop Type). This converts categorical text data into a numerical format that the model can understand, without implying any ordinal relationship between categories.

* **Model Persistence**: After training, the entire `Pipeline` object (containing both the preprocessor and the trained model) and the `LabelEncoder` (for the target variable) were saved to disk as `.joblib` files. This allows us to load the complete, ready-to-use model in our application without retraining.

### 2. Plant Disease Prediction Model (Deep Learning)

This model solves a computer vision problem and was built using the `PyTorch` deep learning framework.

* **Architecture**: The model is based on **MobileNetV2**, a state-of-the-art convolutional neural network (CNN) designed for high efficiency and performance, especially on mobile or low-power devices.

* **Transfer Learning**: Instead of a training a new model from scratch, we used a MobileNetV2 model that was **pre-trained** on the massive ImageNet dataset. This technique, known as transfer learning, leverages the knowledge (like edge, texture, and shape detection) the model has already gained. We only needed to train the final layers to adapt it to our specific task of identifying plant diseases.

* **Custom Classifier Head**: The original classifier layer of MobileNetV2 was replaced with a new `nn.Sequential` block in PyTorch. This new head consists of a Dropout layer (to prevent overfitting) and a new Linear layer that outputs predictions for our 38 unique plant disease classes.

* **Image Augmentation**: To make the model more robust and prevent it from memorizing the training images, we applied several random transformations to the training data using `torchvision.transforms`. These included:

  * Resizing all images to 224x224 pixels.

  * Random horizontal flips.

  * Random rotations.

* **Training**: The model was trained for 10 epochs using the **Adam optimizer** and **Cross-Entropy Loss** function, which are standard choices for multi-class classification tasks.

---
## 🚀 Deployment Process

All three models were deployed as interactive web applications on **Hugging Face Spaces** using the same modern technology stack.

1. **Web Framework (FastAPI)**: We used **FastAPI** to build a fast and efficient Python-based web server.

   * It serves the simple HTML/CSS/JS frontend for user interaction.

   * It provides a `/predict` API endpoint that receives user input as JSON, processes it, and returns the model's prediction.

   * A `lifespan` manager was used to load the ML models into memory on application startup, preventing timeouts on the Hugging Face platform.

2. **Containerization (Docker)**: The entire application, including the Python environment and all dependencies, was packaged into a **Docker** container.

   * The `Dockerfile` defines the blueprint for building the application image. It starts with a base Python image (`python:3.11-slim`), installs all necessary libraries from `requirements.txt`, copies the application code, and specifies the command to run the FastAPI server.

   * This ensures that the application runs in a consistent and reproducible environment, regardless of where it is deployed.

3. **Cloud Platform (Hugging Face Spaces)**:

   * We created a new **Docker Space** on Hugging Face for each application.

   * We uploaded the project files (`app.py`, `Dockerfile`, `requirements.txt`, `index.html`, and the model artifacts folder) to the Space's repository.

   * Hugging Face automatically detected the `Dockerfile`, built the container image, and deployed it.

   * **Critical Configuration**: We configured the `Dockerfile` to expose and run the Uvicorn server on **port `7860`**, which is the port required by Hugging Face Spaces for its internal routing and health checks.

---
## 🧪 Running Locally with Docker

You can test any of these applications on your local machine before deploying.

**Prerequisites**: You must have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

1.  **Get the Project Files**: Make sure you have the complete project folder for the model you want to test on your local machine.

2.  **Open a Terminal**: Navigate into the specific project directory (e.g., `/fertilizer-recommendation-app`) using your command line or terminal.

3.  **Build the Docker Image**: Run the following command to build the image. The `-t` flag lets you name the image (e.g., `fertilizer-app`), and the `.` tells Docker to use the `Dockerfile` in the current directory.
    ```bash
    docker build -t fertilizer-app .
    ```

4.  **Run the Docker Container**: Once the image is built, run this command to start the container. The `-p` flag maps your local port `7860` to the container's port `7860`.
    ```bash
    docker run -d -p 7860:7860 fertilizer-app
    ```

5.  **Access the Application**: Open your web browser and go to `http://localhost:7860`. You should see the application's user interface.

6.  **Stop the Container**: When you are finished, you can stop the running container. First, find its ID with `docker ps`, then use that ID to stop it.
    ```bash
    docker stop <your_container_id>
    ```


