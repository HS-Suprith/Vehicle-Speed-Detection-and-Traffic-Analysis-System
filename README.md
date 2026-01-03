# 🏆 RoadGuardian AI: The Ultimate Traffic Analysis Tool

An advanced, real-time traffic monitoring tool built with Python, YOLOv8, and Streamlit. This application can track, count, and analyze vehicles from a video feed, providing rich insights through an interactive dashboard.

![Demo GIF](link_to_your_demo_gif.gif)  ## ✨ Features

- **Multi-Class Vehicle Counting:** Tracks and counts Cars, Trucks, Buses, and Motorcycles separately.
- **Real-time Speed Estimation:** Calculates the speed of each vehicle in km/h.
- **Direction Detection:** Determines if a vehicle is moving Up or Down in the frame.
- **Traffic Density Analysis:** Classifies traffic as Low, Medium, or High.
- **Abnormal Event Detection:** Flags vehicles that are stopped illegally for more than 5 seconds.
- **Traffic Heatmap:** Generates a live heatmap to visualize traffic hotspots.
- **Interactive Dashboard:** A clean Streamlit UI with live charts and metrics.
- **Data Export:** All detected events can be exported to a CSV or are saved in a local SQLite database.

## 🛠️ Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Streamlit
- Pandas & NumPy
- SQLite3

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/RoadGuardianAI.git](https://github.com/your-username/RoadGuardianAI.git)
    cd RoadGuardianAI
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` file aapko `pip freeze > requirements.txt` command se banani hogi)*

## 🏃‍♂️ How to Run

Navigate to the project's root directory in your terminal and run the following command:

```bash
streamlit run src/app.py