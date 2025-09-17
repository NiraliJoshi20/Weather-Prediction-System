import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import json
from streamlit.components.v1 import html

# Load models and LabelEncoder
rain_model = joblib.load("rain_model.pkl")
temp_model = joblib.load("temp_model.pkl")
hum_model = joblib.load("hum_model.pkl")
le = joblib.load("label_encoder.pkl")

# OpenWeatherMap API
API_KEY = "c7e47a191dedf2e4f8655e21ec5e3072"

def fetch_weather_data(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()

    weather = {
        'current_temp': round(data['main']['temp']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'pressure': data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed'],
        'wind_gust_dir': data.get('wind', {}).get('deg', 0),
        'city': data['name'],
        'country': data['sys']['country'],
        'description': data['weather'][0]['description'].title()
    }
    return weather

# Streamlit UI
st.set_page_config(page_title="🌦️ Weather Predictor", layout="centered")
st.title("🌍 Live Weather Prediction System")

city = st.text_input("Enter City Name", "")

if st.button("Predict Weather Conditions"):
    weather_data = fetch_weather_data(city)
    if weather_data:
        # Display current weather
        st.subheader(f"📍 {weather_data['city']}, {weather_data['country']}")
        st.write(f"**🌤️ Weather Description**: {weather_data['description']}")
        st.write(f"**🌡️ Temperature**: {weather_data['current_temp']} °C")
        st.write(f"**💧 Humidity**: {weather_data['humidity']}%")
        st.write(f"**🔽 Pressure**: {weather_data['pressure']} hPa")
        st.write(f"**🌬️ Wind Speed**: {weather_data['Wind_Gust_Speed']} m/s")

        # Encode wind direction
        wind_dog = weather_data['wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]
        compass_direction = next(point for point, start, end in compass_points if start <= wind_dog < end)
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        # Prepare input for rain_model
        current_data = {
            'MinTemp': weather_data['temp_min'],
            'MaxTemp': weather_data['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': weather_data['Wind_Gust_Speed'],
            'Humidity': weather_data['humidity'],
            'Pressure': weather_data['pressure'],
            'Temp': weather_data['current_temp']
        }
        current_df = pd.DataFrame([current_data])

        # Make predictions
        rain_pred = rain_model.predict(current_df)[0]
        temp_pred = temp_model.predict(np.array([[weather_data['current_temp']]]))[0]
        hum_pred = hum_model.predict(np.array([[weather_data['humidity']]]))[0]

        # Display predictions
        st.success(f"🌧️ **Rain Prediction**: {'Yes' if rain_pred == 1 else 'No'}")
        st.success(f"🌡️ **Predicted Next Temperature**: {round(temp_pred, 2)} °C")
        st.success(f"💧 **Predicted Next Humidity**: {round(hum_pred, 2)}%")

        # Comparison chart data
        chart_data = {
            "labels": ["Temperature (°C)", "Humidity (%)"],
            "datasets": [
                {
                    "label": "Actual",
                    "data": [weather_data["current_temp"], weather_data["humidity"]],
                    "backgroundColor": "rgba(54, 162, 235, 0.5)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 1
                },
                {
                    "label": "Predicted",
                    "data": [temp_pred, hum_pred],
                    "backgroundColor": "rgba(255, 99, 132, 0.5)",
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "borderWidth": 1
                }
            ]
        }

        chart_config = {
            "type": "bar",
            "data": chart_data,
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Value"
                        }
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Metric"
                        }
                    }
                },
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top"
                    },
                    "title": {
                        "display": True,
                        "text": "Actual vs Predicted Weather Metrics"
                    }
                }
            }
        }

        # Render chart using st.components.v1.html
        st.subheader("📊 Actual vs Predicted")
        st.markdown("### Bar Chart")

        # HTML and JavaScript to render the chart
        chart_html = f"""
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <canvas id="myChart" style="max-height: 400px; max-width: 600px;"></canvas>
            <script>
                const ctx = document.getElementById('myChart').getContext('2d');
                const chartConfig = {json.dumps(chart_config)};
                new Chart(ctx, chartConfig);
            </script>
        </body>
        </html>
        """
        html(chart_html, height=450, width=650)

    else:
        st.error("City not found or API limit reached. Please try again.")