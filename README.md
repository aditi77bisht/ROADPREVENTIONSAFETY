ROAD_PREVENTION_SAFETY
AI-Based Roadkill Risk Prediction in Uttarakhand ( BASICALLY THE GHARWAL REGION )
This project focuses on predicting high-risk areas for animal-vehicle collisions (roadkill) in Uttarakhand using Machine Learning (ML) and Geospatial Mapping. The system analyzes factors like road type, traffic, wildlife presence, weather, road curvature, speed limits, and time of day to predict accident-prone locations.

The output is an interactive map highlighting high-risk (red) and low-risk (green) zones to help authorities, conservationists, and drivers take preventive measures.
THIS PROJECT IS MAINLY USING THE GHARWAL REGION AS IT CONTAINS A SYNTHETIC DATASET .
YOU CAN ALSO APPLY TO REAL TIME DATASET WHICH CAN GIVE YOU MORE ACCURATE RESULT AND CAN BE APPLIED TO DIFFERENT REGIONS AS WELL.

Technologies Used
1. Machine Learning (Random Forest)
Algorithm: Random Forest Classifier (20 trees)
Purpose: Classifies locations into high-risk or low-risk zones based on input factors.
Evaluation Metrics: Accuracy, Confusion Matrix, and Classification Report.
2. Data Processing (Pandas & NumPy)
Handles the dataset (uttarakhand_synthetic_roadkill_data.csv).
Feature Engineering: Adds road curvature, speed limits, and time of day.
Encodes categorical data (morning, afternoon, night → numerical values).
3.  Geolocation (Geopy & Folium)
Geopy: Extracts place names from latitude/longitude.
Folium: Creates an interactive risk map for visualization.
4. Visualization (Matplotlib & Seaborn)
Confusion Matrix Heatmap: Evaluates model performance.
Risk Trend Plots: Shows accident frequency based on the time of day.


This project uses AI and geospatial analysis to reduce wildlife-vehicle collisions, helping in wildlife conservation and road safety.
