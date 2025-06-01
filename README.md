🧩 Data Visualization — Real Estate Pricing Project

This repository contains visual analysis of a real estate pricing dataset (~6000 rows) used to train and evaluate a machine learning model for predicting apartment prices in Moscow based on various features such as area, location, year of construction, number of rooms, and more.

The visualization was created using Jupyter Notebook with libraries like pandas, matplotlib, seaborn, and plotly.

🧩 Analysis Topics Covered:
1. Correlation Matrix with New Features
Visualized the correlation between newly engineered features (like age, floor_ratio, price_per_sqm, etc.) and target variable (price) to understand feature importance and redundancy.

2. Correlation Matrix (Only Numerical Values)
A simplified version focusing only on numerical features to highlight linear relationships without noise from categorical or derived features.

3. PERCENTAGE RATIOS (Pie Charts)
Explored categorical distributions using pie charts:

Percentage of new buildings and non-new buildings 
Distribution by number of rooms
Number of ads by regions of Moscow

4. DISTRIBUTION OF PRICES
Histograms and KDE plots showing the distribution of apartment prices before and after outlier removal.

5. PRICE - AREA RELATIONSHIP
Several scatter plots with trend lines.

6. PRICE PER SQUARE METER
Introducing a new feature via bar plot and line plot.

7. IMPACT OF PROXIMITY TO THE SUBWAY

8. FLOOR ANALYSIS

9. IMPACT OF THE YEAR OF CONSTRUCTION

10. ANALYSIS BY NUMBER OF ROOMS

11. LIVING AREA TO TOTAL AREA RATIO

📁 Dataset Info
Size: ~6000 entries
Key features:
price: Target variable
total_area, living_area, ceiling_height
min_to_metro: Distance to nearest subway station in minutes
number_of_rooms
construction_year
is_new, is_apartments
floor, number_of_floors
region_of_moscow, link

🧩 Tools Used
Python (Pandas, NumPy, Matplotlib, Seaborn, Plotly)
Jupyter Notebook (for exploratory data analysis and visualization)

🧩 Notes
The visualizations were created during EDA (Exploratory Data Analysis) phase.
This analysis helped inform feature engineering and preprocessing steps for the final ML model trained separately in PyCharm.

🧩 Related Repository
For the full machine learning pipeline (model training, evaluation, prediction), check out the master branch or the linked project folder in your local setup.
