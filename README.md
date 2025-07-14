# 🏠 House Price Prediction using Machine Learning

This is a Machine Learning project that predicts house prices using the California Housing dataset. The model is deployed as a web app using Streamlit.

## 📁 Project Structure

```
house-price-prediction/
├── app.py                 # Streamlit app interface for live prediction
├── main.py                # Model training and evaluation script
├── linear_model.pkl       # Trained Linear Regression model (saved)
├── feature_names.pkl      # Feature list used in the model
├── requirements.txt       # List of required Python packages
├── README.md              # Project documentation (this file)
└── .gitignore             # Ensures unnecessary files (like venv) are ignored
```

## 📊 Features Used

- `MedInc` (Median Income)
- `HouseAge` (Average age of houses)
- `AveRooms` (Average number of rooms)
- `AveBedrms` (Average number of bedrooms)
- `Population`
- `AveOccup` (Average house occupancy)
- `Latitude` and `Longitude`

## 🤖 Model

- **Type**: Linear Regression
- **Performance**:
  - R² Score: ~0.60
  - RMSE: ~0.72

## 🧪 How to Run Locally

1. **Clone the Repository**

```bash
git clone https://github.com/shreyagrawal09/house-price-prediction.git
cd house-price-prediction
```

2. **Create and Activate Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# OR
source venv/bin/activate     # On Mac/Linux
```

3. **Install Required Libraries**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit App**

```bash
streamlit run app.py
```
<--
## 🌐 Live App

> **Coming Soon:** Will be available after deployment on [Streamlit Cloud](https://streamlit.io/cloud).
-->
## 🚀 Future Enhancements

- Use advanced models (Random Forest, XGBoost)
- Add map-based visualization
- Deploy to custom domain or Heroku
- Add multiple model support

## 👨‍🎓 Author

- **Name**: Shrey Agrawal 
- **Email**: agrawalshrey02@gmail.com
- **College**: Pranveer Singh Institute of Technology

