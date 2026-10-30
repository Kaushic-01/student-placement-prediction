# Student Placement Prediction

## Dataset columns
- CGPA (0-10)
- AptitudeScore (0-100)
- TechnicalScore (0-100)
- Projects (number of projects)
- Internship (0 = No, 1 = Yes)
- Placed (target: 0 = Not Placed, 1 = Placed)

## Run locally
1. Install dependencies: `python -m pip install -r requirements.txt`
2. Train the model (optional): `python train_model.py`
3. Run the app: `python -m streamlit run app.py`

The Streamlit app needs `app.py`, `placement_model.pkl`, and `requirements.txt`.
Keep student-level data private unless you have permission to publish it.
This is an educational demonstration, not a decision-making tool.
