Day 06: Medical Cost Prediction (Regression Project)

This project predicts **medical insurance costs** using Machine Learning models based on patient demographics and lifestyle data.  
It’s part of my ongoing **64-Day AI/ML Challenge** — a journey to master AI/ML through hands-on projects.

📘 Project Overview

We use the **Insurance Charges Dataset** to estimate a person’s medical expenses given:
- Age  
- BMI  
- Smoking status  
- Number of children  
- Region  
- Gender  

🧠 Models Used

1. **Linear Regression** – Baseline model for predictions  
2. **Random Forest Regressor** – Non-linear model for higher accuracy  

 Evaluation Metrics:
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- R² (Explained Variance)

 ⚙️ Tech Stack

Python | Pandas | Scikit-Learn | Matplotlib | Seaborn | VS Code | GitHub

 📊 Results

| Model | MAE | RMSE | R² |
|--------|------|------|------|
| Linear Regression | ~4181 | ~5796 | 0.78 |
| Random Forest | ~2571 | ~4592 | 0.86 |

✅ The **Random Forest model** performed best.
