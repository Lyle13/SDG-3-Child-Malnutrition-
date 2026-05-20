# 🌍 SDG 3: Child Nutrition Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Analytics Techniques and Tools — Finals ALA** *WVSU Information Systems | Business Analytics* 🔗 **[Live Dashboard: View the App Here](https://m4qzcxoetrwixcpmwew6sb.streamlit.app)**

---

## 📌 Project Overview
The world today faces complex challenges regarding health and well-being. This project focuses on **Sustainable Development Goal 3 (SDG 3)**, specifically investigating the global drivers of child and adolescent malnutrition. 

Rather than relying purely on visualization, this project employs inferential statistics and robust regression models to answer the core research question:
> **"What factors influence thinness prevalence among children and adolescents across countries over time?"**

The findings from the statistical models are operationalized into an interactive web dashboard built with Streamlit and Plotly, allowing users to dynamically explore global nutrition trends, identify key drivers, and compare country-level data.

## 📊 Data Source
* **Primary Source:** [UNICEF Global Database on Child Nutrition](https://data.unicef.org/) (SACA Series, August 2025)
* **Age Group:** Children and Adolescents (5–19 years)
* **Temporal Coverage:** 2000–2022
* **Geographical Coverage:** 194 UN-recognized countries

## 🧠 Methodology & Analytical Workflow

1. **Data Preparation:** Merged separate UNICEF datasets for Overweight, Obesity, and Thinness.
   * Cleaned unclassified regional aggregates and standardized ISO-3 country codes.
   * Addressed structural missing values (e.g., mapping USA and Canada to North America).
2. **Exploratory Data Analysis (EDA):**
   * Assessed multicollinearity using Variance Inflation Factors (VIF).
   * Evaluated residual normality via Q-Q plots and the Shapiro-Wilk test.
3. **Regression Analysis:**
   * Detected heteroscedasticity using the Breusch-Pagan test.
   * Applied an **HC3 Robust Multiple Regression Model** to correct for varying variance across income groups and accurately determine the significant predictors of child thinness.
4. **Dashboard Development:**
   * Developed a time-series interactive dashboard using Python, Streamlit, and Plotly Express to visualize the regression insights dynamically.

## 🛠️ Tech Stack
* **Data Processing & Modeling:** `pandas`, `numpy`, `statsmodels`, `scipy`
* **Visualization:** `plotly`, `matplotlib`, `seaborn`
* **Web Framework:** `streamlit`
* **Environment:** `Jupyter Notebook`

## 🗂️ Repository Structure
```text
├── SDG3_Regression_Analysis.ipynb    # Core Jupyter Notebook containing data cleaning, EDA, and robust regression
├── dashboard.py                      # Streamlit application script for the interactive dashboard
├── requirements.txt                  # Python dependencies for deployment
├── unicef_child_nutrition_merged.csv # Cleaned and merged analytical dataset
