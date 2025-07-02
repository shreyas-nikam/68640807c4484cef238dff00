
# Technical Specifications: Multiple Linear Regression Assumptions Analyzer

## Overview

This Streamlit application is designed to educate users on the fundamental assumptions underlying Multiple Linear Regression (MLR) and provide int\fractive tools to visually assess potential violations of these assumptions. By leveraging readily available datasets and statistical visualizations, users can gain \fractical insights into the diagnostic process of linear regression models.

The primary learning outcomes for users of this application are:
- Understanding the core assumptions of multiple linear regression.
- Ability to interpret residual plots and scatterplot matrices to identify potential violations of these assumptions.

The application will feature int\fractive components such as a scatterplot matrix, residuals vs. predicted values plot, and residuals vs. independent variables plots, all generated dynamically from selected datasets.

## Step-by-Step Development Process

The development of the "Multiple Linear Regression Assumptions Analyzer" Streamlit application will follow these steps:

### Step 1: Project Setup
- Create a new Python file, e.g., `app.py`.
- Create a `requirements.txt` file to list all necessary Python libraries.

### Step 2: Library Imports
- Import `streamlit` for the application framework.
- Import `pandas` for data handling and manipulation.
- Import `numpy` for numerical operations.
- Import `sklearn.datasets` to load built-in datasets.
- Import `sklearn.linear_model.LinearRegression` for model training.
- Import `sklearn.model_selection.train_test_split` for data splitting (though for assumption checking, training on full data might be preferred initially).
- Import `matplotlib.pyplot` and `seaborn` for creating visualizations.
- Import `statsmodels.api` for statistical models and diagnostic plots (specifically `sm.qqplot`).
- Import `statsmodels.stats.outliers_influence.variance_inflation_factor` and `statsmodels.tools.tools.add_constant` if VIF calculation is desired (optional for this spec, but good to note).

### Step 3: Data Loading and Preparation
- Define a function to load a selected dataset (e.g., `load_diabetes` from `sklearn.datasets`).
- The function should return a pandas DataFrame with features and target variable combined.
- Include a mechanism for the user to select which dataset to load.

### Step 4: User Input for Variable Selection
- Use Streamlit `st.selectbox` for the user to choose the dependent (target) variable.
- Use Streamlit `st.multiselect` for the user to choose multiple independent (feature) variables.
- Implement validation to ensure at least one independent variable is selected.

### Step 5: Multiple Linear Regression Model Training
- Split the selected data into independent variables ($X$) and the dependent variable ($Y$).
- Instantiate `LinearRegression` from `sklearn.linear_model`.
- Train the model using the selected $X$ and $Y$ data.
- Calculate predicted values ($\hat{Y}$) and residuals ($e_i = Y_i - \hat{Y_i}$).

### Step 6: Calculation of Diagnostic Metrics
- Calculate $R^2$ (coefficient of determination) from the trained model.
- Calculate residuals for plotting.

### Step 7: Implementation of Int\fractive Visualizations
- **Scatterplot Matrix:** Generate a `seaborn.pairplot` to visualize relationships between all selected independent variables and the dependent variable. This helps in identifying multicollinearity and initial linearity.
- **Residuals vs. Predicted Value Plot:** Create a scatter plot of residuals against the predicted values ($\hat{Y}$). Add a horizontal line at $Y=0$ for reference. This plot helps in assessing linearity and homoscedasticity.
- **Regression Residuals vs. Factors Plots:** Iterate through each selected independent variable and generate a scatter plot of residuals against that specific independent variable. Add a horizontal line at $Y=0$. This helps check the independence assumption and individual linearity.
- **Q-Q Plot for Normality:** Generate a Q-Q plot of the residuals to visually check for normality of the error terms.
- **Histogram of Residuals:** Display a histogram of the residuals to complement the Q-Q plot in assessing normality.

### Step 8: Display Explanatory Markdown and Formulae
- Utilize `st.markdown` to provide definitions, explanations, and interpretations of MLR assumptions and diagnostic plots.
- Use `st.latex` to present mathematical formulae clearly and correctly formatted according to the specified template.

### Step 9: Structuring the Streamlit Application
- Use `st.sidebar` for data selection and variable selection controls.
- Organize the main content area with `st.header`, `st.subheader`, and `st.expander` to present model summary, core concepts, and individual assumption checks (with plots and explanations).

## Core Concepts and Mathematical Foundations

This section details the key statistical concepts and their mathematical foundations, crucial for understanding multiple linear regression and its assumptions.

### Multiple Linear Regression Model
The multiple linear regression model describes the linear relationship between a dependent variable and two or more independent variables. It extends simple linear regression to account for multiple predictors. The relationship is calculated using:
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon
$$
Where:
- $Y$: The dependent (response) variable.
- $\beta_0$: The Y-intercept, representing the expected value of $Y$ when all $X_i$ are zero.
- $\beta_i$: The partial regression coefficient for the $i$-th independent variable, representing the expected change in $Y$ for a one-unit change in $X_i$, holding all other independent variables constant.
- $X_i$: The $i$-th independent (predictor) variable.
- $p$: The number of independent variables.
- $\epsilon$: The error term (or residual), representing the unexplained variation in $Y$.

This formula models how the dependent variable can be predicted or explained by a linear combination of multiple independent variables. It's widely used for prediction and to understand the relationships between variables in various fields, from economics to engineering.

### Residuals
Residuals are the differences between the observed values of the dependent variable and the values predicted by the regression model. They represent the errors or unexplained variation in the model. The residual for an observation $i$ is calculated using:
$$
e_i = Y_i - \hat{Y_i}
$$
Where:
- $e_i$: The residual for the $i$-th observation.
- $Y_i$: The observed value of the dependent variable for the $i$-th observation.
- $\hat{Y_i}$: The predicted value of the dependent variable for the $i$-th observation.

Residuals are critical for assessing the validity of a regression model's assumptions. By plotting and analyzing residuals, one can detect patterns that indicate violations of linearity, homoscedasticity, or independence.

### Mean Squared Error (MSE)
Mean Squared Error is a common metric used to quantify the average magnitude of the errors (residuals) of a regression model. It measures the average of the squares of the differences between observed and predicted values. A lower MSE indicates a better fit of the model to the data. It is calculated using:
$$
MSE = \\frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y_i})^2
$$
Where:
- $MSE$: Mean Squared Error.
- $n$: The number of observations in the dataset.
- $Y_i$: The observed value of the dependent variable for the $i$-th observation.
- $\hat{Y_i}$: The predicted value of the dependent variable for the $i$-th observation.

This formula provides a measure of the average squared distance between the actual data points and the regression line, giving a sense of the model's overall prediction acc\fracy.

### Coefficient of Determination ($R^2$)
The Coefficient of Determination, commonly denoted as $R^2$, is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s). It indicates how well the regression model fits the observed data. $R^2$ values range from 0 to 1, where higher values indicate a better fit. It is calculated using:
$$
R^2 = 1 - \\frac{SSE}{SST}
$$
Where:
- $R^2$: Coefficient of Determination.
- $SSE$: Sum of Squares of Residuals (or Error Sum of Squares), which is $\sum_{i=1}^{n} (Y_i - \hat{Y_i})^2$. It represents the variation unexplained by the model.
- $SST$: Total Sum of Squares, which is $\sum_{i=1}^{n} (Y_i - \bar{Y})^2$. It represents the total variation in the dependent variable.
- $Y_i$: The observed value of the dependent variable for the $i$-th observation.
- $\hat{Y_i}$: The predicted value of the dependent variable for the $i$-th observation.
- $\bar{Y}$: The mean of the observed values of the dependent variable.

This formula quantifies the goodness-of-fit of the model. For instance, an $R^2$ of 0.75 means that 75% of the variability in the dependent variable can be explained by the independent variables included in the model.

### Assumptions Underlying Multiple Linear Regression

The validity of multiple linear regression results relies on several key assumptions about the data and error terms. Violations of these assumptions can lead to unreliable coefficient estimates, incorrect standard errors, and misleading hypothesis tests.

#### 1. Linearity
The relationship between the independent variables and the dependent variable is linear.
$$
Y = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p + \epsilon
$$
- $Y$: Dependent variable
- $X_i$: Independent variables
- $\beta_i$: Coefficients
- $\epsilon$: Error term

This assumption means that the changes in the dependent variable are directly proportional to changes in the independent variables. If the relationship is non-linear (e.g., quadratic), the model will not accurately capture the true relationship, leading to biased coefficients and poor predictions.
**\fractical Application:** We assess linearity primarily by examining a plot of residuals versus predicted values and residuals versus each independent variable. A random scatter of points around zero indicates linearity. A discernible pattern (e.g., U-shape, curved pattern) suggests a violation.

#### 2. Independence of Errors (No Autocorrelation)
The error terms ($\epsilon$) are independent of each other. This means that the error for one observation does not influence the error for another observation.
$$
Cov(\epsilon_i, \epsilon_j) = 0 \quad \text{for } i \neq j
$$
- $\epsilon_i$: Error term for observation $i$
- $\epsilon_j$: Error term for observation $j$
- $Cov$: Covariance

This assumption is particularly important for time series data, where errors in one period might be correlated with errors in a subsequent period. If errors are correlated, it can lead to underestimated standard errors and inflated t-statistics, making variables appear more significant than they are.
**\fractical Application:** In cross-sectional data, this is often checked by plotting residuals against each independent variable. For time-series, a Durbin-Watson test or plots of residuals against time are used. A random scatter without any trend or pattern suggests independence.

#### 3. Homoscedasticity (Constant Variance of Errors)
The variance of the error terms ($\epsilon$) is constant across all levels of the independent variables.
$$
Var(\epsilon_i) = \sigma^2 \quad \text{for all } i
$$
- $\epsilon_i$: Error term for observation $i$
- $\sigma^2$: Constant variance
- $Var$: Variance

This assumption means that the spread of the residuals is consistent across the range of predicted values. If the variance of errors changes (heteroscedasticity), typically increasing with the predicted value, it leads to inefficient coefficient estimates and incorrect standard errors, making hypothesis tests unreliable.
**\fractical Application:** This is primarily assessed using a plot of residuals versus predicted values. A "fan" or "funnel" shape in the plot (where the spread of residuals changes) indicates heteroscedasticity, while a constant band of residuals suggests homoscedasticity.

#### 4. Normality of Errors
The error terms ($\epsilon$) are normally distributed.
$$
\epsilon \sim N(0, \sigma^2)
$$
- $\epsilon$: Error term
- $N$: Normal distribution
- $0$: Mean of the error terms (assumed to be zero)
- $\sigma^2$: Variance of the error terms

While the Central Limit Theorem helps with large sample sizes, ensuring normality of errors is important for accurate p-values and confidence intervals. Significant departures from normality can affect the reliability of hypothesis tests.
**\fractical Application:** This assumption is checked using a Q-Q plot of the residuals and a histogram of the residuals. A Q-Q plot where points lie close to the 45-degree line and a bell-shaped histogram centered around zero indicate normality.

#### 5. No Multicollinearity
The independent variables ($X_i$) are not highly correlated with each other. While some correlation is normal, high correlation (multicollinearity) can cause problems.
$$
Corr(X_i, X_j) \approx 0 \quad \text{for } i \neq j
$$
- $X_i$: Independent variable $i$
- $X_j$: Independent variable $j$
- $Corr$: Correlation

High multicollinearity makes it difficult to estimate the individual impact of each independent variable on the dependent variable. It can lead to large standard errors for coefficients, unstable coefficient estimates, and difficulty in interpreting the model.
**\fractical Application:** A scatterplot matrix of the independent variables helps to visually identify strong linear relationships. Variance Inflation Factor (VIF) is a quantitative measure, where high VIF values indicate significant multicollinearity.

## Required Libraries and Dependencies

The application relies on several standard Python libraries for data handling, statistical modeling, and visualization. These should be listed in `requirements.txt`.

-   **`streamlit==1.x.x`**: The core framework for building the int\fractive web application.
    -   **Specific functions/modules used**: `streamlit.header`, `streamlit.subheader`, `streamlit.write`, `streamlit.markdown`, `streamlit.pyplot`, `streamlit.dataframe`, `streamlit.selectbox`, `streamlit.multiselect`, `streamlit.info`, `streamlit.latex`, `streamlit.expander`.
    -   **Role**: Provides the user interface components and overall structure for the application.

-   **`pandas==1.x.x`**: Used for data manipulation and analysis, primarily for handling DataFrames.
    -   **Specific functions/modules used**: `pandas.DataFrame`, `pandas.read_csv` (if custom datasets are supported), DataFrame indexing and selection.
    -   **Role**: Manages the tabular data, allowing for easy selection of dependent and independent variables.

-   **`numpy==1.x.x`**: Provides support for large, multi-dimensional arrays and mathematical functions.
    -   **Specific functions/modules used**: `numpy.array`, `numpy.random` (if synthetic data generation is added, though not strictly in scope).
    -   **Role**: Underpins numerical operations, especially in conjunction with `pandas` and `scikit-learn`.

-   **`scikit-learn==1.x.x` (or `sklearn`)**: A comprehensive library for machine learning, including linear models.
    -   **Specific functions/modules used**:
        -   `sklearn.datasets.load_diabetes` (or `load_boston`, `load_iris`, etc.): For loading readily available datasets.
        -   `sklearn.linear_model.LinearRegression`: For training the multiple linear regression model.
        -   `sklearn.model_selection.train_test_split` (optional, but good \fractice for general ML workflows).
    -   **Role**: Provides the regression model implementation and access to sample datasets for demonstration.

-   **`matplotlib==3.x.x`**: A widely used plotting library for creating static, int\fractive, and animated visualizations in Python.
    -   **Specific functions/modules used**: `matplotlib.pyplot.figure`, `matplotlib.pyplot.scatter`, `matplotlib.pyplot.hist`, `matplotlib.pyplot.axhline`, `matplotlib.pyplot.title`, `matplotlib.pyplot.xlabel`, `matplotlib.pyplot.ylabel`, `matplotlib.pyplot.tight_layout`, `matplotlib.pyplot.show`.
    -   **Role**: Used for generating various plots such as residuals vs. predicted values, residuals vs. factors, and histograms of residuals.

-   **`seaborn==0.x.x`**: A statistical data visualization library based on Matplotlib. It provides a high-level interface for drawing at\fractive and informative statistical graphics.
    -   **Specific functions/modules used**: `seaborn.pairplot`, `seaborn.set_style`, `seaborn.histplot` (alternative to `plt.hist`).
    -   **Role**: Facilitates the creation of the scatterplot matrix for visualizing relationships between variables and multicollinearity.

-   **`statsmodels==0.x.x`**: A library for estimating and testing statistical models.
    -   **Specific functions/modules used**:
        -   `statsmodels.api as sm`: Provides access to statistical models and diagnostic functions.
        -   `statsmodels.graphics.gofplots.qqplot`: For generating Q-Q plots to check for normality of residuals.
        -   `statsmodels.tools.tools.add_constant` (for VIF, if used): Adds a constant (intercept) term to the independent variables matrix.
        -   `statsmodels.stats.outliers_influence.variance_inflation_factor` (for VIF, if used): Calculates the VIF for each independent variable.
    -   **Role**: Essential for generating the Q-Q plot and potentially for advanced regression diagnostics like VIF (if implemented).

**Example Import Statements (in `app.py`):**

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes # or other datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # potentially
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

## Implementation Details

### Data Handling and Preprocessing
The application will provide a selection of pre-loaded datasets from `sklearn.datasets`.
1.  **Dataset Selection**: Users will choose a dataset (e.g., 'Diabetes', 'Boston Housing (deprecated)', 'Iris') via a `st.selectbox` in the sidebar.
2.  **Feature and Target Separation**: Upon selection, the chosen dataset's features will be presented as potential independent variables and targets as potential dependent variables. For datasets like `load_diabetes`, the target is explicit. For others, an appropriate column will be designated as a default dependent variable, and others as independent.
3.  **Variable Selection**: `st.selectbox` will be used for the dependent variable and `st.multiselect` for independent variables. The application will dynamically update available options based on the selected dataset.
4.  **Data Cleaning (Implicit)**: For the chosen `sklearn` datasets, extensive data cleaning (like handling missing values) is generally not required as they are pre-cleaned. However, the internal structure should assume that $X$ and $Y$ are prepared for regression.

### Regression Model
1.  **Model Instantiation**: A `LinearRegression` model from `sklearn.linear_model` will be instantiated: `model = LinearRegression()`.
2.  **Model Training**: The model will be trained using the `fit` method: `model.fit(X, Y)`, where `X` is a pandas DataFrame of selected independent variables and `Y` is a pandas Series of the selected dependent variable.
3.  **Predictions**: After training, predicted values will be generated: `Y_pred = model.predict(X)`.
4.  **Residual Calculation**: Residuals will be calculated as the difference between actual and predicted values: `residuals = Y - Y_pred`.

### Calculation of Diagnostics
-   **Model Coefficients**: `model.coef_` and `model.intercept_` will be displayed to show the learned relationship.
-   **R-squared**: `model.score(X, Y)` will be used to get the $R^2$ value, indicating the model's explanatory power.
-   **Mean Squared Error**: `mean_squared_error(Y, Y_pred)` from `sklearn.metrics` could be used to quantify prediction error.
-   **Residuals**: The calculated `residuals` Series will be used for all diagnostic plots related to assumptions.

### Plotting Logic
All plots will be generated using `matplotlib.pyplot` and `seaborn` and displayed using `st.pyplot(fig)`. Each plot will be enclosed within a `matplotlib.figure.Figure` object for proper display in Streamlit.

1.  **Scatterplot Matrix (Multicollinearity Check)**:
    -   `sns.pairplot(data=df_selected_vars, diag_kind='kde')` will be used to visualize pairwise relationships between all selected variables (independent and dependent). The `diag_kind='kde'` will show kernel density estimates on the diagonal.
    -   **Interpretation**: Visual inspection for strong linear patterns between independent variables indicates multicollinearity.

2.  **Residuals vs. Predicted Value (Linearity & Homoscedasticity Check)**:
    -   A scatter plot (`plt.scatter(Y_pred, residuals)`) will be generated.
    -   A horizontal line at `y=0` (`plt.axhline(y=0, color='r', linestyle='--')`) will be added for reference.
    -   **Interpretation**:
        -   **Linearity**: No discernible pattern (e.g., U-shape, S-shape) in the scatter indicates linearity.
        -   **Homoscedasticity**: Random scatter with constant vertical spread across the range of predicted values indicates homoscedasticity. A "fan" or "funnel" shape suggests heteroscedasticity.

3.  **Regression Residuals vs. Factors (Independence & Individual Linearity Check)**:
    -   For each selected independent variable ($X_j$), a scatter plot of `plt.scatter(X[X_j], residuals)` will be generated.
    -   A horizontal line at `y=0` (`plt.axhline(y=0, color='r', linestyle='--')`) will be added.
    -   **Interpretation**: Similar to the residuals vs. predicted plot, a random scatter implies independence and linearity with that specific factor. Patterns indicate violations.

4.  **Q-Q Plot of Residuals (Normality Check)**:
    -   `fig = sm.qqplot(residuals, line='s', fit=True)` will be used. The `line='s'` option adds a standardized line.
    -   `plt.title('Normal Q-Q Plot of Residuals')` will provide context.
    -   **Interpretation**: If the points closely follow the straight line, the residuals are approximately normally distributed. Deviations, especially at the tails, suggest non-normality.

5.  **Histogram of Residuals (Normality Check)**:
    -   `plt.hist(residuals, bins=30, edgecolor='black', density=True)` will generate a histogram.
    -   Optionally, a normal distribution curve could be overlaid for comparison.
    -   **Interpretation**: A bell-shaped distribution centered around zero supports the normality assumption. Skewness or multiple peaks indicate non-normality.

## User Interface Components

The Streamlit application's user interface will be structured for intuitive navigation and int\fraction.

### Sidebar (`st.sidebar`)
-   **Application Title**: "Multiple Linear Regression Assumptions Analyzer" at the top.
-   **Dataset Selection**:
    -   `st.subheader("Data Selection")`
    -   `st.selectbox("Choose Dataset", options=["Diabetes", "Boston (deprecated)", "Iris"])`: A dropdown to select from pre-defined datasets.
-   **Variable Selection**:
    -   `st.subheader("Variable Configuration")`
    -   `st.selectbox("Select Dependent Variable (Y)", options=df.columns)`: A dropdown to choose the target variable from the selected dataset's columns.
    -   `st.multiselect("Select Independent Variables (X)", options=available_features)`: A multi-select box to choose predictor variables. This will be dynamically populated based on the chosen dataset and excluding the selected dependent variable.
    -   `st.button("Run Analysis")`: A button to trigger the regression analysis and plot generation.

### Main Content Area
The main area of the application will display the overview, model summary, and detailed assumption analyses.

-   **Overview and Learning Outcomes**:
    -   `st.header("Multiple Linear Regression Assumptions Analyzer")`
    -   `st.markdown("This application helps visualize the assumptions...")` (Detailed explanation from "Overview" section).
    -   `st.subheader("Learning Outcomes")`
    -   `st.markdown("- Assumptions underlying multiple linear regression")`
    -   `st.markdown("- Explain the assumptions underlying a multiple linear regression model and interpret residual plots indicating potential violations of these assumptions")`

-   **Core Concepts and Mathematical Foundations (Foldable Section)**:
    -   `st.expander("Core Concepts and Mathematical Foundations")`:
        -   Each mathematical concept (Multiple Linear Regression Model, Residuals, Mean Squared Error, Coefficient of Determination) will be presented using `st.subheader`, `st.markdown` for descriptions, and `st.latex` for formulae, strictly following the "Formula Presentation Template".

-   **Model Summary**:
    -   `st.subheader("Regression Model Summary")`
    -   Display `st.write(f"R-squared: {r_squared:.4f}")` and `st.write(f"Mean Squared Error: {mse:.4f}")`.
    -   `st.dataframe(pd.DataFrame({'Coefficient': model.coef_}, index=X.columns))` (and intercept).
    -   `st.info("The R-squared value indicates the proportion of variance in the dependent variable explained by the model.")`

-   **Assumption Analysis Sections (Foldable Sections for clarity)**:

    #### 1. Linearity and Homoscedasticity
    -   `st.expander("Assumption 1: Linearity & Homoscedasticity")`:
        -   `st.subheader("Residuals vs. Predicted Values Plot")`
        -   `st.pyplot(fig_res_pred)`: Displays the plot generated in `Implementation Details`.
        -   `st.markdown("#### Interpretation:")`
        -   `st.markdown("A random scatter of points around the horizontal line at zero indicates that the assumptions of **linearity** and **homoscedasticity** hold.")`
        -   `st.markdown("Patterns (e.g., U-shape) suggest non-linearity. A changing spread (e.g., funnel shape) indicates heteroscedasticity.")`

    #### 2. Independence of Errors
    -   `st.expander("Assumption 2: Independence of Errors (No Autocorrelation)")`:
        -   `st.subheader("Residuals vs. Individual Factors Plots")`
        -   Loop through each independent variable:
            -   `st.pyplot(fig_res_factor_Xj)`: Displays a plot of residuals against each independent variable.
        -   `st.markdown("#### Interpretation:")`
        -   `st.markdown("Each plot should show a random scatter of residuals around zero, indicating that the error terms are independent of the independent variables and each other.")`
        -   `st.markdown("Any systematic pattern suggests a violation of independence or a missed non-linear relationship.")`

    #### 3. Normality of Errors
    -   `st.expander("Assumption 3: Normality of Errors")`:
        -   `st.subheader("Normal Q-Q Plot of Residuals")`
        -   `st.pyplot(fig_qq_plot)`: Displays the Q-Q plot.
        -   `st.subheader("Histogram of Residuals")`
        -   `st.pyplot(fig_hist_residuals)`: Displays the histogram.
        -   `st.markdown("#### Interpretation:")`
        -   `st.markdown("For the **Normal Q-Q Plot**, if the residuals are normally distributed, the points should fall approximately along the 45-degree reference line.")`
        -   `st.markdown("For the **Histogram**, a bell-shaped distribution symmetric around zero suggests normality.")`

    #### 4. No Multicollinearity
    -   `st.expander("Assumption 4: No Multicollinearity")`:
        -   `st.subheader("Scatterplot Matrix of Variables")`
        -   `st.pyplot(fig_pairplot)`: Displays the scatterplot matrix.
        -   `st.markdown("#### Interpretation:")`
        -   `st.markdown("Visually inspect the scatterplot matrix for strong linear relationships between the independent variables. High correlation (multicollinearity) can inflate standard errors of regression coefficients.")`
        -   `st.markdown("A perfect correlation (points forming a straight line) between two independent variables is a strong indicator of multicollinearity.")`

This structured approach ensures that the application is functional, educational, and easy to navigate for users exploring multiple linear regression assumptions.


### Appendix Code

```code
Exhibit 1: Anheuser-Busch InBev SA/NV Consolidated Income Statement (in Millions of US Dollars) [Excerpt]
12 Months Ended December 31
|             | 2017       | 2016       | 2015       |
| :---------- | :--------- | :--------- | :--------- |
| Revenue     | $56,444    | $45,517    | $43,604    |
| Cost of sales | (21,386)   | (17,803)   | (17,137)   |
| Gross profit | 35,058     | 27,715     | 26,467     |
| Distribution expenses | (5,876) | (4,543) | (4,259) |

Exhibit 2: Molson Coors Brewing Company Consolidated Statement of Operations (in Millions of US Dollars) [Excerpt]
12 Months Ended
|             | Dec. 31, 2017 | Dec. 31, 2016 | Dec. 31, 2015 |
| :---------- | :------------ | :------------ | :------------ |
| Sales       | $13,471.5     | $6,597.4      | $5,127.4      |
| Excise taxes | (2,468.7)     | (1,712.4)     | (1,559.9)     |
| Net sales   | 11,002.8      | 4,885.0       | 3,567.5       |
| Cost of goods sold | (6,217.2)     | (2,987.5)     | (2,131.6)     |
| Gross profit | 4,785.6       | 1,897.5       | 1,435.9       |
| Marketing, general and administrative expenses | (3,032.4) | (1,589.8) | (1,038.3) |
| Special items, net | (28.1) | 2,522.4 | (346.7) |
| Equity Income in MillerCoors | 0 | 500.9 | 516.3 |
| Operating income (loss) | 1,725.1 | 3,331.0 | 567.2 |
| Other income (expense), net |             |               |               |

Exhibit 3: Groupe Danone Consolidated Income Statement (in Millions of Euros) [Excerpt]
Year Ended 31 December
|                          | 2016       | 2017       |
| :----------------------- | :--------- | :--------- |
| Sales                    | 21,944     | 24,677     |
| Cost of goods sold       | (10,744)   | (12,459)   |
| Selling expense          | (5,562)    | (5,890)    |
| General and administrative expense | (2,004)    | (2,225)    |
| Research and development expense | (333)      | (342)      |
| Other income (expense)   | (278)      | (219)      |
| Recurring operating income | 3,022      | 3,543      |
| Other operating income (expense) | (99)       | 192        |
| Operating income         | 2,923      | 3,734      |
| Interest income on cash equivalents and short-term investments | 130        | 151        |
| Interest expense         | (276)      | (414)      |
| Cost of net debt         | (146)      | (263)      |
| Other financial income   | 67         | 137        |
| Other financial expense  | (214)      | (312)      |
| Income before tax        | 2,630      | 3,296      |
| Income tax expense       | (804)      | (842)      |
| Net income from fully consolidated companies | 1,826      | 2,454      |
| Share of profit of associates | 1          | 109        |
| Net income               | 1,827      | 2,563      |
| Net income – Group share | 1,720      | 2,453      |
| Net income - Non-controlling interests | 107        | 110        |

Exhibit 4: CRA International Inc. Consolidated Statements of Operations (Excerpt) (in Thousands of Dollars)
Fiscal Year Ended
|                                            | Dec. 30, 2017 | Dec. 31, 2016 | Jan. 02, 2016 |
| :----------------------------------------- | :------------ | :------------ | :------------ |
| Revenues                                   | $370,075      | $324,779      | $303,559      |
| Costs of services (exclusive of depreciation and amortization) | 258,829       | 227,380       | 207,650       |
| Selling, general and administrative expenses | 86,537        | 70,584        | 72,439        |
| Depreciation and amortization              | 8,945         | 7,896         | 6,552         |
| GNU goodwill impairment                    | —             | —             | 4,524         |
| Income from operations                     | 15,764        | 18,919        | 12,394        |

Exhibit 5: Applying the Converged Revenue Recognition Standard
Part 1 (ref. Example 10)
Builder Co. enters into a con\fract with Customer Co. to construct a commercial building. Builder Co. identifies various goods and services to be provided, such as pre-construction engineering, construction of the building’s individual components, plumbing, electrical wiring, and interior finishes. With respect to "Identifying the Performance Obligation,” should Builder Co. treat each specific item as a separate performance obligation to which revenue should be allocated?
The standard provides two criteria, which must be met, to determine if a good or service is distinct for purposes of identifying performance obligations. First, the customer can benefit from the good or service either on its own or together with other readily available resources. Second, the seller’s "promise to transfer the good or service to the customer is separately identifiable from other promises in the con\fract.” In this example, the second criterion is not met because it is the building for which the customer has con\fracted, not the separate goods and services. The seller will integrate all the goods and services into a combined output and each specific item should not be treated as a distinct good or service but accounted for together as a single performance obligation.

Part 2 (ref. Example 8)
Builder Co’s con\fract with Customer Co. to construct the commercial building specifies consideration of $1 million. Builder Co’s expected total costs are $700,000. The Builder incurs $420,000 in costs in the first year. Assuming that costs incurred provide an appropriate measure of progress toward completing the con\fract, how much revenue should Builder Co. recognize for the first year?
The standard states that for performance obligations satisfied over time (e.g., where there is a long-term con\fract), revenue is recognized over time by measuring progress toward satisfying the obligation. In this case, the Builder has incurred 60% of the total expected costs ($420,000/$700,000) and will thus recognize $600,000 (60% × $1 million) in revenue for the first year.
This is the same amount of revenue that would be recognized using the "percentage-of-completion” method under previous accounting standards, but that term is not used in the converged standard. Instead, the standard refers to performance obligations satisfied over time and requires that progress toward complete satisfaction of the performance obligation be measured based on input method such as the one illustrated here (recognizing revenue based on the proportion of total costs that have been incurred in the period) or an output method (recognizing revenue based on units produced or milestones achieved).

Part 3 (ref. Example 8)
Assume that Builder Co’s con\fract with Customer Co. to construct the commercial building specifies consideration of $1 million plus a bonus of $200,000 if the building is completed within 2 years. Builder Co. has only limited experience with similar types of con\fracts and knows that many factors outside its control (e.g., weather, regulatory requirements) could cause delay. Builder Co’s expected total costs are $700,000. The Builder incurs $420,000 in costs in the first year. Assuming that costs incurred provide an appropriate measure of progress toward completing the con\fract, how much revenue should Builder Co. recognize for the first year?
The standard addresses so-called “variable consideration" as part of determining the transaction price. A company is only allowed to recognize variable consideration if it can conclude that it will not have to reverse the cumulative revenue in the future. In this case, Builder Co. does not recognize any of the bonus in year one because it cannot reach the non-reversible conclusion given its limited experience with similar con\fracts and potential delays from factors outside its control.

Part 4 (ref. Example 8)
Assume all facts from Part 3. In the beginning of year two, Builder Co. and Customer Co. agree to change the building floor plan and modify the con\fract. As a result the consideration will increase by $150,000, and the allowable time for achieving the bonus is extended by 6 months. Builder expects its costs will increase by $120,000. Also, given the additional 6 months to earn the completion bonus, Builder concludes that it now meets the criteria for including the $200,000 bonus in revenue. How should Builder account for this change in the con\fract?
Note that previous standards did not provide a general framework for con\fract modifications. The converged standard provides guidance on whether a change in a con\fract is a new con\fract or a modification of an existing con\fract. To be considered a new con\fract, the change would need to involve goods and services that are distinct from the goods and services already transferred.
In this case, the change does not meet the criteria of a new con\fract and is therefore considered a modification of the existing con\fract, which requires the company to reflect the impact on a cumulative catch-up basis. Therefore, the company must update its transaction price and measure of progress. Builder’s total revenue on the transaction (transaction price) is now $1.35 million ($1 million original plus the $150,000 new consideration plus $200,000 for the completion bonus). Builder Co’s progress toward completion is now 51.2% ($420,000 costs incurred divided by total expected costs of $820,000). Based on the changes in the con\fract, the amount of additional revenue to be recognized is $91,200, calculated as (51.2% × $1.35 million) minus the $600,000 already recognized. The additional $91,200 of revenue would be recognized as a “cumulative catch-up adjustment" on the date of the con\fract modification.

Part 5 (ref. Example 45)
Assume a Company operates a website that enables customers to purchase goods from various suppliers. The customers pay the Company in advance, and orders are nonrefundable. The suppliers deliver the goods directly to the customer, and the Company receives a 10% commission. Should the Company report Total Revenues equal to 100% of the sales amount (gross) or Total Revenues equal to 10% of the sales amount (net)? Revenues are reported gross if the Company is acting as a Principal and net if the Company is acting as an Agent.
In this example, the Company is an Agent because it isn’t primarily responsible for fulfilling the con\fract, doesn’t take any inventory risk or credit risk, doesn’t have discretion in setting the price, and receives compensation in the form of a commission. Because the Company is acting as an Agent, it should report only the amount of commission as its revenue.

EXAMPLE 1
The Matching of Inventory Costs with Revenues
1. Kahn Distribution Limited (KDL), a hypothetical company, purchases
inventory items for resale. At the beginning of 2018, Kahn had no inventory
on hand. During 2018, KDL had the following transactions:
Inventory Purchases
|                |                   |                   |
| :------------- | :---------------- | :---------------- |
| First quarter  | 2,000             | units at $40 per unit |
| Second quarter | 1,500             | units at $41 per unit |
| Third quarter  | 2,200             | units at $43 per unit |
| Fourth quarter | 1,900             | units at $45 per unit |
| Total          | 7,600             | units at a total cost of $321,600 |

KDL sold 5,600 units of inventory during the year at $50 per unit, and
received cash. KDL determines that there were 2,000 remaining units of
inventory and specifically identifies that 1,900 were those purchased in
the fourth quarter and 100 were purchased in the third quarter. What are
the revenue and expense associated with these transactions during 2018
based on specific identification of inventory items as sold or remaining in
inventory? (Assume that the company does not expect any products to be
returned.)
Solution:
The revenue for 2018 would be $280,000 (5,600 units × $50 per unit). Initially, the total cost of the goods purchased would be recorded as inventory (an asset) in the amount of $321,600. During 2018, the cost of the 5,600 units sold would be expensed (matched against the revenue) while the cost of the 2,000 remaining unsold units would remain in inventory as follows:
Cost of Goods Sold
|                       |                                 |           |
| :-------------------- | :------------------------------ | :-------- |
| From the first quarter | 2,000 units at $40 per unit =   | $80,000   |
| From the second quarter | 1,500 units at $41 per unit =   | $61,500   |
| From the third quarter | 2,100 units at $43 per unit =   | $90,300   |
| Total cost of goods sold |                                 | $231,800  |
Cost of Goods Remaining in Inventory
|                        |                                 |           |
| :--------------------- | :------------------------------ | :-------- |
| From the third quarter | 100 units at $43 per unit =     | $4,300    |
| From the fourth quarter | 1,900 units at $45 per unit =   | $85,500   |
| Total remaining (or ending) inventory cost |                 | $89,800   |

To confirm that total costs are accounted for: $231,800 + $89,800 =
$321,600. The cost of the goods sold would be expensed against the revenue
of $280,000 as follows:
|                   |           |
| :---------------- | :-------- |
| Revenue           | $280,000  |
| Cost of goods sold | 231,800   |
| Gross profit      | $48,200   |

EXAMPLE 2
Alternative Inventory Costing Methods
In Example 1, KDL was able to specifically identify which inventory items were
sold and which remained in inventory to be carried over to later periods. This is
called the specific identification method and inventory and cost of goods sold
are based on their physical flow. It is generally not feasible to specifically iden-
tify which items were sold and which remain on hand, so accounting standards
permit the assignment of inventory costs to costs of goods sold and to ending
inventory using cost formulas (IFRS terminology) or cost flow assumptions (US
GAAP). The cost formula or cost flow assumption determines which goods are
assumed to be sold and which goods are assumed to remain in inventory. Both
IFRS and US GAAP permit the use of the first in, first out (FIFO) method, and
the weighted average cost method to assign costs.
Under the FIFO method, the oldest goods purchased (or manufactured) are
assumed to be sold first and the newest goods purchased (or manufactured)
are assumed to remain in inventory. Cost of goods in beginning inventory and
costs of the first items purchased (or manufactured) flow into cost of goods
sold first, as if the earliest items purchased sold first. Ending inventory would,
therefore, include the most recent purchases. It turns out that those items
specifically identified as sold in Example 1 were also the first items purchased,
so in this example, under FIFO, the cost of goods sold would also be $231,800,
calculated as above.
The weighted average cost method assigns the average cost of goods available
for sale to the units sold and remaining in inventory. The assignment is based on
the average cost per unit (total cost of goods available for sale/total units available
for sale) and the number of units sold and the number remaining in inventory.
For KDL, the weighted average cost per unit would be
$321,600/7,600 units = $42.3158 per unit
Cost of goods sold using the weighted average cost method would be
5,600 units at $42.3158 = $236,968

Ending inventory using the weighted average cost method would be
2,000 units at $42.3158 = $84,632

Another method is permitted under US GAAP but is not permitted under IFRS.
This is the last in, first out (LIFO) method. Under the LIFO method, the newest
goods purchased (or manufactured) are assumed to be sold first and the oldest
goods purchased (or manufactured) are assumed to remain in inventory. Costs
of the latest items purchased flow into cost of goods sold first, as if the most
recent items purchased were sold first. Although this may seem contrary to
common sense, it is logical in certain circumstances. For example, lumber in a
lumberyard may be stacked up with the oldest lumber on the bottom. As lumber
is sold, it is sold from the top of the stack, so the last lumber purchased and put
in inventory is the first lumber out. Theoretically, a company should choose a
method linked to the physical inventory flows. Under the LIFO method, in
the KDL example, it would be assumed that the 2,000 units remaining in ending
inventory would have come from the first quarter’s purchases:
Ending inventory 2,000 units at $40 per unit = $80,000
The remaining costs would be allocated to cost of goods sold under LIFO:
Total costs of $321,600 less $80,000 remaining in ending inventory = $241,600
Alternatively, the cost of the last 5,600 units purchased is allocated to cost of
goods sold under LIFO:
1,900 units at $45 per unit + 2,200 units at $43 per unit + 1,500 units at $41 per unit
= $241,600

Exhibit 6: Summary Table on Inventory Costing Methods
| Method                  | Description                                                                                                                                                                                                         | Cost of Goods Sold When Prices Are Rising, Relative to Other Two Methods | Ending Inventory When Prices Are Rising, Relative to Other Two Methods |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------- | :--------------------------------------------------------------------- |
| FIFO (first in, first out) | Costs of the earliest items purchased flow to cost of goods sold first                                                                                                                                                | Lowest                                                                   | Highest                                                                |
| LIFO (last in, first out) | Costs of the most recent items purchased flow to cost of goods sold first                                                                                                                                             | Highest*                                                                 | Lowest*                                                                |
| Weighted average cost   | Averages total costs over total units available                                                                                                                                                                     | Middle                                                                   | Middle                                                                 |

EXAMPLE 3
Sensitivity of Annual Depreciation Expense to Varying Estimates of Useful Life and Residual Value
Using the straight-line method of depreciation, annual depreciation expense is
calculated as:
Cost - Residual value
Estimated useful life
Assume the cost of an asset is $10,000. If, for example, the residual value of
the asset is estimated to be $0 and its useful life is estimated to be 5 years, the
annual depreciation expense under the straight-line method would be ($10,000
– $0)/5 years = $2,000. In contrast, holding the estimated useful life of the asset
constant at 5 years but increasing the estimated residual value of the asset to
$4,000 would result in annual depreciation expense of only $1,200 [calculated as
($10,000 – $4,000)/5 years]. Alternatively, holding the estimated residual value
at $0 but increasing the estimated useful life of the asset to 10 years would result
in annual depreciation expense of only $1,000 [calculated as ($10,000 – $0)/10
years]. Exhibit 7 shows annual depreciation expense for various combinations
of estimated useful life and residual value.

Exhibit 7: Annual Depreciation Expense (in Dollars)
Estimated Residual Value
| Estimated Useful Life (Years) | 0     | 1,000 | 2,000 | 3,000 | 4,000 | 5,000 |
| :-------------------------- | :---- | :---- | :---- | :---- | :---- | :---- |
| 2                           | 5,000 | 4,500 | 4,000 | 3,500 | 3,000 | 2,500 |
| 4                           | 2,500 | 2,250 | 2,000 | 1,750 | 1,500 | 1,250 |
| 5                           | 2,000 | 1,800 | 1,600 | 1,400 | 1,200 | 1,000 |
| 8                           | 1,250 | 1,125 | 1,000 | 875   | 750   | 625   |
| 10                          | 1,000 | 900   | 800   | 700   | 600   | 500   |

EXAMPLE 4
An Illustration of Diminishing Balance Depreciation
Assume the cost of computer equipment was $11,000, the estimated residual
value is $1,000, and the estimated useful life is five years. Under the diminishing
or declining balance method, the first step is to determine the straight-line rate,
the rate at which the asset would be depreciated under the straight-line method.
This rate is measured as 100 percent divided by the useful life or 20 percent for
a five-year useful life. Under the straight-line method, 1/5 or 20 percent of the
depreciable cost of the asset (here, $11,000 – $1,000 = $10,000) would be expensed
each year for five years: The depreciation expense would be $2,000 per year.
The next step is to determine an acceleration factor that approximates the
pattern of the asset’s wear. Common acceleration factors are 150 percent and
200 percent. The latter is known as double declining balance depreciation
because it depreciates the asset at double the straight-line rate. Using the 200
percent acceleration factor, the diminishing balance rate would be 40 percent (20
percent × 2.0). This rate is then applied to the remaining undepreciated balance
of the asset each period (known as the net book value).
At the beginning of the first year, the net book value is $11,000. Depreciation
expense for the first full year of use of the asset would be 40 percent of $11,000,
or $4,400. Under this method, the residual value, if any, is generally not used in
the computation of the depreciation each period (the 40 percent is applied to
$11,000 rather than to $11,000 minus residual value). However, the company
will stop taking depreciation when the salvage value is reached.
At the beginning of Year 2, the net book value is measured as
|                             |           |
| :-------------------------- | :-------- |
| Asset cost                  | $11,000   |
| Less: Accumulated depreciation | (4,400)   |
| Net book value              | $6,600    |

For the second full year, depreciation expense would be $6,600 × 40 percent, or
$2,640. At the end of the second year (i.e., beginning of the third year), a total
of $7,040 ($4,400 + $2,640) of depreciation would have been recorded. So, the
remaining net book value at the beginning of the third year would be
|                             |           |
| :-------------------------- | :-------- |
| Asset cost                  | $11,000   |
| Less: Accumulated depreciation | (7,040)   |
| Net book value              | $3,960    |

For the third full year, depreciation would be $3,960 × 40 percent, or $1,584.
At the end of the third year, a total of $8,624 ($4,400 + $2,640 + $1,584) of
depreciation would have been recorded. So, the remaining net book value at
the beginning of the fourth year would be
|                             |           |
| :-------------------------- | :-------- |
| Asset cost                  | $11,000   |
| Less: Accumulated depreciation | (8,624)   |
| Net book value              | $2,376    |

For the fourth full year, depreciation would be $2,376 × 40 percent, or $950. At
the end of the fourth year, a total of $9,574 ($4,400 + $2,640 + $1,584 + $950)
of depreciation would have been recorded. So, the remaining net book value at
the beginning of the fifth year would be
|                             |           |
| :-------------------------- | :-------- |
| Asset cost                  | $11,000   |
| Less: Accumulated depreciation | (9,574)   |
| Net book value              | $1,426    |

For the fifth year, if deprecation were determined as in previous years, it would
amount to $570 ($1,426 × 40 percent). However, this would result in a remain-
ing net book value of the asset below its estimated residual value of $1,000. So,
instead, only $426 would be depreciated, leaving a $1,000 net book value at the
end of the fifth year.
|                             |           |
| :-------------------------- | :-------- |
| Asset cost                  | $11,000   |
| Less: Accumulated depreciation | (10,000)  |
| Net book value              | $1,000    |

Exhibit 8: Highlighting Infrequent Nature of Items—Excerpt from Groupe Danone footnotes to its 2017 financial statements
NOTE 6. Events and Transactions Outside the Group's Ordinary Activities
[Excerpt]
"Other operating income (expense) is defined under Recommendation
2013-03 of the French CNC relating to the format of consolidated financial
statements prepared under international accounting standards, and com-
prises significant items that, because of their exceptional nature, cannot
be viewed as inherent to Danone’s current activities. These mainly include
capital gains and losses on disposals of fully consolidated companies, impair-
ment charges on goodwill, significant costs related to strategic restructuring
and major external growth transactions, and incurred or estimated costs
related to major crises and major litigation. Furthermore, in connection
with Revised IFRS 3 and Revised IAS 27, Danone also classifies in Other
operating income (expense) (i) acquisition costs related to business com-
binations, (ii) revaluation profit or loss accounted for following a loss of
control, and (iii) changes in earn-outs related to business combinations
and subsequent to the acquisition date.
"In 2017, the net Other operating income of €192 million consisted
mainly of the following items:
| (in € millions)                                    | Related income (expense) |
| :----------------------------------------------- | :----------------------- |
| Capital gain on disposal of Stonyfield           | 628                      |
| Compensation received following the decision of the Singapore arbi-tration court in the Fonterra case | 105                      |
| Territorial risks, mainly in certain countries in the ALMA region | (148)                    |
| Costs associated with the integration of WhiteWave | (118)                    |
| Impairment of several intangible assets in Waters and Specialized Nutrition Reporting entities | (115)                    |
| Remainder of table omitted                       |                          |

EXAMPLE 5
Microsoft Corporation Excerpt from Footnotes to the Financial Statements
The most significant impact of the [new revenue recognition] standard
relates to our accounting for software license revenue. Specifically, for
Windows 10, we recognize revenue predominantly at the time of billing
and delivery rather than ratably over the life of the related device. For
certain multi-year commercial software subscriptions that include both
distinct software licenses and SA, we recognize license revenue at the time
of con\fract execution rather than over the subscription period. Due to the
complexity of certain of our commercial license subscription con\fracts,
the actual revenue recognition treatment required under the standard
depends on con\fract-specific terms and in some instances may vary from
recognition at the time of billing. Revenue recognition related to our
hardware, cloud offerings (such as Office 365), LinkedIn, and professional
services remains substantially unchanged. Refer to Impacts to Previously
Reported Results below for the impact of adoption of the standard in our
consolidated financial statements.
(In $ millions, except per share
amounts)
|                         | As Previously Reported | New Revenue Standard Adjustment | As Restated |
| :---------------------- | :--------------------- | :------------------------------ | :---------- |
| **Income Statements**   |                        |                                 |             |
| **Year Ended June 30, 2017** |                        |                                 |             |
| Revenue                 | 89,950                 | 6,621                           | 96,571      |
| Provision for income taxes | 1,945                  | 2,467                           | 4,412       |
| Net income              | 21,204                 | 4,285                           | 25,489      |
| Diluted earnings per share | 2.71                   | 0.54                            | 3.25        |
| **Year Ended June 30, 2016** |                        |                                 |             |
| Revenue                 | 85,320                 | 5,834                           | 91,154      |
| Provision for income taxes | 2,953                  | 2,147                           | 5,100       |
| Net income              | 16,798                 | 3,741                           | 20,539      |
| Diluted earnings per share | 2.1                    | 0.46                            | 2.56        |

1. Question: Based on the above information, describe whether Microsoft's
results appear better or worse under the new revenue recognition standard.
Solution:
Microsoft's results appear better under the new revenue recognition stan-
dard. Revenues and income are higher under the new standard. The net
profit margin is higher under the new standard. For 2017, the net profit
margin is 26.4% (= 25,489/96,571) under the new standard versus 23.6% (=
21,204/89,950) under the old standard. Reported revenue grew faster under
the new standard. Revenue growth under the new standard was 5.9% [=
(96,571/91,154) – 1] compared to 5.4% [= (89,950/85,320) – 1)] under the
old standard.
Microsoft's presentation of the effects of the new revenue recognition
enables an analyst to identify the impact of the change in accounting stan-
dards.

Exhibit 9: Change in Accounting Estimate
Catalent Inc. discloses a change in the method it uses to calculate both annual
expenses related to its defined benefit pension plans. Rather than use a single,
weighted-average discount rate in its calculations, the company will use the spot
rates applicable to each projected cash flow.
Post-Retirement and Pension Plans
...The measurement of the related benefit obligations and the net periodic
benefit costs recorded each year are based upon actuarial computations,
which require management's judgment as to certain assumptions. These
assumptions include the discount rates used in computing the present value
of the benefit obligations and the net periodic benefit costs...
Effective June 30, 2016, the approach used to estimate the service and
interest components of net periodic benefit cost for benefit plans was
changed to provide a more precise measurement of service and interest
costs. Historically, the Company estimated these service and interest com-
ponents utilizing a single weighted-average discount rate derived from the
yield curve used to measure the benefit obligation at the beginning of the
period. Going forward, the Company has elected to utilize an approach
that discounts the individual expected cash flows using the applicable spot
rates derived from the yield curve over the projected cash flow period. The
Company has accounted for this change as a change in accounting estimate
that is inseparable from a change in accounting principle and accordingly
has accounted for it prospectively.

Exhibit 10: AB InBev's Earnings Per Share
12 Months Ended December 31
|                                   | 2017    | 2016    | 2015    |
| :-------------------------------- | :------ | :------ | :------ |
| Basic earnings per share          | $4.06   | $0.72   | $5.05   |
| Diluted earnings per share        | 3.98    | 0.71    | 4.96    |
| Basic earnings per share from continuing operations | 4.04    | 0.69    | 5.05    |
| Diluted earnings per share from continuing operations | $3.96   | $0.68   | $4.96   |

Basic EPS = Net income - Preferred dividends / Weighted average number of shares outstanding

EXAMPLE 6
A Basic EPS Calculation (1)
1. For the year ended 31 December 2018, Shopalot Company had net income
of $1,950,000. The company had 1,500,000 shares of common stock out-
standing, no preferred stock, and no convertible financial instruments.
What is Shopalot's basic EPS?
Solution:
Shopalot's basic EPS is $1.30 ($1,950,000 divided by 1,500,000 shares).

EXAMPLE 7
A Basic EPS Calculation (2)
For the year ended 31 December 2018, Angler Products had net income of
$2,500,000. The company declared and paid $200,000 of dividends on preferred
stock. The company also had the following common stock share information:
|                                       |           |
| :------------------------------------ | :-------- |
| Shares outstanding on 1 January 2018  | 1,000,000 |
| Shares issued on 1 April 2018         | 200,000   |
| Shares repurchased (treasury shares) on 1 October 2018 | (100,000) |
| Shares outstanding on 31 December 2018 | 1,100,000 |

1. What is the company's weighted average number of shares outstanding?
Solution to 1:
The weighted average number of shares outstanding is determined by the
length of time each quantity of shares was outstanding:
|                                       |           |
| :------------------------------------ | :-------- |
| 1,000,000 x (3 months/12 months) =    | 250,000   |
| 1,200,000 × (6 months/12 months) =    | 600,000   |
| 1,100,000 x (3 months/12 months) =    | 275,000   |
| Weighted average number of shares outstanding | 1,125,000 |

2. What is the company's basic EPS?
Solution to 2:
Basic EPS = (Net income – Preferred dividends)/Weighted average number
of shares = ($2,500,000 – $200,000)/1,125,000 = $2.04

EXAMPLE 8
A Basic EPS Calculation (3)
1. Assume the same facts as Example 7 except that on 1 December 2018, a pre-
viously declared 2-for-1 stock split took effect. Each shareholder of record
receives two shares in exchange for each current share that he or she owns.
What is the company's basic EPS?
Solution:
For EPS calculation purposes, a stock split is treated as if it occurred at the
beginning of the period. The weighted average number of shares would,
therefore, be 2,250,000, and the basic EPS would be $1.02 [= ($2,500,000 –
$200,000)/2,250,000].

Diluted EPS = (Net income) / (Weighted average number of shares outstanding + New common shares that would have been issued at conversion)

EXAMPLE 9
A Diluted EPS Calculation Using the If-Converted Method for Preferred Stock
1. For the year ended 31 December 2018, Bright-Warm Utility Company
(fictitious) had net income of $1,750,000. The company had an average of
500,000 shares of common stock outstanding, 20,000 shares of convertible
preferred, and no other potentially dilutive securities. Each share of pre-
ferred pays a dividend of $10 per share, and each is convertible into five
shares of the company’s common stock. Calculate the company’s basic and
diluted EPS.
Solution:
If the 20,000 shares of convertible preferred had each converted into 5
shares of the company’s common stock, the company would have had an
additional 100,000 shares of common stock (5 shares of common for each of
the 20,000 shares of preferred). If the conversion had taken place, the com-
pany would not have paid preferred dividends of $200,000 ($10 per share for
each of the 20,000 shares of preferred). As shown in Exhibit 11, the compa-
ny’s basic EPS was $3.10 and its diluted EPS was $2.92.

Exhibit 11: Calculation of Diluted EPS for Bright-Warm Utility Company Using the If-Converted Method: Case of Preferred Stock
|                                  | Basic EPS   | Diluted EPS Using If-Converted Method |
| :------------------------------- | :---------- | :------------------------------------ |
| Net income                       | $1,750,000  | $1,750,000                          |
| Preferred dividend               | -200,000    | 0                                     |
| **Numerator**                    | **$1,550,000** | **$1,750,000**                      |
| Weighted average number of shares outstanding | 500,000     | 500,000                             |
| Additional shares issued if preferred converted | 0           | 100,000                             |
| Denominator                      | 500,000     | 600,000                             |
| EPS                              | $3.10       | $2.92                                 |

Diluted EPS = (Net income + After-tax interest on convertible debt - Preferred dividends) / (Weighted average number of shares outstanding + Additional common shares that would have been issued at conversion)

EXAMPLE 10
A Diluted EPS Calculation Using the If-Converted Method for Convertible Debt
1. Oppnox Company (fictitious) reported net income of $750,000 for the year
ended 31 December 2018. The company had a weighted average of 690,000
shares of common stock outstanding. In addition, the company has only
one potentially dilutive security: $50,000 of 6 percent convertible bonds,
convertible into a total of 10,000 shares. Assuming a tax rate of 30 percent,
calculate Oppnox's basic and diluted EPS.
Solution:
If the debt securities had been converted, the debt securities would no
longer be outstanding and instead, an additional 10,000 shares of common
stock would be outstanding. Also, if the debt securities had been converted,
the company would not have paid interest of $3,000 on the convertible debt,
so net income available to common shareholders would have increased by
$2,100 [= $3,000(1 – 0.30)] on an after-tax basis. Exhibit 12 illustrates the
calculation of diluted EPS using the if-converted method for convertible
debt.

Exhibit 12: Calculation of Diluted EPS for Oppnox Company Using the If-Converted Method: Case of a Convertible Bond
|                                  | Basic EPS   | Diluted EPS Using If-Converted Method |
| :------------------------------- | :---------- | :------------------------------------ |
| Net income                       | $750,000    | $750,000                            |
| After-tax cost of interest         |             | 2,100                                 |
| **Numerator**                    | **$750,000** | **$752,100**                        |
| Weighted average number of shares outstanding | 690,000     | 690,000                             |
| If converted                     | 0           | 10,000                              |
| Denominator                      | 690,000     | 700,000                             |
| EPS                              | $1.09       | $1.07                                 |

Diluted EPS = (Net income - Preferred dividends) / [Weighted average number of shares outstanding + (New shares that would have been issued at option exercise - Shares that could have been purchased with cash received upon exercise) x (Proportion of year during which the financial instruments were outstanding)]

EXAMPLE 11
A Diluted EPS Calculation Using the Treasury Stock Method for Options
1. Hihotech Company (fictitious) reported net income of $2.3 million for the
year ended 30 June 2018 and had a weighted average of 800,000 common
shares outstanding. At the beginning of the fiscal year, the company has
outstanding 30,000 options with an exercise price of $35. No other poten-
tially dilutive financial instruments are outstanding. Over the fiscal year, the
company’s market price has averaged $55 per share. Calculate the company’s
basic and diluted EPS.
Solution:
Using the treasury stock method, we first calculate that the company would
have received $1,050,000 ($35 for each of the 30,000 options exercised) if all
the options had been exercised. The options would no longer be outstand-
ing; instead, 30,000 shares of common stock would be outstanding. Under
the treasury stock method, we assume that shares would be repurchased
with the cash received upon exercise of the options. At an average market
price of $55 per share, the $1,050,000 proceeds from option exercise, the
company could have repurchased 19,091 shares. Therefore, the incremental
number of shares issued is 10,909 (calculated as 30,000 minus 19,091). For
the diluted EPS calculation, no change is made to the numerator. As shown
in Exhibit 13, the company’s basic EPS was $2.88 and the diluted EPS was
$2.84.

Exhibit 13: Calculation of Diluted EPS for Hihotech Company Using the Treasury Stock Method: Case of Stock Options
|                                  | Basic EPS   | Diluted EPS Using Treasury Stock Method |
| :------------------------------- | :---------- | :-------------------------------------- |
| Net income                       | $2,300,000  | $2,300,000                            |
| **Numerator**                    | **$2,300,000** | **$2,300,000**                        |
| Weighted average number of shares outstanding | 800,000     | 800,000                               |
| If converted                     | 0           | 10,909                                |
| Denominator                      | 800,000     | 810,909                               |
| EPS                              | $2.88       | $2.84                                 |

EXAMPLE 12
Diluted EPS for Options under IFRS
1. Assuming the same facts as in Example 11, calculate the weighted average
number of shares outstanding for diluted EPS under IFRS.
Solution:
If the options had been exercised, the company would have received
$1,050,000. If this amount had been received from the issuance of new
shares at the average market price of $55 per share, the company would have
issued 19,091 shares. IFRS refer to the 19,091 shares the company would
have issued at market prices as the inferred shares. The number of shares
issued under options (30,000) minus the number of inferred shares (19,091)
equals 10,909. This amount is added to the weighted average number of
shares outstanding of 800,000 to get diluted shares of 810,909. Note that this
is the same result as that obtained under US GAAP; it is just derived in a
different manner.

EXAMPLE 13
An Antidilutive Security
1. For the year ended 31 December 2018, Dim-Cool Utility Company (fic-
titious) had net income of $1,750,000. The company had an average of
500,000 shares of common stock outstanding, 20,000 shares of convertible
preferred, and no other potentially dilutive securities. Each share of pre-
ferred pays a dividend of $10 per share, and each is convertible into three
shares of the company’s common stock. What was the company’s basic and
diluted EPS?
Solution:
If the 20,000 shares of convertible preferred had each converted into 3
shares of the company’s common stock, the company would have had an ad-
ditional 60,000 shares of common stock (3 shares of common for each of the
20,000 shares of preferred). If the conversion had taken place, the company
would not have paid preferred dividends of $200,000 ($10 per share for each
of the 20,000 shares of preferred). The effect of using the if-converted meth-
od would be EPS of $3.13, as shown in Exhibit 14. Because this is greater
than the company’s basic EPS of $3.10, the securities are said to be antidilu-
tive and the effect of their conversion would not be included in diluted EPS.
Diluted EPS would be the same as basic EPS (i.e., $3.10).

Exhibit 14: Calculation for an Antidilutive Security
|                                  | Basic EPS   | Diluted EPS Using If-Converted Method |
| :------------------------------- | :---------- | :------------------------------------ |
| Net income                       | $1,750,000  | $1,750,000                          |
| Preferred dividend               | -200,000    | 0                                     |
| **Numerator**                    | **$1,550,000** | **$1,750,000**                      |
| Weighted average number of shares outstanding | 500,000     | 500,000                             |
| If converted                     | 0           | 60,000                              |
| Denominator                      | 500,000     | 560,000                             |
| EPS                              | $3.10       | $3.13                                 |
-Exceeds basic EPS; security is antidilutive and, therefore, not included. Reported diluted EPS= $3.10.

Exhibit 15
Panel A: Income Statements for Companies A, B, and C ($)
|                               | A            | B            | C           |
| :---------------------------- | :----------- | :----------- | :---------- |
| Sales                         | $10,000,000  | $10,000,000  | $2,000,000  |
| Cost of sales                 | 3,000,000    | 7,500,000    | 600,000     |
| Gross profit                  | 7,000,000    | 2,500,000    | 1,400,000   |
| Selling, general, and administrative expenses | 1,000,000    | 1,000,000    | 200,000     |
| Research and development      | 2,000,000    | —            | 400,000     |
| Advertising                   | 2,000,000    | —            | 400,000     |
| Operating profit              | 2,000,000    | 1,500,000    | 400,000     |

Panel B: Common-Size Income Statements for Companies A, B, and C (%)
|                               | A     | B     | C     |
| :---------------------------- | :---- | :---- | :---- |
| Sales                         | 100%  | 100%  | 100%  |
| Cost of sales                 | 30    | 75    | 30    |
| Gross profit                  | 70    | 25    | 70    |
| Selling, general, and administrative expenses | 10    | 10    | 10    |
| Research and development      | 20    | 0     | 20    |
| Advertising                   | 20    | 0     | 20    |
| Operating profit              | 20    | 15    | 20    |

Exhibit 16: Median Common-Size Income Statement Statistics for the S&P 500 Classified by S&P/MSCI GICS Sector Data for 2017
|                   | Energy | Materials | Industrials | Consumer Discretionary | Consumer Staples | Health Care |
| :---------------- | :----- | :-------- | :---------- | :--------------------- | :--------------- | :---------- |
| Number of observations | 34     | 27        | 69          | 81                     | 34               | 59          |
| Gross Margin      | 37.7%  | 33.0%     | 36.8%       | 37.6%                  | 43.4%            | 59.0%       |
| Operating Margin  | 6.4%   | 14.9%     | 13.5%       | 11.0%                  | 17.2%            | 17.4%       |
| Net Profit Margin | 4.9%   | 9.9%      | 8.8%        | 6.0%                   | 10.9%            | 7.2%        |

|                               | Financials | Information Technology | Telecommunication Services | Utilities | Real Estate |
| :---------------------------- | :--------- | :--------------------- | :------------------------- | :-------- | :---------- |
| Number of observations        | 63         | 64                     | 4                          | 29        | 29          |
| Gross Margin                  | 40.5%      | 62.4%                  | 56.4%                      | 34.3%     | 39.8%       |
| Operating Margin              | 36.5%      | 21.1%                  | 15.4%                      | 21.7%     | 30.1%       |
| Net Profit Margin             | 18.5%      | 11.3%                  | 13.1%                      | 10.1%     | 21.3%       |

Net profit margin = Net income / Revenue
Gross profit margin = Gross profit / Revenue

Exhibit 17: AB InBev's Margins: Abbreviated Common-Size Income Statement
12 Months Ended December 31
|                                   | 2017       | %      | 2016       | %      | 2015       | %      |
| :-------------------------------- | :--------- | :----- | :--------- | :----- | :--------- | :----- |
| Revenue                           | 56,444     | 100.0  | 45,517     | 100.0  | 43,604     | 100.0  |
| Cost of sales                     | (21,386)   | (37.9) | (17,803)   | (39.1) | (17,137)   | (39.3) |
| Gross profit                      | 35,058     | 62.1   | 27,715     | 60.9   | 26,467     | 60.7   |
| Distribution expenses             | (5,876)    | (10.4) | (4,543)    | (10.0) | (4,259)    | (9.8)  |
| Sales and marketing expenses      | (8,382)    | (14.9) | (7,745)    | (17.0) | (6,913)    | (15.9) |
| Administrative expenses           | (3,841)    | (6.8)  | (2,883)    | (6.3)  | (2,560)    | (5.9)  |
| **Portions omitted**              |            |        |            |        |            |        |
| Profit from operations            | 17,152     | 30.4   | 12,882     | 28.3   | 13,904     | 31.9   |
| Finance cost                      | (6,885)    | (12.2) | (9,382)    | (20.6) | (3,142)    | (7.2)  |
| Finance income                    | 378        | 0.7    | 818        | 1.8    | 1,689      | 3.9    |
| Net finance income/(cost)         | (6,507)    | (11.5) | (8,564)    | (18.8) | (1,453)    | (3.3)  |
| Share of result of associates and joint ventures | 430        | 0.8    | 16         | 0.0    | 10         | 0.0    |
| Profit before tax                 | 11,076     | 19.6   | 4,334      | 9.5    | 12,461     | 28.6   |
| Income tax expense                | (1,920)    | (3.4)  | (1,613)    | (3.5)  | (2,594)    | (5.9)  |
| Profit from continuing operations | 9,155      | 16.2   | 2,721      | 6.0    | 9,867      | 22.6   |
| Profit from discontinued operations | 28         | 0.0    | 48         | 0.1    |            |        |
| Profit of the year                | 9,183      | 16.3   | 2,769      | 6.1    | 9,867      | 22.6   |

EXAMPLE 14
Other Comprehensive Income
Assume a company’s beginning shareholders’ equity is €200 million, its net
income for the year is €20 million, its cash dividends for the year are €3 million,
and there was no issuance or repurchase of common stock. The company’s actual
ending shareholders’ equity is €227 million.
1. What amount has bypassed the net income calculation by being classified as
other comprehensive income?
A. €0.
B. €7 million.
C. €10 million.
Solution to 1:
C is correct. If the company’s actual ending shareholders’ equity is €227
million, then €10 million [€227− (€200 + €20 – €3)] has bypassed the net
income calculation by being classified as other comprehensive income.
2. Which of the following statements best describes other comprehensive
income?
A. Income earned from diverse geographic and segment activities.
B. Income that increases stockholders’ equity but is not reflected as part
of net income.
C. Income earned from activities that are not part of the company’s ordi-
nary business activities.
Solution to 2:
B is correct. Answers A and C are not correct because they do not specify
whether such income is reported as part of net income and shown in the
income statement.

EXAMPLE 15
Other Comprehensive Income in Analysis
1. An analyst is looking at two comparable companies. Company A has a lower
price/earnings (P/E) ratio than Company B, and the conclusion that has
been suggested is that Company A is undervalued. As part of examining
this conclusion, the analyst decides to explore the question: What would the
company’s P/E look like if total comprehensive income per share—rather
than net income per share—were used as the relevant metric?
|                       | Company A    | Company B  |
| :-------------------- | :----------- | :--------- |
| Price                 | $35          | $30        |
| EPS                   | $1.60        | $0.90      |
| P/E ratio             | 21.9x        | 33.3x      |
| Other comprehensive income (loss) $ million | ($16.272)    | $(1.757)   |
| Shares (millions)     | 22.6         | 25.1       |

Solution:
As shown in the following table, part of the explanation for Company A’s
lower P/E ratio may be that its significant losses—accounted for as other
comprehensive income (OCI)—are not included in the P/E ratio.
|                           | Company A | Company B |
| :------------------------ | :-------- | :-------- |
| Price                     | $35       | $30       |
| EPS                       | $1.60     | $0.90     |
| OCI (loss) $ million      | ($16.272) | $(1.757)  |
| Shares (millions)         | 22.6      | 25.1      |
| OCI (loss) per share      | $(0.72)   | $(0.07)   |
| Comprehensive EPS = EPS + OCI per share | $ 0.88    | $0.83     |
| Price/Comprehensive EPS ratio | 39.8x     | 36.1x     |

PRACTICE PROBLEMS
1. Expenses on the income statement may be grouped by:
A. nature, but not by function.
B. function, but not by nature.
C. either function or nature.

2. An example of an expense classification by function is:
A. tax expense.
B. interest expense.
C. cost of goods sold.

3. Denali Limited, a manufacturing company, had the following income statement information:
| Revenue                 | $4,000,000 |
| :---------------------- | :--------- |
| Cost of goods sold      | $3,000,000 |
| Other operating expenses | $500,000   |
| Interest expense        | $100,000   |
| Tax expense             | $120,000   |
Denali's gross profit is equal to:
A. $280,000.
B. $500,000.
C. $1,000,000.

4. Under IFRS, income includes increases in economic benefits from:
A. increases in liabilities not related to owners' contributions.
B. enhancements of assets not related to owners' contributions.
C. increases in owners' equity related to owners' contributions.

5. Fairplay had the following information related to the sale of its products during 2009, which was its first year of business:
| Revenue             | $1,000,000 |
| :------------------ | :--------- |
| Returns of goods sold | $100,000   |
| Cash collected      | $800,000   |
| Cost of goods sold  | $700,000   |
Under the accrual basis of accounting, how much net revenue would be reported
on Fairplay's 2009 income statement?
A. $200,000.
B. $900,000.
C. $1,000,000.

6. Apex Consignment sells items over the internet for individuals on a consignment
basis. Apex receives the items from the owner, lists them for sale on the internet,
and receives a 25 percent commission for any items sold. Apex collects the full
amount from the buyer and pays the net amount after commission to the owner.
Unsold items are returned to the owner after 90 days. During 2009, Apex had the
following information:
*   Total sales price of items sold during 2009 on consignment was €2,000,000.
*   Total commissions retained by Apex during 2009 for these items was
€500,000.
How much revenue should Apex report on its 2009 income statement?
Α. €500,000.
Β. €2,000,000.
C. €1,500,000.

7. A company previously expensed the incremental costs of obtaining a con\fract.
All else being equal, adopting the May 2014 IASB and FASB converged account-
ing standards on revenue recognition makes the company's profitability initially
appear:
A. lower.
B. unchanged.
C. higher.

8. During 2009, Accent Toys Plc., which began business in October of that year,
purchased 10,000 units of a toy at a cost of £10 per unit in October. The toy sold
well in October. In anticipation of heavy December sales, Accent purchased 5,000
additional units in November at a cost of £11 per unit. During 2009, Accent sold
12,000 units at a price of £15 per unit. Under the first in, first out (FIFO) method,
what is Accent's cost of goods sold for 2009?
A. £120,000.
B. £122,000.
C. £124,000.

9. Using the same information as in the previous question, what would Accent's cost
of goods sold be under the weighted average cost method?
A. £120,000.
B. £122,000.
C. £124,000.

10. Which inventory method is least likely to be used under IFRS?
A. First in, first out (FIFO).
B. Last in, first out (LIFO).
C. Weighted average.

11. At the beginning of 2009, Glass Manufacturing purchased a new machine for its
assembly line at a cost of $600,000. The machine has an estimated useful life of
10 years and estimated residual value of $50,000. Under the straight-line method,
how much depreciation would Glass take in 2010 for financial reporting purpos-
es?
A. $55,000.
B. $60,000.
C. $65,000.

12. Using the same information as in Question 11, how much depreciation would
Glass take in 2009 for financial reporting purposes under the double-declining
balance method?
A. $60,000.
B. $110,000.
C. $120,000.

13. Which combination of depreciation methods and useful lives is most conserva-
tive in the year a depreciable asset is acquired?
A. Straight-line depreciation with a short useful life.
B. Declining balance depreciation with a long useful life.
C. Declining balance depreciation with a short useful life.

14. Under IFRS, a loss from the destruction of property in a fire would most likely be
classified as:
A. continuing operations.
B. discontinued operations.
C. other comprehensive income.

15. A company chooses to change an accounting policy. This change requires that, if
\fractical, the company restate its financial statements for:
A. all prior periods.
B. current and future periods.
C. prior periods shown in a report.

16. For 2009, Flamingo Products had net income of $1,000,000. At 1 January 2009,
there were 1,000,000 shares outstanding. On 1 July 2009, the company issued
100,000 new shares for $20 per share. The company paid $200,000 in dividends to
common shareholders. What is Flamingo's basic earnings per share for 2009?
A. $0.80.
B. $0.91.
C. $0.95.

17. A company with no debt or convertible securities issued publicly traded common
stock three times during the current fiscal year. Under both IFRS and US GAAP,
the company's:
A. basic EPS equals its diluted EPS.
B. capital structure is considered complex at year-end.
C. basic EPS is calculated by using a simple average number of shares
outstanding.

18. For its fiscal year-end, Sublyme Corporation reported net income of $200 million
and a weighted average of 50,000,000 common shares outstanding. There are
2,000,000 convertible preferred shares outstanding that paid an annual dividend
of $5. Each preferred share is convertible into two shares of the common stock.
The diluted EPS is closest to:
A. $3.52.
B. $3.65.
C. $3.70.

19. For its fiscal year-end, Calvan Water Corporation (CWC) reported net income
of $12 million and a weighted average of 2,000,000 common shares outstanding.
The company paid $800,000 in preferred dividends and had 100,000 options out-
standing with an average exercise price of $20. CWC's market price over the year
averaged $25 per share. CWC's diluted EPS is closest to:
A. $5.33.
Β. $5.54.
C. $5.94.

20. Laurelli Builders (LB) reported the following financial data for year-end 31 De-
cember:
| Common shares outstanding, 1 January   | 2,020,000  |
| :------------------------------------- | :--------- |
| Common shares issued as stock dividend, 1 June | 380,000    |
| Warrants outstanding, 1 January        | 500,000    |
| Net income                             | $3,350,000 |
| Preferred stock dividends paid         | $430,000   |
| Common stock dividends paid            | $240,000   |
Which statement about the calculation of LB's EPS is most accurate?
A. LB's basic EPS is $1.12.
B. LB's diluted EPS is equal to or less than its basic EPS.
C. The weighted average number of shares outstanding is 2,210,000.

21. Cell Services Inc. (CSI) had 1,000,000 average shares outstanding during all of
2009. During 2009, CSI also had 10,000 options outstanding with exercise prices
of $10 each. The average stock price of CSI during 2009 was $15. For purposes
of computing diluted earnings per share, how many shares would be used in the
denominator?
A. 1,003,333.
B. 1,006,667.
C. 1,010,000.

22. When calculating diluted EPS, which of the following securities in the capital
structure increases the weighted average number of common shares outstanding
without affecting net income available to common shareholders?
A. Stock options
B. Convertible debt that is dilutive
C. Convertible preferred stock that is dilutive

23. Which statement is most accurate? A common size income statement:
A. restates each line item of the income statement as a percentage of net
income.
B. allows an analyst to conduct cross-sectional analysis by removing the effect
of company size.
C. standardizes each line item of the income statement but fails to help an
analyst identify differences in companies' strategies.

24. Selected year-end financial statement data for Workhard are shown below.
$ millions
| Beginning shareholders' equity         | 475 |
| :------------------------------------- | :-- |
| Ending shareholders' equity            | 493 |
| Unrealized gain on available-for-sale securities | 5   |
| Unrealized loss on derivatives accounted for as hedges | -3  |
| Foreign currency translation gain on consolidation | 2   |
| Dividends paid                         | 1   |
| Net income                             | 15  |
Workhard's comprehensive income for the year:
A. is $18 million.
B. is increased by the derivatives accounted for as hedges.
C. includes $4 million in other comprehensive income.

25. When preparing an income statement, which of the following items would most
likely be classified as other comprehensive income?
A. A foreign currency translation adjustment
B. An unrealized gain on a security held for trading purposes
C. A realized gain on a derivative con\fract not accounted for as a hedge

SOLUTIONS
1. C is correct. IAS No. 1 states that expenses may be categorized by either nature or function.
2. C is correct. Cost of goods sold is a classification by function. The other two expenses represent classifications by nature.
3. C is correct. Gross margin is revenue minus cost of goods sold. Answer A represents net income and B represents operating income.
4. B is correct. Under IFRS, income includes increases in economic benefits from increases in assets, enhancement of assets, and decreases in liabilities.
5. B is correct. Net revenue is revenue for goods sold during the period less any returns and allowances, or $1,000,000 minus $100,000 = $900,000.
6. A is correct. Apex is not the owner of the goods and should only report its net commission as revenue.
7. C is correct. Under the converged accounting standards, the incremental costs of obtaining a con\fract and certain costs incurred to fulfill a con\fract must be capitalized. If a company expensed these incremental costs in the years prior to adopting the converged standards, all else being equal, its profitability will appear higher under the converged standards.
8. B is correct. Under the first in, first out (FIFO) method, the first 10,000 units sold came from the October purchases at £10, and the next 2,000 units sold came from the November purchases at £11.
9. C is correct. Under the weighted average cost method:
| October purchases   | 10,000 units | $100,000 |
| :------------------ | :----------- | :------- |
| November purchases  | 5,000 units  | $55,000  |
| Total               | 15,000 units | $155,000 |
$155,000/15,000 units = $10.3333
$10.3333 × 12,000 units = $124,000
10. B is correct. The last in, first out (LIFO) method is not permitted under IFRS. The other two methods are permitted.
11. A is correct. Straight-line depreciation would be ($600,000 – $50,000)/10, or $55,000.
12. C is correct. Double-declining balance depreciation would be $600,000 × 20 per-cent (twice the straight-line rate). The residual value is not sub\fracted from the initial book value to calculate depreciation. However, the book value (carrying amount) of the asset will not be reduced below the estimated residual value.
13. C is correct. This would result in the highest amount of depreciation in the first year and hence the lowest amount of net income relative to the other choices.
14. A is correct. A fire may be infrequent, but it would still be part of continuing op-erations and reported in the profit and loss statement. Discontinued operations relate to a decision to dispose of an operating division.
15. C is correct. If a company changes an accounting policy, the financial statements for all fiscal years shown in a company's financial report are presented, if \frac-tical, as if the newly adopted accounting policy had been used throughout the entire period; this retrospective application of the change makes the financial re-sults of any prior years included in the report comparable. Notes to the financial statements describe the change and explain the justification for the change.
16. C is correct. The weighted average number of shares outstanding for 2009 is 1,050,000. Basic earnings per share would be $1,000,000 divided by 1,050,000, or $0.95.
17. A is correct. Basic and diluted EPS are equal for a company with a simple capital structure. A company that issues only common stock, with no financial instru-ments that are potentially convertible into common stock has a simple capital structure. Basic EPS is calculated using the weighted average number of shares outstanding.
18. C is correct.
Diluted EPS = (Net income)/(Weighted average number of shares outstanding + New common shares that would have been issued at conversion)
= $200,000,000/[50,000,000 + (2,000,000 × 2)]
= $3.70
The diluted EPS assumes that the preferred dividend is not paid and that the shares are converted at the beginning of the period.
19. B is correct. The formula to calculate diluted EPS is as follows:
Diluted EPS = (Net income – Preferred dividends)/[Weighted average number of shares outstanding + (New shares that would have been issued at option exer-cise – Shares that could have been purchased with cash received upon exercise) × (Proportion of year during which the financial instruments were outstanding)].
The underlying assumption is that outstanding options are exercised, and then the proceeds from the issuance of new shares are used to repurchase shares already outstanding:
Proceeds from option exercise = 100,000 × $20 = $2,000,000
Shares repurchased = $2,000,000/$25 = 80,000
The net increase in shares outstanding is thus 100,000 – 80,000 = 20,000. There-fore, the diluted EPS for CWC = ($12,000,000 – $800,000)/2,020,000 = $5.54.
20. B is correct. LB has warrants in its capital structure; if the exercise price is less than the weighted average market price during the year, the effect of their conversion is to increase the weighted average number of common shares out-standing, causing diluted EPS to be lower than basic EPS. If the exercise price is equal to the weighted average market price, the number of shares issued equals the number of shares repurchased. Therefore, the weighted average number of common shares outstanding is not affected and diluted EPS equals basic EPS. If the exercise price is greater than the weighted average market price, the effect of their conversion is anti-dilutive. As such, they are not included in the calculation of basic EPS. LB's basic EPS is $1.22 [= ($3,350,000 – $430,000)/2,400,000]. Stock dividends are treated as having been issued retroactively to the beginning of the period.
21. A is correct. With stock options, the treasury stock method must be used. Under that method, the company would receive $100,000 (10,000 × $10) and would re-purchase 6,667 shares ($100,000/$15). The shares for the denominator would be:
| Shares outstanding      | 1,000,000 |
| :---------------------- | :-------- |
| Options exercises       | 10,000    |
| Treasury shares purchased | (6,667)   |
| Denominator             | 1,003,333 |
22. A is correct. When a company has stock options outstanding, diluted EPS is calculated as if the financial instruments had been exercised and the company had used the proceeds from the exercise to repurchase as many shares possible at the weighted average market price of common stock during the period. As a result, the conversion of stock options increases the number of common shares outstanding but has no effect on net income available to common shareholders. The conversion of convertible debt increases the net income available to common shareholders by the after-tax amount of interest expense saved. The conversion of convertible preferred shares increases the net income available to common shareholders by the amount of preferred dividends paid; the numerator becomes the net income.
23. B is correct. Common size income statements facilitate comparison across time periods (time-series analysis) and across companies (cross-sectional analysis) by stating each line item of the income statement as a percentage of revenue. The relative performance of different companies can be more easily assessed because scaling the numbers removes the effect of size. A common size income statement states each line item on the income statement as a percentage of revenue. The standardization of each line item makes a common size income statement useful for identifying differences in companies' strategies.
24. C is correct. Comprehensive income includes both net income and other com-prehensive income.
Other comprehensive income = Unrealized gain on available-for-sale securities – Unrealized loss on derivatives accounted for as hedges + Foreign currency transla-tion gain on consolidation
= $5 million – $3 million + $2 million
= $4 million
Alternatively,
Comprehensive income – Net income = Other comprehensive income
Comprehensive income = (Ending shareholders equity – Beginning shareholders equity) + Dividends
= ($493 million – $475 million) + $1 million
= $18 million + $1 million = $19 million
Net income is $15 million so other comprehensive income is $4 million.
25. A is correct. Other comprehensive income includes items that affect sharehold-ers' equity but are not reflected in the company's income statement. In consoli-dating the financial statements of foreign subsidiaries, the effects of translating the subsidiaries' balance sheet assets and liabilities at current exchange rates are included as other comprehensive income.
```