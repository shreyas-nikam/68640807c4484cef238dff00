
# Technical Specifications for a Streamlit Application: Multiple Linear Regression Assumptions Analyzer

## Overview

The "Multiple Linear Regression Assumptions Analyzer" is a Streamlit application designed to educate users on the critical assumptions underlying multiple linear regression models and provide int\fractive visualizations to diagnose potential violations of these assumptions. The application will leverage readily available datasets from Python libraries, train a linear regression model, and then generate various diagnostic plots and statistical test results.

#### Learning Outcomes
-   Users will understand the assumptions underlying a multiple linear regression model.
-   Users will learn to interpret residual plots and statistical tests to identify potential violations of these assumptions.

## Step-by-Step Development Process

The development of the "Multiple Linear Regression Assumptions Analyzer" application will follow a structured approach:

1.  **Project Setup**: Initialize a new Python project, create `app.py` as the main Streamlit application file, and define a `requirements.txt` file to manage dependencies.
2.  **Library Imports**: Import all necessary libraries, including `streamlit`, `pandas`, `numpy`, `sklearn` (for datasets and linear model), `statsmodels` (for statistical tests and diagnostics), `matplotlib.pyplot`, and `seaborn` (for visualizations).
3.  **Data Loading Module**: Implement a function to load a suitable dataset for multiple linear regression from Python libraries (e.g., `sklearn.datasets.load_boston` or `sklearn.datasets.make_regression`). This function will make the data available for the application.
4.  **Model Training Module**: Develop a function that takes the loaded dataset, identifies dependent and independent variables, and trains a multiple linear regression model using `sklearn.linear_model.LinearRegression`. This module will also generate predicted values ($\hat{Y}$) and residuals ($e_i$).
5.  **Assumption Calculation Module**:
    *   **Residuals & Predicted Values**: Calculate $e_i = Y_i - \hat{Y}_i$ and $\hat{Y}_i$.
    *   **Multicollinearity**: Implement calculation of Variance Inflation Factor (VIF) for each independent variable using `statsmodels.stats.outliers_influence.variance_inflation_factor`.
    *   **Normality Test**: Perform Shapiro-Wilk test for normality of residuals using `scipy.stats.shapiro`.
    *   **Homoscedasticity Test**: Implement Breusch-Pagan test for homoscedasticity using `statsmodels.stats.api.het_breuschpagan`.
    *   **Autocorrelation Test**: Implement Durbin-Watson test for autocorrelation using `statsmodels.stats.stattools.durbin_watson`.
6.  **Visualization Module**: Create dedicated functions for generating each required plot:
    *   **Scatterplot Matrix**: Utilizes `seaborn.pairplot` to visualize relationships between all independent variables and the dependent variable.
    *   **Residuals vs. Predicted Values Plot**: Uses `matplotlib.pyplot.scatter` to plot residuals against predicted values, adding a horizontal line at $y=0$.
    *   **Residuals vs. Factors Plots**: Iteratively plots residuals against each independent variable, using `matplotlib.pyplot.scatter`.
    *   **Q-Q Plot for Normality**: Uses `statsmodels.graphics.gofplots.qqplot` to assess the normality of residuals.
    *   **Histogram of Residuals**: Uses `matplotlib.pyplot.hist` to visualize the distribution of residuals.
7.  **Streamlit UI Integration**:
    *   **Sidebar**: Use `st.sidebar` for dataset selection and potentially other int\fractive controls.
    *   **Main Content Area**: Structure the main page using `st.title`, `st.header`, `st.subheader`, and `st.markdown` for textual explanations. Display dataframes with `st.dataframe` and plots with `st.pyplot` or `st.plotly_chart`.
    *   **Caching**: Employ `@st.cache_data` decorator for functions that perform data loading, model training, and complex calculations to optimize application performance.

## Core Concepts and Mathematical Foundations

This section details the fundamental concepts of multiple linear regression and its underlying assumptions, along with their mathematical representations and \fractical applications, following the specified LaTeX formatting.

### Multiple Linear Regression Model
The multiple linear regression model describes the linear relationship between a dependent variable and two or more independent variables. It is used to predict the value of the dependent variable based on the values of the independent variables. The model is represented as:
$$
Y = \eta_0 + \eta_1 X_1 + \eta_2 X_2 + \ldots + \eta_p X_p + \epsilon
$$
Where:
-   $Y$: The dependent variable (the outcome we are trying to predict)
-   $X_1, X_2, \ldots, X_p$: The independent variables (predictors or factors)
-   $\eta_0$: The Y-intercept (the value of Y when all X variables are zero)
-   $\eta_1, \eta_2, \ldots, \eta_p$: The regression coefficients, representing the change in Y for a one-unit change in the corresponding X, holding other X variables constant
-   $\epsilon$: The error term (or residual), representing the unobserved factors that influence Y

This formula defines how the dependent variable is linearly related to the independent variables, with an added error term to account for variability not explained by the model. Its \fractical application lies in forecasting, understanding relationships between variables, and identifying the strength and direction of these relationships.

### Assumptions of Multiple Linear Regression

For the coefficients of a multiple linear regression model to be Best Linear Unbiased Estimators (BLUE), several assumptions about the data and the error term must hold. Violations of these assumptions can lead to biased or inefficient estimates and unreliable inferences.

#### 1. Linearity
The relationship between the independent variables and the dependent variable is linear. This means that the expected value of the dependent variable is a straight-line function of the independent variables.
$$
E(\epsilon_i | X_{i1}, \ldots, X_{ip}) = 0
$$
Where:
-   $E(\epsilon_i | X_{i1}, \ldots, X_{ip})$: The expected value of the error term for observation $i$, conditional on the independent variables.

This assumption implies that the model correctly captures the functional form of the relationship. A common way to diagnose this is by examining the **Residuals vs. Predicted Values Plot** and **Residuals vs. Independent Variables Plots** for any non-linear patterns (e.g., U-shape, inverted U-shape, or other curves). A random scatter of points around the zero line indicates linearity.

#### 2. Independence of Residuals (No Autocorrelation)
The residuals (error terms) are independent of each other. This means that the error for one observation does not influence the error for any other observation. This is particularly important in time-series data where consecutive observations might be correlated.
This assumption ensures that the observations provide independent information. It is commonly assessed using the **Durbin-Watson test statistic** and by examining the **Residuals vs. Predicted Values Plot** for any discernible patterns or trends (e.g., consecutive positive or negative residuals). A Durbin-Watson statistic near 2 suggests no autocorrelation.

#### 3. Homoscedasticity (Constant Variance of Residuals)
The variance of the residuals is constant across all levels of the independent variables. In other words, the spread of the residuals should be uniform across the range of predicted values and independent variables.
$$
Var(\epsilon_i | X_{i1}, \ldots, X_{ip}) = \sigma^2
$$
Where:
-   $Var(\epsilon_i | X_{i1}, \ldots, X_{ip})$: The variance of the error term for observation $i$, conditional on the independent variables.
-   $\sigma^2$: A constant variance.

This assumption ensures that the precision of the predictions is consistent across the data. Violation, known as heteroscedasticity, often appears as a "fan" or "cone" shape in the **Residuals vs. Predicted Values Plot**, where the spread of residuals increases or decreases with predicted values. The **Breusch-Pagan test** can formally test for homoscedasticity.

#### 4. Normality of Residuals
The residuals are normally distributed. This assumption is crucial for hypothesis testing and confidence interval estimation, especially with smaller sample sizes.
This assumption states that the random errors follow a normal distribution, allowing for valid statistical inferences. It is typically assessed visually using a **Histogram of Residuals** (checking for a bell-shaped curve) and a **Normal Q-Q Plot** (checking if points fall along a straight diagonal line). The **Shapiro-Wilk test** provides a formal statistical test for normality.

#### 5. No Multicollinearity
The independent variables are not highly correlated with each other. High correlation among independent variables (multicollinearity) makes it difficult to ascertain the individual effect of each independent variable on the dependent variable, leading to unstable and less reliable regression coefficients.
This assumption ensures that each independent variable provides unique information to the model. It is diagnosed by examining the **Scatterplot Matrix** (looking for strong linear relationships between independent variables) and, more formally, by calculating the **Variance Inflation Factor (VIF)** for each independent variable.
The Variance Inflation Factor ($VIF_j$) for an independent variable $X_j$ is calculated using:
$$
VIF_j = \\frac{1}{1 - R_j^2}
$$
Where:
-   $R_j^2$: The coefficient of determination obtained from regressing the independent variable $X_j$ on all other independent variables in the model.

A high VIF value (typically above 5 or 10) indicates significant multicollinearity, suggesting that the independent variable $X_j$ is highly correlated with other independent variables. This formula quantifies the extent to which the variance of an estimated regression coefficient is inflated due to multicollinearity, helping to identify problematic predictors.

### Coefficient of Determination ($R^2$)
The coefficient of determination, denoted as $R^2$, measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It provides an indication of how well the regression model fits the observed data.
$$
R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}
$$
Where:
-   $SS_{res}$: The sum of squares of residuals, representing the unexplained variance (sum of squared differences between actual $Y_i$ and predicted $\hat{Y}_i$).
-   $SS_{tot}$: The total sum of squares, representing the total variance in the dependent variable (sum of squared differences between actual $Y_i$ and the mean of $Y$).

This formula quantifies the goodness of fit of the model. An $R^2$ value closer to 1 indicates that a larger proportion of the variance in the dependent variable is explained by the model, implying a better fit. Its \fractical application is to evaluate the explanatory power of the regression model.

## Required Libraries and Dependencies

The application will rely on the following Python libraries for data handling, model building, statistical analysis, and visualization:

-   **`streamlit`**: Version 1.32.2 (or compatible)
    *   **Specific functions/modules used**: `st.title`, `st.header`, `st.subheader`, `st.markdown`, `st.write`, `st.dataframe`, `st.pyplot`, `st.plotly_chart`, `st.sidebar`, `st.selectbox`, `st.slider`, `st.expander`, `st.cache_data`.
    *   **Role**: Core framework for building the int\fractive web application, managing UI components, and handling user input.
-   **`pandas`**: Version 2.2.1 (or compatible)
    *   **Specific functions/modules used**: `pd.DataFrame`, `pd.concat`, `pd.set_option`.
    *   **Role**: Efficient data manipulation, loading datasets into DataFrames, and preparing data for modeling and visualization.
-   **`numpy`**: Version 1.26.4 (or compatible)
    *   **Specific functions/modules used**: `np.array`, `np.sqrt`, `np.mean`.
    *   **Role**: Fundamental package for numerical operations, array manipulation, and mathematical calculations within the application.
-   **`scikit-learn` (sklearn)**: Version 1.4.1.post1 (or compatible)
    *   **Specific functions/modules used**:
        *   `sklearn.linear_model.LinearRegression`: For creating and fitting the linear regression model.
        *   `sklearn.datasets.load_boston` (if available and suitable, or consider `load_diabetes`, `load_wine`, `fetch_california_housing`): For loading readily available datasets.
        *   `sklearn.model_selection.train_test_split`: Although not explicitly requested for model evaluation, it might be implicitly used for robust training.
    *   **Role**: Provides the linear regression model implementation and access to example datasets for demonstration.
-   **`statsmodels`**: Version 0.14.1 (or compatible)
    *   **Specific functions/modules used**:
        *   `statsmodels.api.OLS`: For Ordinary Least Squares regression to get a detailed model summary (including p-values, R-squared, etc.).
        *   `statsmodels.stats.outliers_influence.variance_inflation_factor`: For calculating VIF to diagnose multicollinearity.
        *   `statsmodels.graphics.gofplots.qqplot`: For generating Q-Q plots for normality assessment.
        *   `statsmodels.stats.stattools.durbin_watson`: For Durbin-Watson test of autocorrelation.
        *   `statsmodels.stats.api.het_breuschpagan`: For Breusch-Pagan test of heteroscedasticity.
    *   **Role**: Essential for in-depth statistical analysis, hypothesis testing, and specific diagnostic tests for regression assumptions.
-   **`matplotlib.pyplot`**: Version 3.8.3 (or compatible)
    *   **Specific functions/modules used**: `plt.figure`, `plt.scatter`, `plt.axhline`, `plt.hist`, `plt.title`, `plt.xlabel`, `plt.ylabel`, `plt.tight_layout`, `plt.show`.
    *   **Role**: Primary library for creating static plots (Residuals vs. Predicted, Residuals vs. Factors, Histograms).
-   **`seaborn`**: Version 0.13.2 (or compatible)
    *   **Specific functions/modules used**: `sns.pairplot`.
    *   **Role**: Used for generating high-level statistical graphics, specifically the scatterplot matrix, which is crucial for visualizing relationships between variables and initial multicollinearity checks.
-   **`scipy`**: Version 1.12.0 (or compatible)
    *   **Specific functions/modules used**: `scipy.stats.shapiro`.
    *   **Role**: Provides statistical functions, including the Shapiro-Wilk test for normality.

## Implementation Details

The application's internal structure will be modular, with clear separation of concerns for data handling, model training, assumption testing, and visualization.

### Data Handling (`data_loader.py` or integrated into `app.py`)
-   **`load_sample_data(dataset_name)` function**:
    *   Takes `dataset_name` (e.g., "Boston", "Diabetes", "Synthetic") as input.
    *   For "Boston" (if applicable): Load using `load_boston()`, create Pandas DataFrame, assign feature names and target.
    *   For "Diabetes" or "Wine": Load using respective `load_diabetes()` or `load_wine()`, similar DataFrame creation.
    *   For "Synthetic": Use `sklearn.datasets.make_regression` to generate a custom dataset with controllable number of features, samples, and noise, ensuring a clear linear relationship by default.
    *   Returns the DataFrame containing both independent and dependent variables.

### Model Analysis (`model_analyzer.py` or integrated)
-   **`train_and_predict(df, target_column)` function**:
    *   Splits `df` into features (X) and target (Y) based on `target_column`.
    *   Initializes `sklearn.linear_model.LinearRegression`.
    *   Fits the model to the data.
    *   Calculates predicted values ($\hat{Y}$) and residuals ($e = Y - \hat{Y}$).
    *   Returns the fitted model, predicted values, residuals, and feature names.
-   **`get_ols_summary(X, Y)` function**:
    *   Uses `statsmodels.api.OLS` to fit the model and generate a comprehensive statistical summary, including coefficients, R-squared, p-values, etc. This is preferred for displaying statistical details to the user over `sklearn`'s summary.
    *   Returns the `statsmodels` OLS results object.
-   **`perform_assumption_tests(X, residuals)` function**:
    *   **VIF Calculation**: Iterates through each feature in `X` to calculate its VIF using `variance_inflation_factor` (requires adding a constant to X for `statsmodels.OLS` if not already done).
    *   **Shapiro-Wilk Test**: Applies `shapiro` to the `residuals`. Returns W-statistic and p-value.
    *   **Breusch-Pagan Test**: Applies `het_breuschpagan` to `residuals` and `X`. Returns test statistics and p-values.
    *   **Durbin-Watson Test**: Applies `durbin_watson` to `residuals`. Returns the Durbin-Watson statistic.
    *   Returns a dictionary or object containing all test results.

### Visualization (`plot_generator.py` or integrated)
-   **`plot_scatterplot_matrix(df, target_column)` function**:
    *   Takes the full DataFrame and target column name.
    *   Uses `seaborn.pairplot` to generate the matrix.
    *   Configures `pairplot` to show scatterplots for variable pairs and histograms/KDEs for single variables.
    *   Returns the `matplotlib` figure.
-   **`plot_residuals_vs_predicted(residuals, predicted_values)` function**:
    *   Creates a `matplotlib` figure and axes.
    *   Plots `residuals` against `predicted_values` using `plt.scatter`.
    *   Adds a horizontal line at $y=0$ for reference.
    *   Labels axes and sets title.
    *   Returns the `matplotlib` figure.
-   **`plot_residuals_vs_factors(X, residuals, feature_names)` function**:
    *   Generates individual plots of residuals against each independent variable.
    *   Creates a grid of subplots if there are many features.
    *   Labels axes and sets titles for each subplot.
    *   Returns the `matplotlib` figure.
-   **`plot_qq_plot(residuals)` function**:
    *   Uses `statsmodels.graphics.gofplots.qqplot` with `fit=True` to generate the Q-Q plot.
    *   Adds a reference line (e.g., 45-degree line).
    *   Labels axes and sets title.
    *   Returns the `matplotlib` figure.
-   **`plot_residuals_histogram(residuals)` function**:
    *   Creates a `matplotlib` figure and axes.
    *   Plots a histogram of `residuals` using `plt.hist`.
    *   Adds a normal distribution curve overlay for comparison (optional, but good for visual aid).
    *   Labels axes and sets title.
    *   Returns the `matplotlib` figure.

## User Interface Components

The Streamlit application will present an intuitive and int\fractive interface to guide users through the multiple linear regression assumption analysis.

### Sidebar (`st.sidebar`)
-   **Application Title**: "Multiple Linear Regression Assumptions Analyzer".
-   **Dataset Selection**: A `st.sidebar.selectbox` allowing users to choose from predefined datasets (e.g., "Boston Housing (Sample)", "Synthetic Regression Data").
-   **Synthetic Data Controls (if applicable)**: If "Synthetic Regression Data" is selected, provide `st.sidebar.slider` widgets for:
    *   Number of samples.
    *   Number of features.
    *   Noise level.
-   **Variable Selection (Optional but Recommended)**: For selected datasets, `st.sidebar.selectbox` for choosing the dependent variable and `st.sidebar.multiselect` for selecting independent variables. This allows custom model building.

### Main Content Area
-   **Overview and Learning Outcomes**:
    *   `st.title("Multiple Linear Regression Assumptions Analyzer")`
    *   `st.markdown("This application helps explore and diagnose assumptions of multiple linear regression.")`
    *   `st.subheader("Learning Outcomes:")`
    *   `st.markdown("- **Assumptions underlying multiple linear regression**")`
    *   `st.markdown("- **Explain the assumptions underlying a multiple linear regression model and interpret residual plots indicating potential violations of these assumptions**")`

-   **Dataset Information**:
    *   `st.header("1. Dataset Overview")`
    *   `st.write("Selected Dataset Details:")`
    *   `st.dataframe(df.head())` (display first few rows of the chosen dataset).
    *   `st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")`

-   **Multiple Linear Regression Model**:
    *   `st.header("2. Multiple Linear Regression Model")`
    *   `st.markdown("The multiple linear regression model is expressed as:")`
    *   `st.latex("Y = \eta_0 + \eta_1 X_1 + \eta_2 X_2 + \\ldots + \eta_p X_p + \epsilon")` (using `st.latex` for display equations).
    *   `st.markdown("Where: ...")` (variable descriptions).
    *   `st.subheader("Model Summary")`
    *   Display `statsmodels` OLS summary table using `st.write(model_summary.summary().as_html(), unsafe_allow_html=True)` for detailed statistical output.

-   **Assumption Analysis Section**: This section will be divided into sub-sections for each assumption, with relevant plots and test results.
    *   `st.header("3. Assumption Analysis and Diagnostics")`

    *   **3.1. Linearity**
        *   `st.subheader("3.1. Linearity Assumption")`
        *   `st.markdown("The relationship between independent and dependent variables should be linear. Check Residuals vs. Predicted plot for patterns.")`
        *   `st.subheader("Residuals vs. Predicted Value of Dependent Variable")`
        *   `st.pyplot(plot_residuals_vs_predicted(...))`
        *   `st.markdown("Ideally, residuals should be randomly scattered around zero. Any discernible pattern (e.g., curve, U-shape) indicates a violation.")`
        *   `st.subheader("Regression Residuals vs. Factors (Independent Variables)")`
        *   `st.pyplot(plot_residuals_vs_factors(...))`
        *   `st.markdown("Similarly, check individual plots of residuals against each independent variable for any non-linear trends.")`

    *   **3.2. Independence of Residuals (No Autocorrelation)**
        *   `st.subheader("3.2. Independence of Residuals Assumption")`
        *   `st.markdown("Residuals should be independent of each other (no autocorrelation).")`
        *   `st.write(f"Durbin-Watson Statistic: {durbin_watson_stat:.3f}")`
        *   `st.markdown("A Durbin-Watson statistic close to 2 indicates no autocorrelation. Values significantly below 2 suggest positive autocorrelation, while values significantly above 2 suggest negative autocorrelation.")`

    *   **3.3. Homoscedasticity (Constant Variance of Residuals)**
        *   `st.subheader("3.3. Homoscedasticity Assumption")`
        *   `st.markdown("The variance of residuals should be constant across all levels of independent variables. Look for a 'fan' or 'cone' shape in the Residuals vs. Predicted plot.")`
        *   *(Re-display or reference the "Residuals vs. Predicted Value" plot if needed, or provide direct interpretation for it here)*
        *   `st.subheader("Breusch-Pagan Test for Homoscedasticity")`
        *   `st.write(f"Lagrange Multiplier Statistic: {lm_stat:.3f}, P-value: {bp_p_value:.3f}")`
        *   `st.markdown("A small p-value (e.g., < 0.05) from the Breusch-Pagan test suggests that heteroscedasticity is present.")`

    *   **3.4. Normality of Residuals**
        *   `st.subheader("3.4. Normality of Residuals Assumption")`
        *   `st.markdown("Residuals should follow a normal distribution.")`
        *   `st.subheader("Histogram of Residuals")`
        *   `st.pyplot(plot_residuals_histogram(...))`
        *   `st.markdown("A bell-shaped histogram centered around zero suggests normality.")`
        *   `st.subheader("Normal Q-Q Plot of Residuals")`
        *   `st.pyplot(plot_qq_plot(...))`
        *   `st.markdown("Points falling close to the 45-degree line indicate normality. Deviations from the line, especially at the tails, suggest non-normality.")`
        *   `st.subheader("Shapiro-Wilk Test for Normality")`
        *   `st.write(f"Shapiro-Wilk W-statistic: {shapiro_w_stat:.3f}, P-value: {shapiro_p_value:.3f}")`
        *   `st.markdown("A small p-value (e.g., < 0.05) from the Shapiro-Wilk test suggests that residuals are not normally distributed.")`

    *   **3.5. No Multicollinearity**
        *   `st.subheader("3.5. No Multicollinearity Assumption")`
        *   `st.markdown("Independent variables should not be highly correlated with each other.")`
        *   `st.subheader("Scatterplot Matrix of Returns and Factors")`
        *   `st.pyplot(plot_scatterplot_matrix(...))` (using `st.pyplot` if seaborn figure is returned).
        *   `st.markdown("Visually inspect scatterplots between independent variables for strong linear relationships.")`
        *   `st.subheader("Variance Inflation Factor (VIF) Scores")`
        *   `st.dataframe(vif_df)` (display a DataFrame of VIF scores for each independent variable).
        *   `st.markdown("VIF values generally exceeding 5 or 10 indicate problematic levels of multicollinearity.")`

**Note on Images**: The provided document "Financial Statement Analysis" contains no images or graphs related to multiple linear regression assumptions. Therefore, no such images can be ex\fracted or directly referenced for this application's technical specifications. The application will dynamically generate the required diagnostic plots based on the selected dataset and regression model.
```

### Appendix Code

```
12 Months Ended December 31
2017       2016       2015
Revenue                                   $56,444    $45,517    $43,604
Cost of sales                             (21,386)   (17,803)   (17,137)
Gross profit                              35,058     27,715     26,467
Distribution expenses                     (5,876)    (4,543)    (4,259)
Sales and marketing expenses              (8,382)    (7,745)    (6,913)
Administrative expenses                   (3,841)    (2,883)    (2,560)
Other operating income/(expenses)         854        732        1,032
Restructuring                             (468)      (323)      (171)
Business and asset disposal               (39)       377        524
Acquisition costs business combinations   (155)      (448)      (55)
Impairment of assets                      —          —          (82)
Judicial settlement                       —          —          (80)
Profit from operations                    17,152     12,882     13,904
Finance cost                              (6,885)    (9,216)    (3,142)
Finance income                            378        652        1,689
Net finance income/(cost)                 (6,507)    (8,564)    (1,453)
Share of result of associates and joint ventures 430        16         10
Profit before tax                         11,076     4,334      12,461
Income tax expense                        (1,920)    (1,613)    (2,594)
Profit from continuing operations         9,155      2,721      9,867
Profit from discontinued operations       28         48         —
Profit of the year                        9,183      2,769      9,867
Profit from continuing operations attributable to:
Equity holders of AB InBev                7,968      1,193      8,273
Non-controlling interest                  1,187      1,528      1,594
Profit of the year attributable to:
Equity holders of AB InBev                7,996      1,241      8,273
Non-controlling interest                  $1,187     $1,528     $1,594
```
Reference: Page 3-4, Exhibit 1: Anheuser-Busch InBev SA/NV Consolidated Income Statement (in Millions of US Dollars) [Excerpt]

```
12 Months Ended
Dec. 31, 2017  Dec. 31, 2016  Dec. 31, 2015
Sales                                 $13,471.5      $6,597.4       $5,127.4
Excise taxes                          (2,468.7)      (1,712.4)      (1,559.9)
Net sales                             11,002.8       4,885.0        3,567.5
Cost of goods sold                    (6,217.2)      (2,987.5)      (2,131.6)
Gross profit                          4,785.6        1,897.5        1,435.9
Marketing, general and administrative expenses (3,032.4)      (1,589.8)      (1,038.3)
Special items, net                    (28.1)         2,522.4        (346.7)
Equity Income in MillerCoors          0              500.9          516.3
Operating income (loss)               1,725.1        3,331.0        567.2
Other income (expense), net
```
Reference: Page 4, Exhibit 2: Molson Coors Brewing Company Consolidated Statement of Operations (in Millions of US Dollars) [Excerpt]

```
Year Ended 31 December
2016       2017
Sales                                 21,944     24,677
Cost of goods sold                    (10,744)   (12,459)
Selling expense                       (5,562)    (5,890)
General and administrative expense    (2,004)    (2,225)
Research and development expense      (333)      (342)
Other income (expense)                (278)      (219)
Recurring operating income            3,022      3,543
Other operating income (expense)      (99)       192
Operating income                      2,923      3,734
Interest income on cash equivalents and 130        151
short-term investments
Interest expense                      (276)      (414)
Cost of net debt                      (146)      (263)
Other financial income                67         137
Other financial expense               (214)      (312)
Income before tax                     2,630      3,296
Income tax expense                    (804)      (842)
Net income from fully consolidated companies 1,826      2,454
Share of profit of associates         1          109
Net income                            1,827      2,563
Net income – Group share              1,720      2,453
Net income - Non-controlling interests 107        110
```
Reference: Page 5, Exhibit 3: Groupe Danone Consolidated Income Statement (in Millions of Euros) [Excerpt]

```
Fiscal Year Ended
Dec. 30, 2017  Dec. 31, 2016  Jan. 02, 2016
Revenues                                $370,075       $324,779       $303,559
Costs of services (exclusive of depreciation and amortization) 258,829        227,380        207,650
Selling, general and administrative expenses           86,537         70,584         72,439
Depreciation and amortization           8,945          7,896          6,552
GNU goodwill impairment                 —              —              4,524
Income from operations                  15,764         18,919         12,394
```
Reference: Page 8, Exhibit 4: CRA International Inc. Consolidated Statements of Operations (Excerpt) (in Thousands of Dollars)

```
The standard states that for performance obligations satisfied over time (e.g.,
where there is a long-term con\fract), revenue is recognized over time by mea-
suring progress toward satisfying the obligation. In this case, the Builder has
incurred 60% of the total expected costs ($420,000/$700,000) and will thus
recognize $600,000 (60% × $1 million) in revenue for the first year.
```
Reference: Page 12, Example 5, Part 2 (ref. Example 8)

```
The standard addresses so-called “variable consideration" as part of deter-
mining the transaction price. A company is only allowed to recognize variable
consideration if it can conclude that it will not have to reverse the cumulative
revenue in the future. In this case, Builder Co. does not recognize any of the
bonus in year one because it cannot reach the non-reversible conclusion given
its limited experience with similar con\fracts and potential delays from factors
outside its control.
```
Reference: Page 12, Example 5, Part 3 (ref. Example 8)

```
Builder's total
revenue on the transaction (transaction price) is now $1.35 million ($1 million
original plus the $150,000 new consideration plus $200,000 for the completion
bonus). Builder Co's progress toward completion is now 51.2% ($420,000 costs
incurred divided by total expected costs of $820,000). Based on the changes
in the con\fract, the amount of additional revenue to be recognized is $91,200,
calculated as (51.2% × $1.35 million) minus the $600,000 already recognized. The
additional $91,200 of revenue would be recognized as a “cumulative catch-up
adjustment" on the date of the con\fract modification.
```
Reference: Page 12-13, Example 5, Part 4 (ref. Example 8)

```
In this example, the Company is an Agent because it isn't primarily responsible
for fulfilling the con\fract, doesn't take any inventory risk or credit risk, doesn't
have discretion in setting the price, and receives compensation in the form of a
commission. Because the Company is acting as an Agent, it should report only
the amount of commission as its revenue.
```
Reference: Page 13, Example 5, Part 5 (ref. Example 45)

```
Inventory Purchases
First quarter     2,000 units at $40 per unit
Second quarter    1,500 units at $41 per unit
Third quarter     2,200 units at $43 per unit
Fourth quarter    1,900 units at $45 per unit
Total             7,600 units at a total cost of $321,600
```
Reference: Page 15, Example 1

```
The revenue for 2018 would be $280,000 (5,600 units × $50 per unit). Initial-
ly, the total cost of the goods purchased would be recorded as inventory (an
asset) in the amount of $321,600. During 2018, the cost of the 5,600 units
sold would be expensed (matched against the revenue) while the cost of the
2,000 remaining unsold units would remain in inventory as follows:

Cost of Goods Sold
From the first quarter     2,000 units at $40 per unit =  $80,000
From the second quarter    1,500 units at $41 per unit =  $61,500
From the third quarter     2,100 units at $43 per unit =  $90,300
Total cost of goods sold                              $231,800

Cost of Goods Remaining in Inventory
From the third quarter     100 units at $43 per unit =   $4,300
From the fourth quarter    1,900 units at $45 per unit =  $85,500
Total remaining (or ending) inventory cost            $89,800
```
Reference: Page 15, Example 1 Solution

```
To confirm that total costs are accounted for: $231,800 + $89,800 =
$321,600. The cost of the goods sold would be expensed against the revenue
of $280,000 as follows:

Revenue              $280,000
Cost of goods sold    231,800
Gross profit          $48,200
```
Reference: Page 16, Example 1 (continued solution)

```
For KDL, the weighted average cost per unit would be
$321,600/7,600 units = $42.3158 per unit
Cost of goods sold using the weighted average cost method would be
5,600 units at $42.3158 = $236,968
Ending inventory using the weighted average cost method would be
2,000 units at $42.3158 = $84,632
...
Ending inventory 2,000 units at $40 per unit = $80,000
...
Total costs of $321,600 less $80,000 remaining in ending inventory = $241,600
Alternatively, the cost of the last 5,600 units purchased is allocated to cost of
goods sold under LIFO:
1,900 units at $45 per unit + 2,200 units at $43 per unit + 1,500 units at $41 per
unit
= $241,600
```
Reference: Page 16-17, Example 2

```
Method            Description                                             Cost of Goods Sold When Prices Are Rising, Relative to Other Two Methods    Ending Inventory When Prices Are Rising, Relative to Other Two Methods
FIFO (first in, first out) Costs of the earliest items pur-                     Lowest                                                       Highest
                          chased flow to cost of goods sold
                          first
LIFO (last in, first out) Costs of the most recent items                      Highest*                                                     Lowest*
                          purchased flow to cost of goods
                          sold first
Weighted average cost     Averages total costs over total                     Middle                                                       Middle
                          units available
```
Reference: Page 17-18, Exhibit 6: Summary Table on Inventory Costing Methods

```
Using the straight-line method of depreciation, annual depreciation expense is
calculated as:
Cost - Residual value
Estimated useful life
Assume the cost of an asset is $10,000. If, for example, the residual value of
the asset is estimated to be $0 and its useful life is estimated to be 5 years, the
annual depreciation expense under the straight-line method would be ($10,000
$0)/5 years = $2,000. In contrast, holding the estimated useful life of the asset
constant at 5 years but increasing the estimated residual value of the asset to
$4,000 would result in annual depreciation expense of only $1,200 [calculated as
($10,000 – $4,000)/5 years]. Alternatively, holding the estimated residual value
at $0 but increasing the estimated useful life of the asset to 10 years would result
in annual depreciation expense of only $1,000 [calculated as ($10,000 – $0)/10
years].
```
Reference: Page 20, Example 3

```
Estimated
Useful Life
(Years)                 Estimated Residual Value
0       1,000   2,000   3,000   4,000   5,000
2       5,000   4,500   4,000   3,500   3,000   2,500
4       2,500   2,250   2,000   1,750   1,500   1,250
5       2,000   1,800   1,600   1,400   1,200   1,000
8       1,250   1,125   1,000   875     750     625
10      1,000   900     800     700     600     500
```
Reference: Page 20-21, Exhibit 7: Annual Depreciation Expense (in Dollars)

```
At the beginning of the first year, the net book value is $11,000. Depreciation
expense for the first full year of use of the asset would be 40 percent of $11,000,
or $4,400. Under this method, the residual value, if any, is generally not used in
the computation of the depreciation each period (the 40 percent is applied to
$11,000 rather than to $11,000 minus residual value). However, the company
will stop taking depreciation when the salvage value is reached.

At the beginning of Year 2, the net book value is measured as
Asset cost                 $11,000
Less: Accumulated depreciation (4,400)
Net book value             $6,600

For the second full year, depreciation expense would be $6,600 × 40 percent, or
$2,640. At the end of the second year (i.e., beginning of the third year), a total
of $7,040 ($4,400 + $2,640) of depreciation would have been recorded. So, the
remaining net book value at the beginning of the third year would be
Asset cost                 $11,000
Less: Accumulated depreciation (7,040)
Net book value             $3,960

For the third full year, depreciation would be $3,960 × 40 percent, or $1,584.
At the end of the third year, a total of $8,624 ($4,400 + $2,640 + $1,584) of
depreciation would have been recorded. So, the remaining net book value at
the beginning of the fourth year would be
Asset cost                 $11,000
Less: Accumulated depreciation (8,624)
Net book value             $2,376

For the fourth full year, depreciation would be $2,376 × 40 percent, or $950. At
the end of the fourth year, a total of $9,574 ($4,400 + $2,640 + $1,584 + $950)
of depreciation would have been recorded. So, the remaining net book value at
the beginning of the fifth year would be
Asset cost                 $11,000
Less: Accumulated depreciation (9,574)
Net book value             $1,426

For the fifth year, if deprecation were determined as in previous years, it would
amount to $570 ($1,426 × 40 percent). However, this would result in a remain-
ing net book value of the asset below its estimated residual value of $1,000. So,
instead, only $426 would be depreciated, leaving a $1,000 net book value at the
end of the fifth year.
Asset cost                 $11,000
Less: Accumulated depreciation (10,000)
Net book value             $1,000
```
Reference: Page 21-22, Example 4: An Illustration of Diminishing Balance Depreciation

```
(in € millions)                           Related income
                                           (expense)
Capital gain on disposal of Stonyfield           628
Compensation received following the decision of the Singapore arbi- 105
tration court in the Fonterra case
Territorial risks, mainly in certain countries in the ALMA region  (148)
Costs associated with the integration of WhiteWave                 (118)
Impairment of several intangible assets in Waters and Specialized  (115)
Nutrition Reporting entities
Remainder of table omitted
```
Reference: Page 26, Exhibit 8: Highlighting Infrequent Nature of Items—Excerpt from Groupe Danone footnotes to its 2017 financial statements

```
(In $ millions, except per share
amounts)               As Previously  New Revenue  As Restated
                       Reported       Standard
                                      Adjustment
Income Statements
Year Ended June 30, 2017
Revenue                89,950         6,621        96,571
Provision for income taxes 1,945          2,467        4,412
Net income             21,204         4,285        25,489
Diluted earnings per share 2.71           0.54         3.25

Year Ended June 30, 2016
Revenue                85,320         5,834        91,154
Provision for income taxes 2,953          2,147        5,100
Net income             16,798         3,741        20,539
Diluted earnings per share 2.1            0.46         2.56
```
Reference: Page 27, Example 5: Microsoft Corporation Excerpt from Footnotes to the Financial Statements

```
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
```
Reference: Page 28, Example 5 Solution (continued)

```
12 Months Ended December 31
2017       2016       2015
Basic earnings per share                  $4.06      $0.72      $5.05
Diluted earnings per share                3.98       0.71       4.96
Basic earnings per share from continuing  4.04       0.69       5.05
operations
Diluted earnings per share from continuing $3.96      $0.68      $4.96
operations
```
Reference: Page 31, Exhibit 10: AB InBev's Earnings Per Share

```
Basic EPS = Net income - Preferred dividends
            -----------------------------------
            Weighted average number of shares outstanding
```
Reference: Page 31, Basic EPS Formula

```
Solution:
Shopalot's basic EPS is $1.30 ($1,950,000 divided by 1,500,000 shares).
```
Reference: Page 32, Example 6 Solution

```
Shares outstanding on 1 January 2018      1,000,000
Shares issued on 1 April 2018             200,000
Shares repurchased (treasury shares) on 1 October 2018 (100,000)
Shares outstanding on 31 December 2018    1,100,000

Solution to 1:
The weighted average number of shares outstanding is determined by the
length of time each quantity of shares was outstanding:
1,000,000 x (3 months/12 months) =        250,000
1,200,000 × (6 months/12 months) =        600,000
1,100,000 x (3 months/12 months) =        275,000
Weighted average number of shares outstanding 1,125,000
```
Reference: Page 32, Example 7 and Solution to 1

```
Solution to 2:
Basic EPS = (Net income – Preferred dividends)/Weighted average number
of shares = ($2,500,000 – $200,000)/1,125,000 = $2.04
```
Reference: Page 33, Example 7 Solution to 2

```
Solution:
For EPS calculation purposes, a stock split is treated as if it occurred at the
beginning of the period. The weighted average number of shares would,
therefore, be 2,250,000, and the basic EPS would be $1.02 [= ($2,500,000 –
$200,000)/2,250,000].
```
Reference: Page 33, Example 8 Solution

```
Diluted EPS = (Net income)
              -----------------------------------
              (Weighted average number of shares
              outstanding + New common shares that
              would have been issued at conversion)
```
Reference: Page 34, Diluted EPS Formula (2)

```
Solution:
If the 20,000 shares of convertible preferred had each converted into 5
shares of the company's common stock, the company would have had an
additional 100,000 shares of common stock (5 shares of common for each of
the 20,000 shares of preferred). If the conversion had taken place, the com-
pany would not have paid preferred dividends of $200,000 ($10 per share for
each of the 20,000 shares of preferred). As shown in Exhibit 11, the compa-
ny's basic EPS was $3.10 and its diluted EPS was $2.92.
```
Reference: Page 34, Example 9 Solution

```
                          Basic EPS   Diluted EPS Using
                                      If-Converted Method
Net income                $1,750,000  $1,750,000
Preferred dividend        -200,000    0
Numerator                 $1,550,000  $1,750,000
Weighted average number of shares 500,000   500,000
outstanding
Additional shares issued if preferred 0         100,000
Denominator               500,000     600,000
EPS                       $3.10       $2.92
```
Reference: Page 34-35, Exhibit 11: Calculation of Diluted EPS for Bright-Warm Utility Company Using the If-Converted Method: Case of Preferred Stock

```
Diluted EPS = (Net income + After-tax interest on
              convertible debt - Preferred dividends)
              -------------------------------------
              (Weighted average number of shares
              outstanding + Additional common
              shares that would have been
              issued at conversion)
```
Reference: Page 35, Diluted EPS Formula (3)

```
Solution:
If the debt securities had been converted, the debt securities would no
longer be outstanding and instead, an additional 10,000 shares of common
stock would be outstanding. Also, if the debt securities had been converted,
the company would not have paid interest of $3,000 on the convertible debt,
so net income available to common shareholders would have increased by
$2,100 [= $3,000(1 – 0.30)] on an after-tax basis. Exhibit 12 illustrates the
calculation of diluted EPS using the if-converted method for convertible
debt.
```
Reference: Page 35, Example 10 Solution

```
                          Basic EPS   Diluted EPS Using
                                      If-Converted Method
Net income                $750,000    $750,000
After-tax cost of interest  —         2,100
Numerator                 $750,000    $752,100
Weighted average number of shares 690,000   690,000
outstanding
If converted              0           10,000
Denominator               690,000     700,000
EPS                       $1.09       $1.07
```
Reference: Page 36, Exhibit 12: Calculation of Diluted EPS for Oppnox Company Using the If-Converted Method: Case of a Convertible Bond

```
Diluted EPS = (Net income - Preferred dividends)
              ---------------------------------------
              [Weighted average number of shares
              outstanding + (New shares that would
              have been issued at option exercise-
              Shares that could have been purchased
              with cash received upon exercise) x
              (Proportion of year during which the
              financial instruments were outstanding)]
```
Reference: Page 37, Diluted EPS Formula (4)

```
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
in Exhibit 13, the company's basic EPS was $2.88 and the diluted EPS was
$2.84.
```
Reference: Page 37, Example 11 Solution

```
                          Basic EPS   Diluted EPS Using
                                      Treasury Stock Method
Net income                $2,300,000  $2,300,000
Numerator                 $2,300,000  $2,300,000
Weighted average number of shares 800,000   800,000
outstanding
If converted              0           10,909
Denominator               800,000     810,909
EPS                       $2.88       $2.84
```
Reference: Page 38, Exhibit 13: Calculation of Diluted EPS for Hihotech Company Using the Treasury Stock Method: Case of Stock Options

```
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
```
Reference: Page 38, Example 12 Solution

```
Solution:
If the 20,000 shares of convertible preferred had each converted into 3
shares of the company's common stock, the company would have had an ad-
ditional 60,000 shares of common stock (3 shares of common for each of the
20,000 shares of preferred). If the conversion had taken place, the company
would not have paid preferred dividends of $200,000 ($10 per share for each
of the 20,000 shares of preferred). The effect of using the if-converted meth-
od would be EPS of $3.13, as shown in Exhibit 14. Because this is greater
than the company's basic EPS of $3.10, the securities are said to be antidilu-
tive and the effect of their conversion would not be included in diluted EPS.
Diluted EPS would be the same as basic EPS (i.e., $3.10).
```
Reference: Page 39, Example 13 Solution

```
                          Basic EPS   Diluted EPS Using
                                      If-Converted Method
Net income                $1,750,000  $1,750,000
Preferred dividend        -200,000    0
Numerator                 $1,550,000  $1,750,000
Weighted average number of shares 500,000   500,000
outstanding
If converted              0           60,000
Denominator               500,000     560,000
EPS                       $3.10       $3.13
                                      -Exceeds basic EPS; security
                                      is antidilutive and, therefore,
                                      not included. Reported
                                      diluted EPS= $3.10.
```
Reference: Page 39-40, Exhibit 14: Calculation for an Antidilutive Security

```
Panel A: Income Statements for Companies A, B, and C ($)
                        A             B             C
Sales                   $10,000,000   $10,000,000   $2,000,000
Cost of sales           3,000,000     7,500,000     600,000
Gross profit            7,000,000     2,500,000     1,400,000
Selling, general, and administrative expenses 1,000,000     1,000,000     200,000
Research and development 2,000,000   —           400,000
Advertising             2,000,000   —           400,000
Operating profit        2,000,000   1,500,000   400,000

Panel B: Common-Size Income Statements for Companies A, B, and C (%)
                        A     B     C
Sales                   100%  100%  100%
Cost of sales           30    75    30
Gross profit            70    25    70
Selling, general, and administrative expenses 10    10    10
Research and development 20    0     20
Advertising             20    0     20
Operating profit        20    15    20
```
Reference: Page 41-42, Exhibit 15: Income Statements for Companies A, B, and C

```
                        Energy Materials Industrials Consumer Consumer Health Care
                                                   Discretionary Staples
Number of observations  34     27        69          81        34        59
Gross Margin            37.7%  33.0%     36.8%       37.6%     43.4%     59.0%
Operating Margin        6.4%   14.9%     13.5%       11.0%     17.2%     17.4%
Net Profit Margin       4.9%   9.9%      8.8%        6.0%      10.9%     7.2%
```
Reference: Page 42, Exhibit 16: Median Common-Size Income Statement Statistics for the S&P 500 Classified by S&P/MSCI GICS Sector Data for 2017

```
                        Financials Information Telecommunication Utilities Real Estate
                                 Technology   Services
Number of observations  63         64           4                29        29
Gross Margin            40.5%      62.4%        56.4%            34.3%     39.8%
Operating Margin        36.5%      21.1%        15.4%            21.7%     30.1%
Net Profit Margin       18.5%      11.3%        13.1%            10.1%     21.3%
```
Reference: Page 43, Financial Ratios by Sector (part of Exhibit 16)

```
Net profit margin = Net income
                    -----------
                    Revenue
```
Reference: Page 43, Net Profit Margin Formula

```
Gross profit margin = Gross profit
                      ------------
                      Revenue
```
Reference: Page 43, Gross Profit Margin Formula

```
12 Months Ended December 31
                         2017        2016        2015
                         $       %     $       %     $       %
Revenue                  56,444  100.0   45,517  100.0   43,604  100.0
Cost of sales            (21,386) (37.9)  (17,803) (39.1)  (17,137) (39.3)
Gross profit             35,058  62.1    27,715  60.9    26,467  60.7
Distribution expenses    (5,876) (10.4)  (4,543) (10.0)  (4,259) (9.8)
Sales and marketing expenses (8,382) (14.9)  (7,745) (17.0)  (6,913) (15.9)
Administrative expenses  (3,841) (6.8)   (2,883) (6.3)   (2,560) (5.9)
Portions omitted
Profit from operations   17,152  30.4    12,882  28.3    13,904  31.9
Finance cost             (6,885) (12.2)  (9,382) (20.6)  (3,142) (7.2)
Finance income           378     0.7     818     1.8     1,689   3.9
Net finance income/(cost) (6,507) (11.5)  (8,564) (18.8)  (1,453) (3.3)
Share of result of associates and joint ventures 430     0.8     16      0.0     10      0.0
Profit before tax        11,076  19.6    4,334   9.5     12,461  28.6
Income tax expense       (1,920) (3.4)   (1,613) (3.5)   (2,594) (5.9)
Profit from continuing operations 9,155   16.2    2,721   6.0     9,867   22.6
Profit from discontinued operations 28      0.0     48      0.1     —       —
Profit of the year       9,183   16.3    2,769   6.1     9,867   22.6
```
Reference: Page 44, Exhibit 17: AB InBev's Margins: Abbreviated Common-Size Income Statement

```
Solution to 1:
C is correct. If the company's actual ending shareholders' equity is €227
million, then €10 million [€227− (€200 + €20 – €3)] has bypassed the net
income calculation by being classified as other comprehensive income.

Solution to 2:
B is correct. Answers A and C are not correct because they do not specify
whether such income is reported as part of net income and shown in the
income statement.
```
Reference: Page 47, Example 14 Solutions

```
Company A    Company B
Price                                $35          $30
EPS                                  $1.60        $0.90
P/E ratio                            21.9x        33.3x
Other comprehensive income (loss) $ million ($16.272)    $(1.757)
Shares (millions)                    22.6         25.1

Solution:
As shown in the following table, part of the explanation for Company A's
lower P/E ratio may be that its significant losses—accounted for as other
comprehensive income (OCI)—are not included in the P/E ratio.

Company A Company B
Price                                $35          $30
EPS                                  $1.60        $0.90
OCI (loss) $ million                 ($16.272)    $(1.757)
Shares (millions)                    22.6         25.1
OCI (loss) per share                 $(0.72)      $(0.07)
Comprehensive EPS = EPS + OCI per share $ 0.88       $0.83
Price/Comprehensive EPS ratio        39.8x        36.1x
```
Reference: Page 48, Example 15 Problem and Solution

```
Revenue                   $4,000,000
Cost of goods sold        $3,000,000
Other operating expenses  $500,000
Interest expense          $100,000
Tax expense               $120,000
```
Reference: Page 51, Problem 3

```
Revenue           $1,000,000
Returns of goods sold $100,000
Cash collected    $800,000
Cost of goods sold $700,000
```
Reference: Page 51, Problem 5

```
Total sales price of items sold during 2009 on consignment was €2,000,000.
Total commissions retained by Apex during 2009 for these items was €500,000.
```
Reference: Page 52, Problem 6

```
Purchased 10,000 units of a toy at a cost of £10 per unit in October.
Purchased 5,000 additional units in November at a cost of £11 per unit.
Sold 12,000 units at a price of £15 per unit.
```
Reference: Page 52, Problem 8 & 9

```
Cost of $600,000.
Estimated useful life of 10 years and estimated residual value of $50,000.
```
Reference: Page 53, Problem 11 & 12

```
Net income of $1,000,000.
1 January 2009: 1,000,000 shares outstanding.
1 July 2009: issued 100,000 new shares for $20 per share.
Paid $200,000 in dividends to common shareholders.
```
Reference: Page 53, Problem 16

```
Net income of $200 million.
Weighted average of 50,000,000 common shares outstanding.
2,000,000 convertible preferred shares outstanding that paid an annual dividend of $5.
Each preferred share is convertible into two shares of the common stock.
```
Reference: Page 54, Problem 18

```
Net income of $12 million.
Weighted average of 2,000,000 common shares outstanding.
Paid $800,000 in preferred dividends.
100,000 options outstanding with an average exercise price of $20.
CWC's market price over the year averaged $25 per share.
```
Reference: Page 54, Problem 19

```
Common shares outstanding, 1 January   2,020,000
Common shares issued as stock dividend, 1 June 380,000
Warrants outstanding, 1 January        500,000
Net income                             $3,350,000
Preferred stock dividends paid         $430,000
Common stock dividends paid            $240,000
```
Reference: Page 54, Problem 20

```
1,000,000 average shares outstanding during all of 2009.
10,000 options outstanding with exercise prices of $10 each.
Average stock price of CSI during 2009 was $15.
```
Reference: Page 54, Problem 21

```
$ millions
Beginning shareholders' equity         475
Ending shareholders' equity            493
Unrealized gain on available-for-sale securities 5
Unrealized loss on derivatives accounted for as hedges -3
Foreign currency translation gain on consolidation 2
Dividends paid                         1
Net income                             15
```
Reference: Page 55, Problem 24

```
3. C is correct. Gross margin is revenue minus cost of goods sold. Answer A rep-
resents net income and B represents operating income.
```
Reference: Page 56, Solution 3

```
5. B is correct. Net revenue is revenue for goods sold during the period less any
returns and allowances, or $1,000,000 minus $100,000 = $900,000.
```
Reference: Page 56, Solution 5

```
6. A is correct. Apex is not the owner of the goods and should only report its net
commission as revenue.
```
Reference: Page 56, Solution 6

```
8. B is correct. Under the first in, first out (FIFO) method, the first 10,000 units sold
came from the October purchases at £10, and the next 2,000 units sold came
from the November purchases at £11.
```
Reference: Page 56, Solution 8

```
9. C is correct. Under the weighted average cost method:
October purchases       10,000 units   $100,000
November purchases      5,000 units    $55,000
Total                   15,000 units   $155,000
$155,000/15,000 units = $10.3333
$10.3333 × 12,000 units = $124,000
```
Reference: Page 56, Solution 9

```
11. A is correct. Straight-line depreciation would be ($600,000 – $50,000)/10, or
$55,000.
```
Reference: Page 56, Solution 11

```
12. C is correct. Double-declining balance depreciation would be $600,000 × 20 per-
cent (twice the straight-line rate). The residual value is not sub\fracted from the
initial book value to calculate depreciation. However, the book value (carrying
amount) of the asset will not be reduced below the estimated residual value.
```
Reference: Page 56, Solution 12

```
16. C is correct. The weighted average number of shares outstanding for 2009 is
1,050,000. Basic earnings per share would be $1,000,000 divided by 1,050,000, or
$0.95.
```
Reference: Page 57, Solution 16

```
18. C is correct.
Diluted EPS = (Net income)/(Weighted average number of shares outstanding +
New common shares that would have been issued at conversion)
= $200,000,000/[50,000,000 + (2,000,000 × 2)]
= $3.70
The diluted EPS assumes that the preferred dividend is not paid and that the
shares are converted at the beginning of the period.
```
Reference: Page 57, Solution 18

```
19. B is correct. The formula to calculate diluted EPS is as follows:
Diluted EPS = (Net income – Preferred dividends)/[Weighted average number of
shares outstanding + (New shares that would have been issued at option exer-
cise – Shares that could have been purchased with cash received upon exercise) ×
(Proportion of year during which the financial instruments were outstanding)].
The underlying assumption is that outstanding options are exercised, and then
the proceeds from the issuance of new shares are used to repurchase shares
already outstanding:
Proceeds from option exercise = 100,000 × $20 = $2,000,000
Shares repurchased = $2,000,000/$25 = 80,000
The net increase in shares outstanding is thus 100,000 – 80,000 = 20,000. There-
fore, the diluted EPS for CWC = ($12,000,000 – $800,000)/2,020,000 = $5.54.
```
Reference: Page 57, Solution 19

```
21. A is correct. With stock options, the treasury stock method must be used. Under
that method, the company would receive $100,000 (10,000 × $10) and would re-
purchase 6,667 shares ($100,000/$15). The shares for the denominator would be:
Shares outstanding        1,000,000
Options exercises          10,000
Treasury shares purchased  (6,667)
Denominator              1,003,333
```
Reference: Page 58, Solution 21

```
24. C is correct. Comprehensive income includes both net income and other com-
prehensive income.
Other comprehensive income = Unrealized gain on available-for-sale securities –
Unrealized loss on derivatives accounted for as hedges + Foreign currency transla-
tion gain on consolidation
= $5 million – $3 million + $2 million
= $4 million
Alternatively,
Comprehensive income – Net income = Other comprehensive income
Comprehensive income = (Ending shareholders equity – Beginning shareholders
equity) + Dividends
= ($493 million – $475 million) + $1 million
= $18 million + $1 million = $19 million
Net income is $15 million so other comprehensive income is $4 million.
```
Reference: Page 58, Solution 24