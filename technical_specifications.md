
# Technical Specifications: Multiple Linear Regression Assumptions Analyzer

## Overview

The "Multiple Linear Regression Assumptions Analyzer" is a Streamlit application designed to provide an int\fractive platform for understanding and visualizing the core assumptions underlying multiple linear regression models. By utilizing readily available datasets and int\fractive plotting components, users can explore relationships between variables, examine model residuals, and identify potential violations of critical regression assumptions such as linearity, homoscedasticity, normality of residuals, and absence of multicollinearity. The application aims to enhance learning outcomes related to interpreting residual plots and understanding the \fractical implications of these assumptions in statistical modeling.

## Step-by-Step Development Process

The development of the Streamlit application will follow these logical steps:

1.  **Project Setup and Environment**:
    *   Create a new Python virtual environment.
    *   Install necessary libraries: `streamlit`, `pandas`, `numpy`, `statsmodels`, `plotly`, `scikit-learn`.
    *   Create the main application file, `app.py`.

2.  **Data Loading and Preparation**:
    *   Implement a function to load a readily available dataset (e.g., `load_diabetes` from `sklearn.datasets`).
    *   Structure the data into a Pandas DataFrame, clearly defining the dependent variable (target) and independent variables (features).
    *   Provide options in the Streamlit sidebar for users to select independent variables from the loaded dataset.

3.  **Multiple Linear Regression Model Implementation**:
    *   Use the `statsmodels.api.OLS` module to build and fit the multiple linear regression model based on the user-selected variables.
    *   Ex\fract model summary statistics, predicted values, and residuals after fitting the model.

4.  **Core Concepts and Explanatory Content Integration**:
    *   Embed comprehensive Markdown explanations for each assumption of multiple linear regression.
    *   Integrate the mathematical formula for multiple linear regression following the specified LaTeX template.
    *   Provide \fractical examples and real-world context for each assumption and its potential violations.

5.  **Int\fractive Visualization Development**:
    *   **Scatterplot Matrix**: Generate a scatterplot matrix using `plotly.express.scatter_matrix` to visualize relationships between all selected independent variables and the dependent variable.
    *   **Residuals vs. Predicted Value Plot**: Create a scatter plot using `plotly.graph_objects.Scatter` to visualize the residuals against the predicted values of the dependent variable. Add a horizontal line at $y=0$ for reference.
    *   **Regression Residuals vs. Factors Plots**: For each independent variable, generate individual scatter plots of residuals against that factor.
    *   **Normality Plots**:
        *   A Histogram of residuals to visually assess their distribution.
        *   A Q-Q plot (Quantile-Quantile plot) of residuals against a theoretical normal distribution using `statsmodels.api.qqplot`.

6.  **Streamlit User Interface (UI) Construction**:
    *   Design the sidebar for user inputs (dataset selection, independent variable checkboxes).
    *   Arrange the main content area to display:
        *   An introductory overview.
        *   Explanations of multiple linear regression and its assumptions.
        *   The int\fractive plots, dynamically updated based on user selections.
        *   Model summary output from `statsmodels`.

7.  **Int\fractivity and Dynamic Updates**:
    *   Ensure that changing variable selections in the sidebar dynamically re-runs the regression model and updates all associated plots and model summary.
    *   Implement clear loading indicators or messages where computations might take a moment.

## Core Concepts and Mathematical Foundations

### Multiple Linear Regression Model
Multiple linear regression is a statistical technique used to model the relationship between a dependent variable and two or more independent variables by fitting a linear equation to observed data. It aims to predict the value of a dependent variable based on the values of several independent variables.

The multiple linear regression model is represented as:
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p + \epsilon
$$
Where:
- $Y$: The dependent variable (the variable being predicted or explained)
- $\beta_0$: The Y-intercept, representing the expected value of $Y$ when all independent variables ($X_i$) are zero
- $\beta_i$: The regression coefficient for the $i$-th independent variable, $X_i$, representing the change in $Y$ for a one-unit increase in $X_i$, holding all other independent variables constant
- $X_i$: The $i$-th independent variable (predictor or explanatory variable)
- $p$: The number of independent variables
- $\epsilon$: The error term (or residual), representing the portion of $Y$ that is not explained by the linear relationship with the $X$ variables. It accounts for all other factors influencing $Y$ that are not included in the model.

This formula establishes a linear relationship between the dependent variable and a set of independent variables, seeking to minimize the sum of squared differences between observed and predicted values. It is widely used for prediction and to understand the strength and direction of relationships between variables in various fields, from economics to healthcare.

### Assumptions Underlying Multiple Linear Regression

For the coefficients in a multiple linear regression model to be reliable and for hypothesis tests (like t-tests and F-tests) to be valid, several assumptions about the data and the error term must be met. The application will help users visualize potential violations of these.

#### Linearity
The relationship between the independent variables ($X_i$) and the dependent variable ($Y$) is calculated using:
$$
Y = \beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p + \epsilon
$$
Where:
- $Y$: Dependent variable
- $\beta_i$: Regression coefficients
- $X_i$: Independent variables
- $\epsilon$: Error term

This formula assumes that the mean of the dependent variable is a linear combination of the independent variables. If the relationship is non-linear, the model will not accurately capture the true relationship, leading to biased coefficients and inaccurate predictions.
**\fractical Application**: This assumption is crucial because linear regression models, by their nature, are designed to fit straight lines or hyperplanes. Violations often manifest as curved patterns in residual plots (e.g., "U" shape or inverted "U" shape). The Scatterplot Matrix helps visually inspect initial linear relationships, while the Residuals vs. Predicted Value plot is critical for detecting non-linearity; a random scatter around $0$ suggests linearity, while a discernible pattern indicates a violation.

#### Independence of Residuals
The error terms ($\epsilon$) for different observations are assumed to be independent. Mathematically, for any two distinct observations $i$ and $j$:
$$
Cov(\epsilon_i, \epsilon_j) = 0 \quad \text{for } i \neq j
$$
Where:
- $Cov(\epsilon_i, \epsilon_j)$: Covariance between the error terms of observation $i$ and observation $j$

This equation states that the errors associated with one observation do not influence the errors of another observation. Violations, known as autocorrelation (or serial correlation), commonly occur in time-series data where errors from one period are correlated with errors from previous periods.
**\fractical Application**: Independence of residuals is vital because dependent errors can lead to underestimated standard errors, inflated t-statistics, and misleading p-values, making coefficients appear more significant than they are. The Residuals vs. Factors plots are useful for detecting patterns that might suggest dependencies, especially if the independent variables are time-ordered. A common visual cue for autocorrelation is a "snake-like" or cyclical pattern in residual plots when the independent variable is time or an ordered sequence.

#### Homoscedasticity
The variance of the error terms ($\epsilon$) is constant across all levels of the independent variables. This is expressed as:
$$
Var(\epsilon) = \sigma^2
$$
Where:
- $Var(\epsilon)$: Variance of the error term
- $\sigma^2$: A constant variance

This formula implies that the spread of residuals should be roughly the same across the range of predicted values and independent variables. If the variance of the errors is not constant, it is called heteroscedasticity.
**\fractical Application**: Homoscedasticity ensures that the model's predictions are equally precise across the range of data. Heteroscedasticity leads to inefficient (though still unbiased) coefficient estimates and incorrect standard errors, making confidence intervals and hypothesis tests unreliable. The **Residuals vs. Predicted Value** plot is the primary tool for checking this assumption. A fan-like shape (widening or narrowing spread of residuals) or a cone shape indicates heteroscedasticity, whereas a uniform band of residuals around zero suggests homoscedasticity.

#### Normality of Residuals
The error terms ($\epsilon$) are normally distributed. This is formally stated as:
$$
\epsilon \sim N(0, \sigma^2)
$$
Where:
- $\epsilon$: Error term
- $N(0, \sigma^2)$: Normal distribution with a mean of $0$ and a constant variance $\sigma^2$

This statement means that the errors should follow a bell-shaped curve, centered at zero. While violations of this assumption primarily impact the reliability of hypothesis tests and confidence intervals for small sample sizes, larger sample sizes often make the central limit theorem applicable, mitigating this issue.
**\fractical Application**: Normality of residuals is important for the validity of statistical inference (e.g., calculating p-values and confidence intervals). Departures from normality, such as skewness or heavy tails, can be identified using a **Histogram of Residuals** (which should approximate a bell curve) and a **Q-Q plot**. In a Q-Q plot, residuals should roughly follow a straight line if they are normally distributed; deviations from this line indicate non-normality.

#### No Multicollinearity
The independent variables ($X_i$) are not highly correlated with each other. This implies that none of the independent variables can be perfectly predicted from a linear combination of the others. Mathematically, it means the design matrix $X$ has full column rank.
$$
Rank(X) = p+1
$$
Where:
- $X$: The design matrix of independent variables (including a column of ones for the intercept)
- $p$: The number of independent variables

This condition states that the independent variables must not be perfectly linearly related. High correlation between independent variables, known as multicollinearity, makes it difficult to ascertain the individual effect of each independent variable on the dependent variable.
**\fractical Application**: Multicollinearity makes it challenging to interpret individual regression coefficients because their estimates become unstable and highly sensitive to small changes in the data. While not directly visualized by residual plots, the **Scatterplot Matrix** is crucial for initial detection by showing strong correlations between independent variables. Variance Inflation Factor (VIF) scores (often part of `statsmodels` output or calculated separately) are a quantitative measure to detect severe multicollinearity.

## Required Libraries and Dependencies

The application relies on the following Python libraries for data handling, statistical modeling, and int\fractive visualization:

*   **`streamlit`**: Version `1.33.0` (or compatible latest stable version).
    *   **Role**: Used for building the int\fractive web application interface.
    *   **Specific Functions/Modules**: `st.sidebar`, `st.columns`, `st.header`, `st.subheader`, `st.write`, `st.dataframe`, `st.plotly_chart`, `st.checkbox`, `st.markdown`, `st.expander`.
    *   **Import Example**: `import streamlit as st`

*   **`pandas`**: Version `2.2.2` (or compatible latest stable version).
    *   **Role**: Fundamental for data manipulation and analysis, primarily for handling datasets as DataFrames.
    *   **Specific Functions/Modules**: `pd.DataFrame`, `pd.read_csv` (if using local files, though `sklearn.datasets` is preferred here).
    *   **Import Example**: `import pandas as pd`

*   **`numpy`**: Version `1.26.4` (or compatible latest stable version).
    *   **Role**: Provides numerical operations, especially useful for array manipulations and mathematical functions.
    *   **Specific Functions/Modules**: `np.array`, `np.mean`.
    *   **Import Example**: `import numpy as np`

*   **`statsmodels`**: Version `0.14.2` (or compatible latest stable version).
    *   **Role**: Crucial for statistical modeling, particularly for Ordinary Least Squares (OLS) regression and statistical tests/diagnostics.
    *   **Specific Functions/Modules**: `statsmodels.api.OLS` for regression, `statsmodels.api.add_constant` to add an intercept term, `statsmodels.graphics.gofplots.qqplot` for Q-Q plots.
    *   **Import Example**: `import statsmodels.api as sm`

*   **`plotly`**: Version `5.21.0` (or compatible latest stable version).
    *   **Role**: Used for generating int\fractive and publication-quality visualizations, essential for the scatterplot matrix and residual plots.
    *   **Specific Functions/Modules**: `plotly.express.scatter_matrix` for high-level multi-panel plots, `plotly.graph_objects.Figure` and `plotly.graph_objects.Scatter` for custom scatter plots, `plotly.subplots.make_subplots` for arranging multiple plots.
    *   **Import Example**: `import plotly.express as px`, `import plotly.graph_objects as go`

*   **`scikit-learn`**: Version `1.4.2` (or compatible latest stable version).
    *   **Role**: Provides access to various datasets and potentially some preprocessing utilities if needed.
    *   **Specific Functions/Modules**: `sklearn.datasets.load_diabetes` for loading a sample dataset.
    *   **Import Example**: `from sklearn.datasets import load_diabetes`

## Implementation Details

### Application Structure (`app.py`)

```python
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_diabetes # Example dataset

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Multiple Linear Regression Assumptions Analyzer")

# --- Data Loading ---
@st.cache_data # Cache data to avoid reloading on every rerun
def get_diabetes_data():
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame
    df.columns = [col.replace(' ', '_') for col in df.columns] # Clean column names
    return df, diabetes.feature_names, diabetes.target_name

df, feature_names, target_name = get_diabetes_data()

# --- Streamlit UI Layout ---
st.title("Multiple Linear Regression Assumptions Analyzer")

st.markdown("""
This application allows you to explore the assumptions underlying multiple linear regression models
using int\fractive visualizations. Select independent variables to build a model and
examine the diagnostics plots.
""")

st.sidebar.header("Configuration")
st.sidebar.markdown("---")

# Dependent Variable selection (fixed for this example)
st.sidebar.subheader("Dependent Variable")
st.sidebar.write(f"**{target_name.replace('_', ' ').title()}** (fixed for demonstration)")

# Independent Variables selection
st.sidebar.subheader("Select Independent Variables")
selected_features = []
default_selected = feature_names[:3] # Select first 3 features by default

for feature in feature_names:
    if st.sidebar.checkbox(feature.replace('_', ' ').title(), value=feature in default_selected):
        selected_features.append(feature)

st.sidebar.markdown("---")

# Display selected data and run regression only if features are selected
if not selected_features:
    st.warning("Please select at least one independent variable.")
else:
    # Prepare data for statsmodels
    X = df[selected_features]
    y = df[target_name]
    X = sm.add_constant(X) # Add an intercept term

    # --- Model Training ---
    try:
        model = sm.OLS(y, X).fit()
        predictions = model.predict(X)
        residuals = model.resid

        st.subheader("1. Regression Model Summary")
        st.code(model.summary().as_text()) # Display model summary

        # --- Explanations and Visualizations ---
        st.markdown("---")
        st.header("Core Concepts and Assumptions Analysis")

        # Linearity and Multicollinearity (initial check)
        st.subheader("2. Linearity and Multicollinearity Visual Check: Scatterplot Matrix")
        st.markdown("""
        The **linearity** assumption states that the relationship between independent variables and the
        dependent variable is linear. The **no multicollinearity** assumption requires that independent variables
        are not highly correlated with each other. A scatterplot matrix helps visualize pairwise relationships.
        """)
        
        # Combine dependent and independent variables for scatter_matrix
        plot_vars = [target_name] + selected_features
        fig_scatter_matrix = px.scatter_matrix(df, dimensions=plot_vars)
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)
        st.markdown("""
        *   **Linearity**: Look for linear patterns in the scatter plots between independent variables and the dependent variable.
            Curved patterns suggest non-linearity.
        *   **Multicollinearity**: Observe the scatter plots between independent variables. Strong linear relationships
            (e.g., highly clustered points forming a line) indicate potential multicollinearity.
        """)

        # Homoscedasticity and Linearity (further check)
        st.subheader("3. Homoscedasticity and Linearity Check: Residuals vs. Predicted Values")
        st.markdown("""
        The **homoscedasticity** assumption states that the variance of the error terms is constant across all
        levels of the independent variables. Violations (heteroscedasticity) often appear as a fan-like shape in this plot.
        This plot also helps confirm **linearity**; a random scatter around zero indicates linearity.
        """)
        fig_homo = go.Figure()
        fig_homo.add_\frace(go.Scatter(x=predictions, y=residuals, mode='markers', name='Residuals'))
        fig_homo.add_hline(y=0, line_dash="dash", line_color="red")
        fig_homo.update_layout(title="Residuals vs. Predicted Values",
                               xaxis_title=f"Predicted {target_name.replace('_', ' ').title()}",
                               yaxis_title="Residuals")
        st.plotly_chart(fig_homo, use_container_width=True)
        st.markdown("""
        *   **Homoscedasticity**: Look for a random scatter of points around the red line ($y=0$). If the spread of
            residuals widens or narrows as predicted values increase (a "fan" or "cone" shape),
            it suggests heteroscedasticity.
        *   **Linearity**: Any clear pattern (e.g., U-shape, S-shape) in the residuals indicates a violation of linearity.
            Ideally, residuals should be randomly scattered.
        """)

        # Independence of Residuals
        st.subheader("4. Independence of Residuals Check: Residuals vs. Independent Variables")
        st.markdown("""
        The **independence of residuals** assumption states that the error terms for different observations are
        uncorrelated. This is especially important for time-series data. Patterns in these plots might suggest
        autocorrelation or other uncaptured relationships.
        """)
        
        # Create subplots for residuals vs. each independent variable
        num_cols = 3 # Number of columns for subplot grid
        rows = (len(selected_features) + num_cols - 1) // num_cols
        fig_indep = make_subplots(rows=rows, cols=num_cols,
                                 subplot_titles=[f"Residuals vs. {feat.replace('_', ' ').title()}" for feat in selected_features])
        
        for i, feature in enumerate(selected_features):
            row = (i // num_cols) + 1
            col = (i % num_cols) + 1
            fig_indep.add_\frace(go.Scatter(x=df[feature], y=residuals, mode='markers', name=feature.replace('_', ' ').title()),
                                row=row, col=col)
            fig_indep.add_hline(y=0, line_dash="dash", line_color="red", row=row, col=col)
            
        fig_indep.update_layout(height=400*rows, showlegend=False)
        st.plotly_chart(fig_indep, use_container_width=True)
        st.markdown("""
        *   **Independence**: For each plot, ideally, residuals should be randomly scattered around the red line.
            Any discernible pattern (e.g., cyclical, trending) might indicate a lack of independence or that the
            independent variable has a non-linear relationship not captured by the model.
        """)

        # Normality of Residuals
        st.subheader("5. Normality of Residuals Check: Histogram and Q-Q Plot")
        st.markdown("""
        The **normality of residuals** assumption states that the error terms are normally distributed.
        This is important for the validity of statistical inference, especially with smaller sample sizes.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Histogram of Residuals")
            fig_hist = px.histogram(residuals, nbins=30, title="Distribution of Residuals")
            fig_hist.update_layout(xaxis_title="Residuals", yaxis_title="Frequency")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("""
            *   **Histogram**: Look for a bell-shaped distribution, centered around zero. Skewness or
                multiple peaks can indicate non-normality.
            """)
        
        with col2:
            st.markdown("##### Q-Q Plot of Residuals")
            fig_qq = sm.qqplot(residuals, line='s', fit=True)
            # Convert matplotlib figure to Plotly
            plotly_qq = go.Figure(data=fig_qq.data, layout=fig_qq.layout)
            plotly_qq.update_layout(title="Q-Q Plot of Residuals",
                                    xaxis_title="Theoretical Quantiles",
                                    yaxis_title="Sample Quantiles")
            st.plotly_chart(plotly_qq, use_container_width=True)
            st.markdown("""
            *   **Q-Q Plot**: Points should closely follow the straight line. Deviations from the line,
                especially at the tails, suggest departures from normality.
            """)

    except Exception as e:
        st.error(f"An error occurred during model fitting or plotting: {e}")
        st.warning("Please check your selected variables. Some combinations may cause issues (e.g., perfect multicollinearity).")

### Data Processing and Model Logic
*   **Data Loading**: The `get_diabetes_data()` function uses `sklearn.datasets.load_diabetes(as_frame=True)` to directly load the dataset into a Pandas DataFrame. Column names are cleaned for better display. Using `@st.cache_data` ensures efficient re-execution.
*   **Model Specification**:
    *   The selected independent variables (`selected_features`) form the `X` matrix.
    *   `sm.add_constant(X)` is used to explicitly add an intercept column to the independent variables, as `statsmodels` OLS does not add it by default.
    *   The dependent variable `y` is set to the `target_name` column.
*   **Model Fitting**: `sm.OLS(y, X).fit()` fits the Ordinary Least Squares regression model.
*   **Results Ex\fraction**:
    *   `model.summary()` provides a comprehensive statistical summary, including coefficients, R-squared, p-values, and various diagnostic statistics. `as_text()` converts it to a string for display.
    *   `model.predict(X)` calculates the predicted values ($\hat{Y}$) for the given `X`.
    *   `model.resid` calculates the residuals ($e = Y - \hat{Y}$).

### Visualization Logic
*   **Scatterplot Matrix**: `plotly.express.scatter_matrix(df, dimensions=plot_vars)` generates a grid of scatter plots for all pairwise combinations of the specified variables (`plot_vars`). This is a powerful tool for initial visual inspection of relationships and potential multicollinearity.
*   **Residuals vs. Predicted Values**:
    *   A `go.Figure()` is initialized.
    *   `go.Scatter(x=predictions, y=residuals, mode='markers')` creates the scatter plot.
    *   `fig_homo.add_hline(y=0, ...)` adds a horizontal reference line at zero, crucial for interpreting residuals.
*   **Residuals vs. Factors Plots**:
    *   `make_subplots` is used to create a grid of plots, one for each selected independent variable.
    *   Each subplot contains a scatter \frace of the respective independent variable against the residuals, along with a zero-reference line.
*   **Normality Plots**:
    *   **Histogram**: `px.histogram(residuals, ...)` generates a histogram to show the distribution of the residuals.
    *   **Q-Q Plot**: `sm.qqplot(residuals, line='s', fit=True)` generates a Q-Q plot using `statsmodels`. The `line='s'` argument adds a standardized line to compare against. The `fit=True` argument estimates mean and standard deviation from the data. The matplotlib figure generated by `sm.qqplot` is then converted to a Plotly figure object for compatibility with `st.plotly_chart`.

## User Interface Components

The application's user interface will be composed of the following key components:

1.  **Main Title and Overview**:
    *   `st.title("Multiple Linear Regression Assumptions Analyzer")`: Prominent title at the top.
    *   `st.markdown(...)`: A brief introductory paragraph explaining the application's purpose and how to use it.

2.  **Sidebar (`st.sidebar`)**:
    *   **Configuration Header**: `st.sidebar.header("Configuration")`
    *   **Dependent Variable Display**: `st.sidebar.subheader("Dependent Variable")` and `st.sidebar.write()` to inform the user which variable is the dependent one (fixed in this demonstration).
    *   **Independent Variable Selection**:
        *   `st.sidebar.subheader("Select Independent Variables")`
        *   Multiple `st.sidebar.checkbox()` widgets, one for each available independent variable in the dataset. Users can select/deselect variables to include in the regression model. Default variables are pre-selected for convenience.
    *   **Separators**: `st.sidebar.markdown("---")` for visual separation.

3.  **Main Content Area**:
    *   **Regression Model Summary**:
        *   `st.subheader("1. Regression Model Summary")`
        *   `st.code(model.summary().as_text())`: Displays the detailed statistical output from the `statsmodels` OLS regression, including coefficients, p-values, R-squared, and other diagnostic metrics.
    *   **Core Concepts and Assumptions Analysis Header**: `st.header("Core Concepts and Assumptions Analysis")` to structure the diagnostic sections.
    *   **Individual Assumption Sections (Subheaders, Explanations, and Plots)**:
        *   **Linearity and Multicollinearity (Scatterplot Matrix)**:
            *   `st.subheader("2. Linearity and Multicollinearity Visual Check: Scatterplot Matrix")`
            *   `st.markdown(...)`: Explanations of linearity and multicollinearity, and how the scatterplot matrix helps in their visual assessment.
            *   `st.plotly_chart(fig_scatter_matrix, use_container_width=True)`: Displays the int\fractive scatterplot matrix.
        *   **Homoscedasticity and Linearity (Residuals vs. Predicted Values)**:
            *   `st.subheader("3. Homoscedasticity and Linearity Check: Residuals vs. Predicted Values")`
            *   `st.markdown(...)`: Explanations of homoscedasticity and further linearity checks.
            *   `st.plotly_chart(fig_homo, use_container_width=True)`: Displays the int\fractive residuals vs. predicted values plot.
        *   **Independence of Residuals (Residuals vs. Independent Variables)**:
            *   `st.subheader("4. Independence of Residuals Check: Residuals vs. Independent Variables")`
            *   `st.markdown(...)`: Explanation of residual independence.
            *   `st.plotly_chart(fig_indep, use_container_width=True)`: Displays the int\fractive grid of residuals vs. individual independent variables plots.
        *   **Normality of Residuals (Histogram and Q-Q Plot)**:
            *   `st.subheader("5. Normality of Residuals Check: Histogram and Q-Q Plot")`
            *   `st.markdown(...)`: Explanation of residual normality.
            *   `st.columns(2)`: Divides the section into two columns for side-by-side display of the histogram and Q-Q plot.
            *   `st.plotly_chart(fig_hist, use_container_width=True)`: Displays the histogram of residuals.
            *   `st.plotly_chart(plotly_qq, use_container_width=True)`: Displays the Q-Q plot of residuals.

This structured approach ensures that the application is functional, educational, and user-friendly, directly addressing the specified requirements for a Streamlit application focused on multiple linear regression assumptions.


### Appendix Code

```code
Revenue recognition calculation (Reference: Page 12, Part 2, Example 8):
```
$600,000 (60% × $1 million) in revenue for the first year.
```

Net income definition (Reference: Page 6, "The definition of income..."):
```
Net income equals (i) revenue minus expenses in the
ordinary activities of the business, plus (ii) other income minus other expenses, plus
(iii) gains minus losses.
```

Inventory Purchases data (Reference: Page 15, Example 1):
```
Inventory Purchases
First quarter       2,000 units at $40 per unit
Second quarter      1,500 units at $41 per unit
Third quarter       2,200 units at $43 per unit
Fourth quarter      1,900 units at $45 per unit
Total               7,600 units at a total cost of $321,600
```

Cost of Goods Sold and Remaining Inventory (Reference: Page 15, Example 1):
```
Cost of Goods Sold
From the first quarter       2,000 units at $40 per unit = $80,000
From the second quarter      1,500 units at $41 per unit = $61,500
From the third quarter       2,100 units at $43 per unit = $90,300
Total cost of goods sold                                 $231,800

Cost of Goods Remaining in Inventory
From the third quarter       100 units at $43 per unit = $4,300
From the fourth quarter      1,900 units at $45 per unit = $85,500
Total remaining (or ending) inventory cost               $89,800
```

Cost Reconciliation and Gross Profit (Reference: Page 16, following Example 1):
```
To confirm that total costs are accounted for: $231,800 + $89,800 = $321,600.
The cost of the goods sold would be expensed against the revenue of $280,000 as follows:
Revenue           $280,000
Cost of goods sold 231,800
Gross profit       $48,200
```

Weighted Average Cost per Unit (Reference: Page 16, Example 2):
```
For KDL, the weighted average cost per unit would be
$321,600/7,600 units = $42.3158 per unit
```

Cost of Goods Sold using Weighted Average Cost Method (Reference: Page 16, Example 2):
```
Cost of goods sold using the weighted average cost method would be
5,600 units at $42.3158 = $236,968
```

Ending Inventory using Weighted Average Cost Method (Reference: Page 17, Example 2):
```
Ending inventory using the weighted average cost method would be
2,000 units at $42.3158 = $84,632
```

LIFO Inventory and Cost of Goods Sold (Reference: Page 17, Example 2):
```
Ending inventory 2,000 units at $40 per unit = $80,000
The remaining costs would be allocated to cost of goods sold under LIFO:
Total costs of $321,600 less $80,000 remaining in ending inventory = $241,600
Alternatively, the cost of the last 5,600 units purchased is allocated to cost of goods sold under LIFO:
1,900 units at $45 per unit + 2,200 units at $43 per unit + 1,500 units at $41 per unit = $241,600
```

Straight-Line Depreciation Formula (Reference: Page 20, Example 3):
```
Cost - Residual value
Estimated useful life
```

Straight-Line Depreciation Calculations (Reference: Page 20, Example 3):
```
($10,000 – $0)/5 years = $2,000.
($10,000 – $4,000)/5 years = $1,200.
($10,000 – $0)/10 years = $1,000.
```

Diminishing Balance Depreciation - Year 1 (Reference: Page 21, Example 4):
```
Depreciation expense for the first full year of use of the asset would be 40 percent of $11,000,
or $4,400.
```

Diminishing Balance Depreciation - Net Book Value Year 2 (Reference: Page 21, Example 4):
```
Asset cost                 $11,000
Less: Accumulated depreciation (4,400)
Net book value             $6,600
```

Diminishing Balance Depreciation - Year 2 (Reference: Page 21, Example 4):
```
For the second full year, depreciation expense would be $6,600 × 40 percent, or
$2,640.
```

Diminishing Balance Depreciation - Net Book Value Year 3 (Reference: Page 21, Example 4):
```
Asset cost                 $11,000
Less: Accumulated depreciation (7,040)
Net book value             $3,960
```

Diminishing Balance Depreciation - Year 3 (Reference: Page 22, Example 4):
```
For the third full year, depreciation would be $3,960 × 40 percent, or $1,584.
```

Diminishing Balance Depreciation - Net Book Value Year 4 (Reference: Page 22, Example 4):
```
Asset cost                 $11,000
Less: Accumulated depreciation (8,624)
Net book value             $2,376
```

Diminishing Balance Depreciation - Year 4 (Reference: Page 22, Example 4):
```
For the fourth full year, depreciation would be $2,376 × 40 percent, or $950.
```

Diminishing Balance Depreciation - Net Book Value Year 5 (Reference: Page 22, Example 4):
```
Asset cost                 $11,000
Less: Accumulated depreciation (9,574)
Net book value             $1,426
```

Diminishing Balance Depreciation - Year 5 and Final Net Book Value (Reference: Page 22, Example 4):
```
For the fifth year, if deprecation were determined as in previous years, it would
amount to $570 ($1,426 × 40 percent). However, this would result in a remain-
ing net book value of the asset below its estimated residual value of $1,000. So,
instead, only $426 would be depreciated, leaving a $1,000 net book value at the
end of the fifth year.
Asset cost                 $11,000
Less: Accumulated depreciation (10,000)
Net book value             $1,000
```

Groupe Danone Footnotes Excerpt (Reference: Page 26, Exhibit 8):
```
(in € millions)                               Related income (expense)
Capital gain on disposal of Stonyfield               628
Compensation received following the decision of the Singapore arbi-
tration court in the Fonterra case                   105
Territorial risks, mainly in certain countries in the ALMA region   (148)
Costs associated with the integration of WhiteWave        (118)
Impairment of several intangible assets in Waters and Specialized
Nutrition Reporting entities                         (115)
Remainder of table omitted
```

Microsoft Corporation Income Statement Adjustment (Reference: Page 27, Example 5):
```
(In $ millions, except per share   As           New
amounts)                       Previously   Revenue
                                   Reported     Standard     As
                                                Adjustment   Restated

Income Statements
Year Ended June 30, 2017
Revenue                        89,950       6,621        96,571
Provision for income taxes     1,945        2,467        4,412
Net income                     21,204       4,285        25,489
Diluted earnings per share     2.71         0.54         3.25

Year Ended June 30, 2016
Revenue                        85,320       5,834        91,154
Provision for income taxes     2,953        2,147        5,100
Net income                     16,798       3,741        20,539
Diluted earnings per share     2.1          0.46         2.56
```

Microsoft Revenue and Profit Margin Analysis (Reference: Page 28, Example 5 Solution):
```
The net profit margin is 26.4% (= 25,489/96,571) under the new standard versus 23.6% (=
21,204/89,950) under the old standard. Reported revenue grew faster under
the new standard. Revenue growth under the new standard was 5.9% [=
(96,571/91,154) – 1] compared to 5.4% [= (89,950/85,320) – 1)] under the
old standard.
```

Basic EPS Formula (Reference: Page 31, "Basic EPS" section):
```
Basic EPS = Net income - Preferred dividends
            -------------------------------------
            Weighted average number of shares outstanding
```

AB InBev's Earnings Per Share (Reference: Page 31, Exhibit 10):
```
12 Months Ended December 31
2017       2016       2015

Basic earnings per share                   $4.06      $0.72      $5.05
Diluted earnings per share                 3.98       0.71       4.96
Basic earnings per share from continuing
operations                                 4.04       0.69       5.05
Diluted earnings per share from continuing
operations                                 $3.96      $0.68      $4.96
```

Basic EPS Calculation (Reference: Page 32, Example 6):
```
Shopalot's basic EPS is $1.30 ($1,950,000 divided by 1,500,000 shares).
```

Angler Products Common Stock Share Information (Reference: Page 32, Example 7):
```
Shares outstanding on 1 January 2018           1,000,000
Shares issued on 1 April 2018                  200,000
Shares repurchased (treasury shares) on 1 October 2018 (100,000)
Shares outstanding on 31 December 2018         1,100,000
```

Angler Products Weighted Average Number of Shares Outstanding (Reference: Page 32, Example 7):
```
1,000,000 x (3 months/12 months) = 250,000
1,200,000 × (6 months/12 months) = 600,000
1,100,000 x (3 months/12 months) = 275,000
Weighted average number of shares outstanding   1,125,000
```

Angler Products Basic EPS (Reference: Page 33, Example 7):
```
Basic EPS = (Net income – Preferred dividends)/Weighted average number
of shares = ($2,500,000 – $200,000)/1,125,000 = $2.04
```

Angler Products Basic EPS with Stock Split (Reference: Page 33, Example 8):
```
The weighted average number of shares would,
therefore, be 2,250,000, and the basic EPS would be $1.02 [= ($2,500,000 –
$200,000)/2,250,000].
```

Diluted EPS Formula - Preferred Stock (Reference: Page 34, Formula 2):
```
Diluted EPS = (Net income)
              ------------------------------------------
              (Weighted average number of shares
              outstanding + New common shares that
              would have been issued at conversion)
```

Bright-Warm Utility Company Diluted EPS Calculation (Reference: Page 34, Example 9):
```
If the 20,000 shares of convertible preferred had each converted into 5
shares of the company's common stock, the company would have had an
additional 100,000 shares of common stock (5 shares of common for each of
the 20,000 shares of preferred). If the conversion had taken place, the com-
pany would not have paid preferred dividends of $200,000 ($10 per share for
each of the 20,000 shares of preferred). As shown in Exhibit 11, the compa-
ny's basic EPS was $3.10 and its diluted EPS was $2.92.
```

Bright-Warm Utility Company Diluted EPS Table (Reference: Page 34, Exhibit 11):
```
Basic EPS    Diluted EPS Using
             If-Converted Method

Net income             $1,750,000   $1,750,000
Preferred dividend     -200,000     0
Numerator              $1,550,000   $1,750,000
Weighted average number of shares
outstanding            500,000      500,000
Additional shares issued if preferred
converted              0            100,000
```

Bright-Warm Utility Company Diluted EPS Table Continued (Reference: Page 35, Exhibit 11):
```
Denominator            500,000      600,000
EPS                    $3.10        $2.92
```

Diluted EPS Formula - Convertible Debt (Reference: Page 35, Formula 3):
```
Diluted EPS = (Net income + After-tax interest on
               convertible debt - Preferred dividends)
               -------------------------------------
               (Weighted average number of shares
               outstanding + Additional common
               shares that would have been
               issued at conversion)
```

Oppnox Company Diluted EPS Calculation (Reference: Page 35, Example 10):
```
If the debt securities had been converted, the debt securities would no
longer be outstanding and instead, an additional 10,000 shares of common
stock would be outstanding. Also, if the debt securities had been converted,
the company would not have paid interest of $3,000 on the convertible debt,
so net income available to common shareholders would have increased by
$2,100 [= $3,000(1 – 0.30)] on an after-tax basis.
```

Oppnox Company Diluted EPS Table (Reference: Page 36, Exhibit 12):
```
Basic EPS    Diluted EPS Using
             If-Converted Method

Net income             $750,000     $750,000
After-tax cost of interest           2,100
Numerator              $750,000     $752,100
Weighted average number of shares
outstanding            690,000      690,000
If converted           0            10,000
Denominator            690,000      700,000
EPS                    $1.09        $1.07
```

Diluted EPS Formula - Options (Reference: Page 37, Formula 4):
```
Diluted EPS = (Net income - Preferred dividends)
              --------------------------------------------------------------
              [Weighted average number of shares
              outstanding + (New shares that would
              have been issued at option exercise-
              Shares that could have been purchased
              with cash received upon exercise) x
              (Proportion of year during which the
              financial instruments were outstanding)]
```

Hihotech Company Diluted EPS Calculation (Reference: Page 37, Example 11):
```
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

Hihotech Company Diluted EPS Table (Reference: Page 38, Exhibit 13):
```
Basic EPS    Diluted EPS Using
             Treasury Stock
             Method

Net income             $2,300,000   $2,300,000
Numerator              $2,300,000   $2,300,000
Weighted average number of shares
outstanding            800,000      800,000
If converted           0            10,909
Denominator            800,000      810,909
EPS                    $2.88        $2.84
```

Hihotech Company Diluted EPS under IFRS (Reference: Page 38, Example 12):
```
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

Antidilutive Security Calculation (Reference: Page 39, Example 13):
```
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

Antidilutive Security Table (Reference: Page 39, Exhibit 14):
```
Basic EPS    Diluted EPS Using
             If-Converted Method

Net income             $1,750,000   $1,750,000
Preferred dividend     -200,000     0
Numerator              $1,550,000   $1,750,000
Weighted average number of shares
outstanding            500,000      500,000
If converted           0            60,000
Denominator            500,000      560,000
```

Antidilutive Security Table Continued (Reference: Page 40, Exhibit 14):
```
EPS                    $3.10        $3.13
-Exceeds basic EPS; security
is antidilutive and, therefore,
not included. Reported
diluted EPS= $3.10.
```

Net Profit Margin Formula (Reference: Page 43, "Net profit margin" section):
```
Net profit margin = Net income
                    -----------
                    Revenue
```

Gross Profit Margin Formula (Reference: Page 43, "Gross profit margin" section):
```
Gross profit margin = Gross profit
                      ------------
                      Revenue
```

Income Statements for Companies A, B, and C ($) (Reference: Page 41, Exhibit 15, Panel A):
```
A            B            C

Sales                       $10,000,000  $10,000,000  $2,000,000
Cost of sales               3,000,000    7,500,000    600,000
Gross profit                7,000,000    2,500,000    1,400,000
Selling, general, and administrative expenses
                            1,000,000    1,000,000    200,000
Research and development    2,000,000    —            400,000
Advertising                 2,000,000    —            400,000
Operating profit            2,000,000    1,500,000    400,000
```

Common-Size Income Statements for Companies A, B, and C (%) (Reference: Page 42, Exhibit 15, Panel B):
```
A           B           C

Sales                       100%        100%        100%
Cost of sales               30          75          30
Gross profit                70          25          70
Selling, general, and administrative expenses
                            10          10          10
Research and development    20          0           20
Advertising                 20          0           20
Operating profit            20          15          20
```

Median Common-Size Income Statement Statistics for the S&P 500 (Reference: Page 42, Exhibit 16):
```
Energy    Materials Industrials Consumer    Consumer    Health Care
                               Discretionary Staples
Number of observations  34        27        69          81          34          59
Gross Margin            37.7%     33.0%     36.8%       37.6%       43.4%       59.0%
Operating Margin        6.4%      14.9%     13.5%       11.0%       17.2%       17.4%
Net Profit Margin       4.9%      9.9%      8.8%        6.0%        10.9%       7.2%
```

AB InBev's Margins: Abbreviated Common-Size Income Statement (Reference: Page 44, Exhibit 17):
```
12 Months Ended December 31
2017                   2016                   2015
$        %             $        %             $        %
Revenue                     56,444   100.0        45,517   100.0        43,604   100.0
Cost of sales             (21,386)   (37.9)     (17,803)   (39.1)     (17,137)   (39.3)
Gross profit                35,058    62.1        27,715    60.9        26,467    60.7
Distribution expenses      (5,876)   (10.4)      (4,543)   (10.0)      (4,259)    (9.8)
Sales and marketing expenses
                          (8,382)   (14.9)      (7,745)   (17.0)      (6,913)   (15.9)
Administrative expenses    (3,841)    (6.8)      (2,883)    (6.3)      (2,560)    (5.9)
Portions omitted
Profit from operations      17,152    30.4        12,882    28.3        13,904    31.9
Finance cost               (6,885)   (12.2)      (9,382)   (20.6)      (3,142)    (7.2)
Finance income                 378     0.7           818     1.8         1,689     3.9
Net finance income/(cost)  (6,507)   (11.5)      (8,564)   (18.8)      (1,453)    (3.3)
Share of result of associates and joint
ventures                       430     0.8            16     0.0            10     0.0
Profit before tax           11,076    19.6         4,334     9.5        12,461    28.6
Income tax expense         (1,920)    (3.4)      (1,613)    (3.5)      (2,594)    (5.9)
Profit from continuing operations
                             9,155    16.2         2,721     6.0         9,867    22.6
Profit from discontinued operations
                                28     0.0            48     0.1          —       —
Profit of the year           9,183    16.3         2,769     6.1         9,867    22.6
```

Other Comprehensive Income Calculation 1 (Reference: Page 47, Example 14, Solution to 1):
```
€10 million [€227− (€200 + €20 – €3)]
```

Other Comprehensive Income in Analysis - Initial Table (Reference: Page 47, Example 15):
```
Company A    Company B
Price                          $35          $30
EPS                            $1.60        $0.90
P/E ratio                      21.9x        33.3x
Other comprehensive income (loss) $ million ($16.272)    $(1.757)
Shares (millions)              22.6         25.1
```

Other Comprehensive Income in Analysis - Solution Table (Reference: Page 48, Example 15, Solution):
```
Company A    Company B
Price                          $35          $30
EPS                            $1.60        $0.90
OCI (loss) $ million           ($16.272)    $(1.757)
Shares (millions)              22.6         25.1
OCI (loss) per share           $(0.72)      $(0.07)
Comprehensive EPS = EPS + OCI per share $ 0.88       $0.83
Price/Comprehensive EPS ratio  39.8x        36.1x
```

Denali Limited Income Statement Information (Reference: Page 51, \fractice Problems, Question 3):
```
Revenue                  $4,000,000
Cost of goods sold       $3,000,000
Other operating expenses $500,000
Interest expense         $100,000
Tax expense              $120,000
```

Fairplay Sale Information (Reference: Page 51, \fractice Problems, Question 5):
```
Revenue            $1,000,000
Returns of goods sold $100,000
Cash collected     $800,000
Cost of goods sold $700,000
```

Laurelli Builders Financial Data (Reference: Page 54, \fractice Problems, Question 20):
```
Common shares outstanding, 1 January     2,020,000
Common shares issued as stock dividend, 1 June 380,000
Warrants outstanding, 1 January          500,000
Net income                               $3,350,000
Preferred stock dividends paid           $430,000
Common stock dividends paid              $240,000
```

Workhard Financial Statement Data (Reference: Page 55, \fractice Problems, Question 24):
```
$ millions
Beginning shareholders' equity     475
Ending shareholders' equity        493
Unrealized gain on available-for-sale securities 5
Unrealized loss on derivatives accounted for as hedges -3
Foreign currency translation gain on consolidation 2
Dividends paid                     1
Net income                         15
```

FIFO Calculation (Reference: Page 56, Solutions, Question 8):
```
Under the first in, first out (FIFO) method, the first 10,000 units sold
came from the October purchases at £10, and the next 2,000 units sold came
from the November purchases at £11.
```

Weighted Average Cost Method Calculation (Reference: Page 56, Solutions, Question 9):
```
October purchases    10,000 units    $100,000
November purchases    5,000 units    $55,000
Total              15,000 units    $155,000

$155,000/15,000 units = $10.3333
$10.3333 × 12,000 units = $124,000
```

Straight-Line Depreciation Calculation (Reference: Page 56, Solutions, Question 11):
```
Straight-line depreciation would be ($600,000 – $50,000)/10, or $55,000.
```

Double-Declining Balance Depreciation Calculation (Reference: Page 56, Solutions, Question 12):
```
Double-declining balance depreciation would be $600,000 × 20 per-
cent (twice the straight-line rate).
```

Basic Earnings Per Share Calculation (Reference: Page 57, Solutions, Question 16):
```
The weighted average number of shares outstanding for 2009 is
1,050,000. Basic earnings per share would be $1,000,000 divided by 1,050,000, or
$0.95.
```

Diluted EPS Calculation (Reference: Page 57, Solutions, Question 18):
```
Diluted EPS = (Net income)/(Weighted average number of shares outstanding +
New common shares that would have been issued at conversion)
= $200,000,000/[50,000,000 + (2,000,000 × 2)]
= $3.70
```

Diluted EPS Calculation (Reference: Page 57, Solutions, Question 19):
```
Proceeds from option exercise = 100,000 × $20 = $2,000,000
Shares repurchased = $2,000,000/$25 = 80,000
The net increase in shares outstanding is thus 100,000 – 80,000 = 20,000. There-
fore, the diluted EPS for CWC = ($12,000,000 – $800,000)/2,020,000 = $5.54.
```

Basic EPS Calculation (Reference: Page 57, Solutions, Question 20):
```
LB's basic EPS is $1.22 [= ($3,350,000 – $430,000)/2,400,000].
```

Treasury Stock Method Calculation (Reference: Page 58, Solutions, Question 21):
```
The company would receive $100,000 (10,000 × $10) and would re-
purchase 6,667 shares ($100,000/$15). The shares for the denominator would be:
Shares outstanding    1,000,000
Options exercises      10,000
Treasury shares purchased (6,667)
Denominator          1,003,333
```

Comprehensive Income Calculation (Reference: Page 58, Solutions, Question 24):
```
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