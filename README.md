# Marketing Campaign Causal Analysis

This project analyzes the effectiveness of a marketing campaign using various data science techniques, including exploratory data analysis, statistical significance testing, and causal inference analysis.

## Project Structure

- `data/`: Contains the input data file (marketing_ab.csv)
- `notebooks/`: Jupyter notebooks for showcasing the analysis process and results
- `src/`: Python scripts for each analysis component
- `results/`: Generated plots and results
- `README.md`: Project documentation
- `requirements.txt`: Required Python packages

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/marketing-campaign-causal-analysis.git
   cd marketing-campaign-causal-analysis
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place the "marketing_ab.csv" file in the `data/` directory.

2. Run the analysis components:
   ```
   python src/eda.py
   python src/statistical_tests.py
   python src/causal_inference.py
   ```

3. Open and run the Jupyter notebook:
   ```
   jupyter notebook notebooks/marketing_campaign_analysis.ipynb
   ```

## Analysis Components

1. Exploratory Data Analysis (EDA):
   - Basic statistical analysis
   - Data visualizations (histograms, box plots, scatter plots)
   - Correlation analysis

2. Statistical Significance Testing:
   - T-tests for comparing means between groups
   - Chi-square tests for categorical variable relationships
   - ANOVA for multi-group comparisons

3. Causal Inference Analysis:
   - Propensity Score Matching (PSM)
   - Difference-in-Differences (DiD) analysis

For detailed results and interpretations, please refer to the Jupyter notebook in the `notebooks/` directory.