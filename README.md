# 🔮 Auto Data Sorcerer

An advanced, automated data science platform built with Streamlit that provides comprehensive data analysis, outlier detection, and machine learning capabilities.

## ✨ Features

### 📊 **Exploratory Data Analysis (EDA)**
- **Data Preview & Info**: Quick overview of dataset structure and types
- **Advanced Column Type Detection**: Automatically categorizes columns into:
  - Continuous and discrete numerical
  - Categorical (nominal vs ordinal)
  - DateTime columns
  - Text columns
  - Boolean columns
  - ID columns
- **Missing Value Analysis**: Visualization and handling strategies
- **Statistical Analysis**: Comprehensive descriptive statistics
- **Correlation Analysis**: Interactive heatmaps for numerical data
- **Data Visualization**: Dynamic histograms and pie charts

### 🎯 **Advanced Outlier Detection & Handling**
- **Multiple Detection Methods**:
  - **IQR (Interquartile Range)**: Traditional statistical method
  - **Z-Score**: Standard deviation based detection
  - **Modified Z-Score**: Robust method using median absolute deviation
  - **Isolation Forest**: Multivariate outlier detection using machine learning

- **Comprehensive Visualization**:
  - Box plots with outlier highlighting
  - Distribution histograms
  - Scatter plots with outlier markers
  - Q-Q plots for normality assessment

- **Flexible Handling Strategies**:
  - **Remove outliers**: Drop outlier rows
  - **Cap outliers**: Limit values to IQR bounds
  - **Log transformation**: Natural log transformation for skewed data
  - **Square root transformation**: Reduce impact of extreme values
  - **Winsorization**: Replace extreme values with percentile limits

### 🧹 **Data Quality & Cleaning**
- **Missing Value Handling**: Multiple imputation strategies
- **Data Type Standardization**: Automatic type conversion and validation
- **Interactive Processing**: User-controlled data cleaning pipeline
- **Before/After Comparison**: Statistical summaries of cleaning effects

### 🤖 **AutoML Integration** (Coming Soon)
- Automated machine learning model training
- Model comparison and selection
- Performance evaluation

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/vaibhavdubey06/AI-data-scientist.git
cd AI-data-scientist/AutoDS
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
AutoDS/
├── app.py                 # Main Streamlit application
├── eda_utils.py          # EDA functionality and visualizations
├── data_quality.py       # Outlier detection and data quality tools
├── ml_utils.py           # Machine learning utilities (in development)
├── requirements.txt      # Python dependencies
├── sample_data/
│   └── titanic.csv      # Sample dataset for testing
└── README.md            # Project documentation
```

## 🔧 Usage

### 1. **Data Upload**
- Upload your CSV or Excel files using the sidebar
- Or use the built-in Titanic dataset for testing

### 2. **Exploratory Data Analysis**
- View data preview and statistical summaries
- Analyze column types and distributions
- Handle missing values with various strategies
- Generate correlation heatmaps and visualizations

### 3. **Outlier Analysis**
- Choose from multiple detection methods
- Visualize outliers with interactive plots
- Apply different handling strategies
- Compare before/after statistics
- Download processed datasets

### 4. **Advanced Features**
- Session state management for processed data
- Export cleaned datasets
- Comprehensive outlier summary reports

## 📊 Supported Data Types

- **Numerical**: Continuous and discrete variables
- **Categorical**: Nominal and ordinal categories
- **DateTime**: Automatic date/time parsing
- **Text**: Long-form text content
- **Boolean**: Binary variables
- **Mixed Types**: Automatic type inference and conversion

## 🛠️ Dependencies

```
streamlit>=1.25.0
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
plotly>=5.0.0
seaborn>=0.11.0
matplotlib>=3.5.0
lazypredict
ydata-profiling
sweetviz
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Roadmap

- [ ] **Enhanced AutoML**: Complete ML pipeline integration
- [ ] **Advanced Imputation**: KNN and iterative imputation methods
- [ ] **Data Profiling**: Automated data quality reports
- [ ] **Feature Engineering**: Automated feature creation and selection
- [ ] **Model Deployment**: One-click model deployment
- [ ] **API Integration**: REST API for programmatic access

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vaibhav Dubey**
- GitHub: [@vaibhavdubey06](https://github.com/vaibhavdubey06)
- Repository: [AI-data-scientist](https://github.com/vaibhavdubey06/AI-data-scientist)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data visualization powered by [Plotly](https://plotly.com/)
- Statistical analysis using [SciPy](https://scipy.org/)
- Machine learning capabilities via [scikit-learn](https://scikit-learn.org/)

---

**Made with ❤️ for the data science community**
