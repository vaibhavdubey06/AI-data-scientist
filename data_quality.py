import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def detect_outliers_iqr(series):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (series < lower_bound) | (series > upper_bound)
    return outliers_mask, lower_bound, upper_bound


def detect_outliers_zscore(series, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(series.dropna()))
    outliers_mask = pd.Series(False, index=series.index)
    outliers_mask.loc[series.dropna().index] = z_scores > threshold
    return outliers_mask, threshold


def detect_outliers_modified_zscore(series, threshold=3.5):
    """Detect outliers using Modified Z-score method (more robust)"""
    median = series.median()
    mad = np.median(np.abs(series - median))
    modified_z_scores = 0.6745 * (series - median) / mad
    outliers_mask = np.abs(modified_z_scores) > threshold
    return outliers_mask, threshold


def detect_outliers_isolation_forest(df, contamination=0.1):
    """Detect outliers using Isolation Forest (multivariate)"""
    try:
        from sklearn.ensemble import IsolationForest
        
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.Series(False, index=df.index), "No numeric columns"
        
        # Prepare data
        X = df[numeric_cols].dropna()
        if len(X) < 10:
            return pd.Series(False, index=df.index), "Insufficient data"
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers_pred = iso_forest.fit_predict(X)
        
        # Create mask for all rows
        outliers_mask = pd.Series(False, index=df.index)
        outliers_mask.loc[X.index] = outliers_pred == -1
        
        return outliers_mask, contamination
    except ImportError:
        return pd.Series(False, index=df.index), "Scikit-learn not available"


def handle_outliers_cap(series, lower_bound, upper_bound):
    """Handle outliers by capping to bounds"""
    return series.clip(lower=lower_bound, upper=upper_bound)


def handle_outliers_remove(df, outliers_mask):
    """Handle outliers by removing rows"""
    return df[~outliers_mask]


def handle_outliers_transform_log(series):
    """Handle outliers using log transformation"""
    if (series > 0).all():
        return np.log1p(series)
    else:
        # Add constant to make all values positive
        min_val = series.min()
        if min_val <= 0:
            series_shifted = series - min_val + 1
            return np.log1p(series_shifted)
        return np.log1p(series)


def handle_outliers_transform_sqrt(series):
    """Handle outliers using square root transformation"""
    if (series >= 0).all():
        return np.sqrt(series)
    else:
        # Add constant to make all values non-negative
        min_val = series.min()
        if min_val < 0:
            series_shifted = series - min_val
            return np.sqrt(series_shifted)
        return np.sqrt(series)


def handle_outliers_winsorize(series, limits=(0.05, 0.05)):
    """Handle outliers using winsorization"""
    try:
        from scipy.stats import mstats
        return pd.Series(mstats.winsorize(series, limits=limits), index=series.index)
    except ImportError:
        # Fallback to manual winsorization
        lower_percentile = series.quantile(limits[0])
        upper_percentile = series.quantile(1 - limits[1])
        return series.clip(lower=lower_percentile, upper=upper_percentile)


def visualize_outliers(df, column, outliers_mask, method_name):
    """Create comprehensive outlier visualization"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'Box Plot - {column}',
            f'Histogram - {column}',
            f'Scatter Plot - {column} vs Index',
            f'Q-Q Plot - {column}'
        ],
        specs=[[{"type": "box"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=df[column], name='Normal', boxpoints='outliers'),
        row=1, col=1
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df[column], name='Distribution', nbinsx=30),
        row=1, col=2
    )
    
    # Scatter plot with outliers highlighted
    colors = ['red' if outlier else 'blue' for outlier in outliers_mask]
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[column],
            mode='markers',
            marker=dict(color=colors, size=6),
            name='Data Points',
            text=[f'Index: {i}<br>Value: {v}<br>Outlier: {o}' 
                  for i, v, o in zip(df.index, df[column], outliers_mask)]
        ),
        row=2, col=1
    )
    
    # Q-Q plot
    try:
        from scipy import stats
        sorted_data = np.sort(df[column].dropna())
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_data,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=theoretical_quantiles * sorted_data.std() + sorted_data.mean(),
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=2
        )
    except:
        # If Q-Q plot fails, show a simple scatter
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='markers', name='Q-Q Plot unavailable'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text=f"Outlier Analysis for {column} using {method_name}",
        showlegend=True
    )
    
    return fig


def run_outlier_analysis(df):
    """Main function to run comprehensive outlier analysis"""
    st.subheader("🎯 Outlier Detection & Handling")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found for outlier analysis.")
        return df
    
    st.write(f"**Found {len(numeric_cols)} numeric columns:** {', '.join(numeric_cols)}")
    
    # Column selection
    selected_columns = st.multiselect(
        "Select columns for outlier analysis:",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )
    
    if not selected_columns:
        st.warning("Please select at least one column.")
        return df
    
    # Method selection
    st.subheader("🔍 Detection Methods")
    detection_method = st.selectbox(
        "Choose outlier detection method:",
        [
            "IQR (Interquartile Range)",
            "Z-Score",
            "Modified Z-Score (Robust)",
            "Isolation Forest (Multivariate)"
        ]
    )
    
    # Parameters for different methods
    if detection_method == "Z-Score":
        z_threshold = st.slider("Z-Score threshold:", 2.0, 4.0, 3.0, 0.1)
    elif detection_method == "Modified Z-Score (Robust)":
        modified_z_threshold = st.slider("Modified Z-Score threshold:", 2.5, 5.0, 3.5, 0.1)
    elif detection_method == "Isolation Forest (Multivariate)":
        contamination = st.slider("Expected contamination rate:", 0.01, 0.3, 0.1, 0.01)
    
    # Analysis button
    if st.button("🔍 Detect Outliers"):
        outlier_results = {}
        
        for col in selected_columns:
            st.write(f"\n**Analysis for column: {col}**")
            
            # Detect outliers based on selected method
            if detection_method == "IQR (Interquartile Range)":
                outliers_mask, lower_bound, upper_bound = detect_outliers_iqr(df[col])
                method_info = f"IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
            elif detection_method == "Z-Score":
                outliers_mask, threshold = detect_outliers_zscore(df[col], z_threshold)
                method_info = f"Z-Score threshold: {threshold}"
            elif detection_method == "Modified Z-Score (Robust)":
                outliers_mask, threshold = detect_outliers_modified_zscore(df[col], modified_z_threshold)
                method_info = f"Modified Z-Score threshold: {threshold}"
            elif detection_method == "Isolation Forest (Multivariate)":
                outliers_mask, contamination_rate = detect_outliers_isolation_forest(df[selected_columns], contamination)
                method_info = f"Contamination rate: {contamination_rate}"
            
            outlier_count = outliers_mask.sum()
            outlier_percentage = (outlier_count / len(df)) * 100
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Outliers Found", outlier_count)
            with col2:
                st.metric("Percentage", f"{outlier_percentage:.2f}%")
            with col3:
                st.metric("Method", detection_method.split()[0])
            
            st.write(f"**Method details:** {method_info}")
            
            # Store results
            outlier_results[col] = {
                'mask': outliers_mask,
                'count': outlier_count,
                'percentage': outlier_percentage,
                'method': detection_method
            }
            
            # Visualization
            if outlier_count > 0:
                fig = visualize_outliers(df, col, outliers_mask, detection_method)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show outlier values
                if st.expander(f"📋 View outlier values for {col}"):
                    outlier_data = df[outliers_mask][col].sort_values()
                    st.write(f"**{len(outlier_data)} outlier values:**")
                    st.dataframe(outlier_data.to_frame())
            else:
                st.success(f"No outliers detected in {col}")
        
        # Handle outliers section
        if any(result['count'] > 0 for result in outlier_results.values()):
            st.subheader("🛠️ Outlier Handling")
            
            handling_method = st.selectbox(
                "Choose outlier handling strategy:",
                [
                    "Keep outliers (no action)",
                    "Remove outliers",
                    "Cap outliers (IQR bounds)",
                    "Log transformation",
                    "Square root transformation",
                    "Winsorization (5% each tail)"
                ]
            )
            
            if handling_method != "Keep outliers (no action)":
                if st.button(f"🔧 Apply {handling_method}"):
                    df_processed = df.copy()
                    
                    for col in selected_columns:
                        if outlier_results[col]['count'] > 0:
                            outliers_mask = outlier_results[col]['mask']
                            
                            if handling_method == "Remove outliers":
                                df_processed = handle_outliers_remove(df_processed, outliers_mask)
                                st.success(f"Removed {outliers_mask.sum()} outlier rows")
                                
                            elif handling_method == "Cap outliers (IQR bounds)":
                                _, lower_bound, upper_bound = detect_outliers_iqr(df[col])
                                df_processed[col] = handle_outliers_cap(df_processed[col], lower_bound, upper_bound)
                                st.success(f"Capped outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")
                                
                            elif handling_method == "Log transformation":
                                df_processed[col] = handle_outliers_transform_log(df_processed[col])
                                st.success(f"Applied log transformation to {col}")
                                
                            elif handling_method == "Square root transformation":
                                df_processed[col] = handle_outliers_transform_sqrt(df_processed[col])
                                st.success(f"Applied square root transformation to {col}")
                                
                            elif handling_method == "Winsorization (5% each tail)":
                                df_processed[col] = handle_outliers_winsorize(df_processed[col])
                                st.success(f"Applied winsorization to {col}")
                    
                    # Show before/after comparison
                    st.subheader("📊 Before vs After Comparison")
                    
                    for col in selected_columns:
                        if outlier_results[col]['count'] > 0:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Before - {col}**")
                                st.write(f"Mean: {df[col].mean():.3f}")
                                st.write(f"Std: {df[col].std():.3f}")
                                st.write(f"Min: {df[col].min():.3f}")
                                st.write(f"Max: {df[col].max():.3f}")
                            
                            with col2:
                                st.write(f"**After - {col}**")
                                if col in df_processed.columns:
                                    st.write(f"Mean: {df_processed[col].mean():.3f}")
                                    st.write(f"Std: {df_processed[col].std():.3f}")
                                    st.write(f"Min: {df_processed[col].min():.3f}")
                                    st.write(f"Max: {df_processed[col].max():.3f}")
                    
                    # Update session state with processed data
                    st.session_state['df_processed'] = df_processed
                    st.success("✅ Outlier handling applied! The processed dataset is now available.")
                    
                    # Option to download processed data
                    csv = df_processed.to_csv(index=False)
                    st.download_button(
                        label="📥 Download processed data as CSV",
                        data=csv,
                        file_name="processed_data_outliers_handled.csv",
                        mime="text/csv"
                    )
                    
                    return df_processed
    
    return df


def outlier_summary_report(df):
    """Generate a summary report of outliers across all numeric columns"""
    st.subheader("📋 Outlier Summary Report")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found.")
        return
    
    # Create summary table
    summary_data = []
    
    for col in numeric_cols:
        # IQR method
        outliers_iqr, _, _ = detect_outliers_iqr(df[col])
        iqr_count = outliers_iqr.sum()
        iqr_percentage = (iqr_count / len(df)) * 100
        
        # Z-score method
        outliers_z, _ = detect_outliers_zscore(df[col])
        z_count = outliers_z.sum()
        z_percentage = (z_count / len(df)) * 100
        
        summary_data.append({
            'Column': col,
            'Data Points': len(df[col].dropna()),
            'IQR Outliers': iqr_count,
            'IQR %': f"{iqr_percentage:.2f}%",
            'Z-Score Outliers': z_count,
            'Z-Score %': f"{z_percentage:.2f}%",
            'Mean': f"{df[col].mean():.3f}",
            'Std Dev': f"{df[col].std():.3f}",
            'Min': f"{df[col].min():.3f}",
            'Max': f"{df[col].max():.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Visual summary
    fig = px.bar(
        summary_df, 
        x='Column', 
        y=['IQR Outliers', 'Z-Score Outliers'],
        title="Outlier Count Comparison by Method",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
