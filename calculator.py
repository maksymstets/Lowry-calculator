
import streamlit as st
import textwrap
import pandas as pd
import numpy as np

from scipy import stats

import matplotlib.pyplot as plt


from fpdf import FPDF
from datetime import datetime
import io         
import tempfile    
import os          

def input_table():
    """
    Initializes and displays Streamlit data editors for user input.
    
    Creates two dynamic dataframes: one for the calibration standards 
    (concentration vs. absorbance) and one for the experimental samples 
    (sample names, dilution factors, and triplicate absorbance readings).
    
    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - cal_data (pd.DataFrame): The calibration curve data.
            - sample_data (pd.DataFrame): The experimental sample data.
    """
    df_cal=pd.DataFrame(
        {'Calibration solution concentration, mg/ml':[None], 
         'Absorbance':[None]
    })
    cal_data = st.data_editor(df_cal, num_rows="dynamic")
    df_samples = pd.DataFrame(
        {'Sample name':[None], 
         "Dilution factor": [1.0],
         'Absorbance1':[None],
         'Absorbance2':[None],
         'Absorbance3':[None]
         })
    sample_data = st.data_editor(df_samples, num_rows="dynamic", column_config={
            'Dilution factor': st.column_config.NumberColumn(
                'Dilution factor',
                default=1.0,
                min_value=0.0
            )
        })
    return cal_data, sample_data



def calibration_curve(cal_data):
    """
    Cleans calibration data and calculates the linear regression parameters.
    
    Extracts valid numeric rows, checks for common data entry errors 
    (e.g., negative absorbance, insufficient points), and computes the 
    line of best fit using scipy.stats.linregress.
    
    Args:
        cal_data (pd.DataFrame): The raw calibration data from the user interface.
        
    Returns:
        tuple: A tuple containing:
            - slope (float): The slope of the regression line.
            - intercept (float): The y-intercept of the regression line.
            - r_squared (float): The coefficient of determination.
            - x (numpy.ndarray): The cleaned concentration values.
            - y (numpy.ndarray): The cleaned absorbance values.
    """
    # 1. Coerce each column to numeric individually
    cal_data['Calibration solution concentration, mg/ml'] = pd.to_numeric(
        cal_data['Calibration solution concentration, mg/ml'], errors='coerce'
    )
    cal_data['Absorbance'] = pd.to_numeric(
        cal_data['Absorbance'], errors='coerce'
    )
    
    # 2. Drop rows that have a NaN in ANY of those specific columns
    cal_data_clean = cal_data.dropna(subset=['Calibration solution concentration, mg/ml', 'Absorbance'])
    
    # 3. Extract the clean arrays
    x = cal_data_clean['Calibration solution concentration, mg/ml'].values
    y = cal_data_clean['Absorbance'].values
    
    # 4. Safety check!  at least 2 points to calculate a line
    if len(x) < 2:
        st.error("Not enough valid data! Please enter at least two complete calibration points.")
        st.stop()  

    if (y < 0).any():
        st.error(" **Warning:** Negative absorbance detected in your standards. Please verify your blanking.")
        st.stop() 

    if len(np.unique(x)) < 2:
        st.error("All calibration values are identical. Please enter standards with different concentrations.")
        st.stop()
    # Calculate regression
    slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
    r_squared = r_value**2
    
    return slope, intercept, r_squared, x, y

def plot_calibration_curve(slope, intercept, r_squared, x, y):
    """
    Renders and displays a scatter plot of the calibration standards with a fitted trendline.
    
    Color-codes the regression line based on the quality of the fit (R-squared value)
    and outputs the plot to the Streamlit app.
    
    Args:
        slope (float): The slope of the regression line.
        intercept (float): The y-intercept of the regression line.
        r_squared (float): The coefficient of determination.
        x (numpy.ndarray): The x-axis values (concentration).
        y (numpy.ndarray): The y-axis values (absorbance).
    """
    # Trendline colour based on R²
    if r_squared >= 0.99:
        line_color = 'green'
        st.success(f"R² = {r_squared:.4f} — Excellent fit")
    elif r_squared >= 0.95:
        line_color = 'orange'
        st.warning(f"R² = {r_squared:.4f} — Acceptable fit")
    else:
        line_color = 'red'
        st.error(f"R² = {r_squared:.4f} — Poor fit")

    # Generate trendline points
    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept

    # Build plot
    fig, ax = plt.subplots()

    ax.scatter(x, y, color='blue', zorder=5, label='Standards')
    ax.plot(x_line, y_line, color=line_color,
            label=f'y = {slope:.4f}x + {intercept:.4f}')

    ax.set_xlabel('Concentration (mg/mL)')
    ax.set_ylabel('Absorbance')
    ax.set_title('Lowry Calibration Curve')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig)

def protein_concentration(sample_data, slope, intercept):
    """
    Calculates the protein concentration for each sample from absorbance readings.
    
    Converts raw absorbance values to concentrations using the standard curve 
    equation (x = (y - intercept) / slope), accounts for dilution factors, 
    and computes the mean and standard deviation across replicates.
    
    Args:
        sample_data (pd.DataFrame): The dataframe containing sample names, 
                                    dilution factors, and absorbance readings.
        slope (float): The slope of the calibration curve.
        intercept (float): The y-intercept of the calibration curve.
        
    Returns:
        tuple: A tuple containing two lists:
            - sample_means (list): Mean protein concentration for each sample.
            - sample_sds (list): Sample standard deviation (ddof=1) for each sample.
    """
    sample_means = []
    sample_sds = []
    negative_abs_found = False
    negative_conc_found = False
    for index, row in sample_data.iterrows():
        abs1 = pd.to_numeric(row['Absorbance1'], errors='coerce')
        abs2 = pd.to_numeric(row['Absorbance2'], errors='coerce')
        abs3 = pd.to_numeric(row['Absorbance3'], errors='coerce')
        dilution_factor = pd.to_numeric(row['Dilution factor'], errors='coerce')
        
        # Calculate concentration for each replicate individually
        concentrations = []
        for absorbance in [abs1, abs2, abs3]:
            if pd.notna(absorbance):
                if absorbance < 0:
                    negative_abs_found = True
                    continue
                if  pd.notna(dilution_factor):
                    concentration = (absorbance - intercept) / slope * dilution_factor
                    if concentration < 0:
                        negative_conc_found = True
                        
                concentrations.append(concentration)
        
        if len(concentrations) > 0:
            sample_means.append(np.mean(concentrations))
            
            if len(concentrations) > 1:
                sample_sds.append(np.std(concentrations, ddof=1))
            else:
                sample_sds.append(None) 
        else:
            sample_means.append(None)
            sample_sds.append(None)
    if negative_abs_found:
        st.warning(" **Warning:** Negative absorbance values detected.")
        
    if negative_conc_found:
        st.warning(" **Warning:** Negative protein concentrations calculated. Check if your absorbance is not lower than the lowest calibration point")
    return sample_means, sample_sds



def protein_concentration_plot(sample_means, sample_sds, sample_data):
    """
    Generates a bar chart visualizing the calculated protein concentrations.
    
    Includes error bars denoting standard deviation. Dynamically adjusts the 
    figure width and x-axis label rotation based on the number of samples.
    
    Args:
        sample_means (list): Mean concentrations for the samples.
        sample_sds (list): Standard deviations for the samples.
        sample_data (pd.DataFrame): Dataframe containing the sample names.
        
    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure object.
    """
    names = sample_data['Sample name'].tolist()
    
    # Width based on number of samples
    num_samples = len(names)
    fig_width = max(8, num_samples * 1.5)  # at least 8, grows with sample count
    
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    wrapped_names = [textwrap.fill(str(name), width=15) for name in names]
    sample_sds_plot = [np.nan if sd is None else sd for sd in sample_sds]
    ax.bar(wrapped_names, sample_means, yerr=sample_sds_plot, capsize=5)
    ax.set_xlabel('Sample name')
    ax.set_ylabel('Concentration (mg/mL)')
    ax.set_title('Protein Concentrations')
    if num_samples > 3:
        plt.xticks(rotation=45, ha='right')
    else:
        plt.xticks(rotation=0, ha='center')
    plt.tight_layout()
    return fig

def protein_concentration_table(sample_data, sample_means, sample_sds):
    """
    Displays a tabular summary of the final protein concentrations.
    
    Compiles the sample names, mean concentrations, and standard deviations 
    into a pandas dataframe and renders it in the Streamlit UI.
    
    Args:
        sample_data (pd.DataFrame): The sample input data containing names.
        sample_means (list): Calculated mean concentrations.
        sample_sds (list): Calculated standard deviations.
    """
    result_df = pd.DataFrame({
        'Sample name': sample_data['Sample name'],
        'Protein concentration (mg/mL)': sample_means,
        'SD': sample_sds
    })
    st.dataframe(result_df)

    
def generate_pdf(slope, intercept, r_squared, sample_data, sample_means, sample_sds, fig):
    """
    Compiles the assay results into a downloadable PDF report.
    
    The report includes the calibration curve formula, a table of the final 
    calculated sample concentrations with standard deviations, and embeds 
    the generated bar chart visualization. Exposes a download button in Streamlit.
    
    Args:
        slope (float): Calibration curve slope.
        intercept (float): Calibration curve y-intercept.
        r_squared (float): Calibration curve R-squared value.
        sample_data (pd.DataFrame): Dataframe containing sample names.
        sample_means (list): Calculated mean concentrations.
        sample_sds (list): Calculated standard deviations.
        fig (matplotlib.figure.Figure): The bar chart figure to embed.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Lowry Protein Assay Report", ln=True, align='C')
    
    # Calibration curve details
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Calibration Curve: y = {slope:.4f}x + {intercept:.4f}, R² = {r_squared:.4f}", ln=True)
    
    # Sample results table
    pdf.cell(0, 10, "Sample Results:", ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, "Sample name", border=1)
    pdf.cell(70, 10, "Protein concentration (mg/mL)", border=1)
    pdf.cell(20, 10, "SD", border=1, ln=True)
    
    pdf.set_font("Arial", '', 12)
    for name, mean, sd in zip(sample_data['Sample name'], sample_means, sample_sds):
        if pdf.get_y() > 250:  # raise threshold back up, 80 is too aggressive
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Lowry Protein Assay Report", ln=True, align='C')
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Sample Results:", ln=True)
            pdf.cell(100, 10, "Sample name", border=1)
            pdf.cell(70, 10, "Protein concentration (mg/mL)", border=1)
            pdf.cell(20, 10, "SD", border=1, ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 10, str(name), border=1)
        pdf.cell(70, 10, f"{mean:.4f}" if mean is not None else "N/A", border=1)
        pdf.cell(20, 10, f"{sd:.4f}" if sd is not None else "N/A", border=1, ln=True)
        
    # Save to temporary file and embed in PDF
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        fig.savefig(tmp.name, bbox_inches='tight')
        pdf.ln(10)
        if pdf.get_y() > 250:
            pdf.add_page()
        pdf.image(tmp.name, x=10, w=180)
    plt.close(fig)
    os.unlink(tmp.name)  # delete temp file after embedding

    # Save and download
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Lowry_Protein_Determination_{timestamp}.pdf"
    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    
    st.download_button(
        label="Download PDF Report",
        data=buffer,
        file_name=filename,
        mime="application/pdf"
    )

def main():
    cal_data, sample_data = input_table()
    if st.button("Calculate"):
        sample_data_clean = sample_data.dropna(subset=['Absorbance1', 'Absorbance2', 'Absorbance3'], how='all').copy()
        slope, intercept, r_squared, x, y = calibration_curve(cal_data)
        plot_calibration_curve(slope, intercept, r_squared, x, y)
        
        
        if not sample_data_clean.empty:
            sample_data_clean['Sample name'] = sample_data_clean['Sample name'].fillna('Unknown')
            print(sample_data_clean['Sample name'])
            sample_means, sample_sds = protein_concentration(sample_data_clean, slope, intercept)
            my_chart = protein_concentration_plot(sample_means, sample_sds, sample_data_clean)
        
       
            st.pyplot(my_chart)
           
            protein_concentration_table(sample_data_clean, sample_means, sample_sds)
            generate_pdf(slope, intercept, r_squared, sample_data_clean, sample_means, sample_sds, my_chart)
        else:
            
            st.info("Calibration curve generated successfully! Add data to the sample table to calculate concentrations.")
if __name__ == "__main__":
    main()
