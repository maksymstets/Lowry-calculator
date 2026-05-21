import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from unittest.mock import patch

from calculator import calibration_curve, protein_concentration, protein_concentration_plot




def make_cal_df(concentrations, absorbances):
    #Build a calibration dataframe from plain lists
    return pd.DataFrame({
        'Calibration solution concentration, mg/ml': concentrations,
        'Absorbance': absorbances
    })


def make_sample_df(names, dilutions, abs1_list, abs2_list, abs3_list):
    #Build a sample dataframe from plain lists
    return pd.DataFrame({
        'Sample name': names,
        'Dilution factor': dilutions,
        'Absorbance1': abs1_list,
        'Absorbance2': abs2_list,
        'Absorbance3': abs3_list
    })


# Calibration line used across protein_concentration tests: y = 0.5x + 0.1
SLOPE = 0.5
INTERCEPT = 0.1



def test_calibration_curve_happy_path():
    #Known linear data should return exact slope and intercept.
    # y = 0.5x + 0.1  =>  slope=0.5, intercept=0.1
    cal = make_cal_df([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    with patch('calculator.st'):
        slope, intercept, r_squared, x, y = calibration_curve(cal)
    assert pytest.approx(slope, abs=1e-4) == 0.5
    assert pytest.approx(intercept, abs=1e-4) == 0.1
    assert pytest.approx(r_squared, abs=1e-4) == 1.0


def test_calibration_curve_perfect_r_squared():
    #Perfectly linear data must give R² = 1.0.
    cal = make_cal_df([0, 1, 2, 3, 4], [0, 2, 4, 6, 8])
    with patch('calculator.st'):
        _, _, r_squared, _, _ = calibration_curve(cal)
    assert pytest.approx(r_squared, abs=1e-6) == 1.0


def test_calibration_curve_fewer_than_two_points_calls_stop():
    #Single data point must trigger st.error and st.stop.
    cal = make_cal_df([0.5], [0.3])
    with patch('calculator.st') as mock_st:
        mock_st.stop.side_effect = SystemExit
        with pytest.raises(SystemExit):
            calibration_curve(cal)
    mock_st.error.assert_called()
    mock_st.stop.assert_called()


def test_calibration_curve_empty_dataframe_calls_stop():
    #Completely empty calibration table must trigger st.error and st.stop.
    cal = make_cal_df([], [])
    with patch('calculator.st') as mock_st:
        mock_st.stop.side_effect = SystemExit
        with pytest.raises(SystemExit):
            calibration_curve(cal)
    mock_st.error.assert_called()
    mock_st.stop.assert_called()


def test_calibration_curve_negative_absorbance_calls_stop():
    #Any negative absorbance in standards must trigger st.error and st.stop.
    cal = make_cal_df([0.0, 0.5, 1.0], [-0.1, 0.3, 0.6])
    with patch('calculator.st') as mock_st:
        mock_st.stop.side_effect = SystemExit
        with pytest.raises(SystemExit):
            calibration_curve(cal)
    mock_st.error.assert_called()
    mock_st.stop.assert_called()


def test_calibration_curve_identical_x_values_calls_stop():
    #Duplicate concentration values must trigger st.error and st.stop.
    cal = make_cal_df([0.5, 0.5, 0.5], [0.2, 0.3, 0.4])
    with patch('calculator.st') as mock_st:
        mock_st.stop.side_effect = SystemExit
        with pytest.raises(SystemExit):
            calibration_curve(cal)
    mock_st.error.assert_called()
    mock_st.stop.assert_called()


def test_calibration_curve_non_numeric_row_is_dropped():
    #A string in one cell should be dropped; regression runs on remaining rows.
    cal = make_cal_df([0.0, 0.5, 'abc', 1.0], [0.1, 0.35, 0.3, 0.6])
    with patch('calculator.st'):
        slope, intercept, r_squared, x, y = calibration_curve(cal)
    assert len(x) == 3
    assert len(y) == 3


def test_calibration_curve_returns_equal_length_arrays():
    #x and y arrays returned must always be the same length.
    cal = make_cal_df([0.0, 0.25, 0.5, 0.75, 1.0],
                      [0.05, 0.18, 0.32, 0.47, 0.61])
    with patch('calculator.st'):
        _, _, _, x, y = calibration_curve(cal)
    assert len(x) == len(y)



def test_protein_concentration_three_replicates_no_dilution():
    #Mean and SD calculated correctly for three replicates with dilution=1.
    # abs 0.35 => conc 0.5;  abs 0.40 => conc 0.6;  abs 0.45 => conc 0.7
    df = make_sample_df(['S1'], [1.0], [0.35], [0.40], [0.45])
    with patch('calculator.st'):
        means, sds = protein_concentration(df, SLOPE, INTERCEPT)
    assert pytest.approx(means[0], abs=1e-4) == 0.6
    assert sds[0] is not None
    assert sds[0] > 0


def test_protein_concentration_dilution_factor_doubles_result():
    #Dilution factor of 2 must double the calculated concentration.
    df_undiluted = make_sample_df(['S1'], [1.0], [0.35], [0.40], [0.45])
    df_diluted   = make_sample_df(['S1'], [2.0], [0.35], [0.40], [0.45])
    with patch('calculator.st'):
        means_u, _ = protein_concentration(df_undiluted, SLOPE, INTERCEPT)
        means_d, _ = protein_concentration(df_diluted,   SLOPE, INTERCEPT)
    assert pytest.approx(means_d[0], abs=1e-4) == means_u[0] * 2


def test_protein_concentration_single_replicate_sd_is_none():
    #One valid replicate must return a mean but SD must be None.
    df = make_sample_df(['S1'], [1.0], [0.40], [None], [None])
    with patch('calculator.st'):
        means, sds = protein_concentration(df, SLOPE, INTERCEPT)
    assert means[0] is not None
    assert sds[0] is None


def test_protein_concentration_all_replicates_nan_returns_none():
    #All-NaN replicates must return None for both mean and SD
    df = make_sample_df(['S1'], [1.0], [None], [None], [None])
    with patch('calculator.st'):
        means, sds = protein_concentration(df, SLOPE, INTERCEPT)
    assert means[0] is None
    assert sds[0] is None


def test_protein_concentration_negative_absorbance_skipped_and_warning_fired():
    #Negative absorbance replicates must be excluded from the mean and trigger a warning
    df = make_sample_df(['S1'], [1.0], [-0.1], [0.40], [0.45])
    with patch('calculator.st') as mock_st:
        means, sds = protein_concentration(df, SLOPE, INTERCEPT)
    mock_st.warning.assert_called()
    expected_mean = np.mean([(0.40 - INTERCEPT) / SLOPE,
                              (0.45 - INTERCEPT) / SLOPE])
    assert pytest.approx(means[0], abs=1e-4) == expected_mean


def test_protein_concentration_negative_concentration_included_and_warning_fired():
    #Negative concentration must be included in the mean unchanged and trigger a warning.
    # abs 0.05 => conc (0.05-0.1)/0.5 = -0.1  (below intercept)
    df = make_sample_df(['S1'], [1.0], [0.05], [0.40], [0.45])
    with patch('calculator.st') as mock_st:
        means, sds = protein_concentration(df, SLOPE, INTERCEPT)
    mock_st.warning.assert_called()
    expected_mean = np.mean([(0.05 - INTERCEPT) / SLOPE,
                              (0.40 - INTERCEPT) / SLOPE,
                              (0.45 - INTERCEPT) / SLOPE])
    assert pytest.approx(means[0], abs=1e-4) == expected_mean


def test_protein_concentration_multiple_samples_correct_length():
    #Output lists must have one entry per input row
    df = make_sample_df(
        ['S1', 'S2', 'S3'],
        [1.0,  1.0,  2.0],
        [0.30, 0.40, 0.50],
        [0.32, 0.42, 0.52],
        [0.31, 0.41, 0.51]
    )
    with patch('calculator.st'):
        means, sds = protein_concentration(df, SLOPE, INTERCEPT)
    assert len(means) == 3
    assert len(sds) == 3


def test_protein_concentration_lists_always_same_length():
    #means and sds must always be the same length regardless of replicate count.
    df = make_sample_df(
        ['full', 'one',  'none'],
        [1.0,    1.0,    1.0],
        [0.30,   0.40,   None],
        [0.32,   None,   None],
        [0.31,   None,   None]
    )
    with patch('calculator.st'):
        means, sds = protein_concentration(df, SLOPE, INTERCEPT)
    assert len(means) == len(sds)


def test_protein_concentration_two_replicates_sd_calculated():
    #Two valid replicates must produce a non-None SD.
    df = make_sample_df(['S1'], [1.0], [0.40], [0.45], [None])
    with patch('calculator.st'):
        means, sds = protein_concentration(df, SLOPE, INTERCEPT)
    assert sds[0] is not None
    assert sds[0] > 0



def test_plot_returns_matplotlib_figure():
    #Function must return a matplotlib Figure instance.
    df = pd.DataFrame({'Sample name': ['S1', 'S2']})
    fig = protein_concentration_plot([0.3, 0.5], [0.01, 0.02], df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_correct_number_of_bars():
    #Number of bars must equal number of samples.
    df = pd.DataFrame({'Sample name': ['A', 'B', 'C']})
    fig = protein_concentration_plot([0.2, 0.4, 0.6], [0.01, 0.02, 0.01], df)
    ax = fig.axes[0]
    assert len(ax.patches) == 3
    plt.close(fig)


def test_plot_none_sd_does_not_crash():
    #None in sds list (single replicate) must not raise an error.
    df = pd.DataFrame({'Sample name': ['S1', 'S2']})
    fig = protein_concentration_plot([0.3, 0.5], [None, 0.02], df)
    plt.close(fig)


def test_plot_all_none_sds_does_not_crash():
    #All-None sds must not raise an error.
    df = pd.DataFrame({'Sample name': ['S1', 'S2']})
    fig = protein_concentration_plot([0.3, 0.5], [None, None], df)
    plt.close(fig)


def test_plot_single_sample_renders():
    #A single sample must render without error.
    df = pd.DataFrame({'Sample name': ['OnlySample']})
    fig = protein_concentration_plot([0.4], [None], df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_axis_labels_set():
    #X and Y axis labels must be present.
    df = pd.DataFrame({'Sample name': ['S1']})
    fig = protein_concentration_plot([0.3], [0.01], df)
    ax = fig.axes[0]
    assert ax.get_xlabel() != ''
    assert ax.get_ylabel() != ''
    plt.close(fig)