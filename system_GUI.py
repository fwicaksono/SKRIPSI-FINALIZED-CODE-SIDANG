import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from scipy.stats import kurtosis, skew
from joblib import load

# Fungsi untuk mengekstrak fitur dari file MSD/MSEED
def extract_features_from_file(file):
    st = read(file)
    features = []
    for trace in st:
        trace_data = trace.data
        mean = np.mean(trace_data)
        std = np.std(trace_data)
        median = np.median(trace_data)
        min_val = np.min(trace_data)
        max_val = np.max(trace_data)
        rng = np.ptp(trace_data)  # Range
        iqr = np.percentile(trace_data, 75) - np.percentile(trace_data, 25)  # Interquartile Range (IQR)
        q1 = np.percentile(trace_data, 25)  # Interquartile First Quarter (Q1)
        q3 = np.percentile(trace_data, 75)  # Interquartile Third Quarter (Q3)
        kurt = kurtosis(trace_data)
        skewness = skew(trace_data)
        
        station = trace.stats.station
        channel = trace.stats.channel
        
        features.append([mean, std, median, min_val, max_val, rng, iqr, q1, q3, kurt, skewness, station, channel])
    return features, st

# Fungsi untuk menggabungkan baris
def merge_rows(df):
    merged_data = {
        'mean_1': [], 'standard_deviation_1': [], 'median_1': [], 'min_value_1': [], 'max_value_1': [], 'range_1': [], 'interquartile_range_1': [],
        'q1_1': [], 'q3_1': [], 'kurtosis_1': [], 'skewness_1': [], 'station_1': [], 'channel_1': [],
        'mean_2': [], 'standard_deviation_2': [], 'median_2': [], 'min_value_2': [], 'max_value_2': [], 'range_2': [], 'interquartile_range_2': [],
        'q1_2': [], 'q3_2': [], 'kurtosis_2': [], 'skewness_2': [], 'station_2': [], 'channel_2': []
    }

    visited = set()
    for i, row1 in df.iterrows():
        if i in visited:
            continue
        found_pair = False
        for j, row2 in df.iterrows():
            if i != j and row1['channel'] == row2['channel'] and row1['station'] != row2['station']:
                for col in df.columns:
                    if col not in ['station', 'channel']:
                        merged_data[f'{col}_1'].append(row1[col])
                        merged_data[f'{col}_2'].append(row2[col])
                merged_data['station_1'].append(row1['station'])
                merged_data['channel_1'].append(row1['channel'])
                merged_data['station_2'].append(row2['station'])
                merged_data['channel_2'].append(row2['channel'])
                visited.add(j)
                found_pair = True
                break
        if not found_pair:
            for col in df.columns:
                if col not in ['station', 'channel']:
                    merged_data[f'{col}_1'].append(row1[col])
                    merged_data[f'{col}_2'].append(None)
            merged_data['station_1'].append(row1['station'])
            merged_data['channel_1'].append(row1['channel'])
            merged_data['station_2'].append(None)
            merged_data['channel_2'].append(None)

    merged_df = pd.DataFrame(merged_data)
    merged_df.fillna(0, inplace=True)
    return merged_df

# Load model dan class mapping
def load_model_and_mapping(model_path):
    model = load(model_path)
    class_mapping = {
        0: 'BEBENG',
        1: 'BOYONG',
        2: 'GENDOL',
    }
    return model, class_mapping

# Lakukan prediksi
def predict_classes(model, class_mapping, merged_df):
    X_pred = merged_df.drop(['station_1', 'channel_1', 'station_2', 'channel_2'], axis=1)
    predictions = model.predict(X_pred)
    predicted_classes = [class_mapping[pred] for pred in predictions]
    merged_df['predicted_class'] = predicted_classes
    return merged_df

# Streamlit app
st.title("Sistem Identifikasi Arah Guguran Merapi Berdasarkan Sinyal Seismik")

# Pilih model
model_path = 'knn_model_to_be_implemented.pkl'

model, class_mapping = load_model_and_mapping(model_path)

uploaded_file = st.file_uploader("Upload MSD/MSEED file", type=["msd", "mseed"])

if uploaded_file is not None:
    features, st_object = extract_features_from_file(uploaded_file)
    
    st.subheader("Visualisasi Sinyal Seismik Pada Trace Pertama")

     # Plot sinyal seismik untuk trace pertama
    trace_data = st_object[0].data
    plt.figure(figsize=(8, 2))
    plt.plot(trace_data, label='Trace 1')
    plt.title('Seismic Signal - Trace 1')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    st.pyplot(plt)
    
    df = pd.DataFrame(features, columns=['mean', 'standard_deviation', 'median', 'min_value', 'max_value', 
                                          'range', 'interquartile_range', 'q1', 'q3', 'kurtosis', 'skewness',
                                          'station', 'channel'])
    
    st.subheader("Extracted Features")
    st.write(df)
    
    merged_df = merge_rows(df)
    
    st.subheader("Merged DataFrame")
    st.write(merged_df)
    
    st.subheader("Merged DataFrame After Cleaning")
    st.write(merged_df)
    
    result_df = predict_classes(model, class_mapping, merged_df)
    
    st.subheader("Prediction Results")
    st.write(result_df)
    
    # Calculate and display prediction percentages
    prediction_counts = result_df['predicted_class'].value_counts()
    total_predictions = len(result_df)
    prediction_percentage = (prediction_counts / total_predictions) * 100
    
    st.subheader("Prediction Percentages")
    st.write(prediction_percentage)
    
    # Centered display of each class percentage
    st.subheader("Prediksi Kelas")
    for class_name, percentage in prediction_percentage.items():
        st.markdown(f"<h2 style='text-align: center;'>{class_name}: {percentage:.2f}%</h2>", unsafe_allow_html=True)