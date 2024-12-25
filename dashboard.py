import streamlit as st
import pandas as pd
import numpy as np
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson


def convert_answers(row):
    row['Jenis Kelamin'] = 1 if row['Jenis Kelamin'] == 'Laki-laki' else 2
    
    if row['Berapa Umur Anda'] == ' <25 tahun: Usia muda':
        row['Berapa Umur Anda'] = 1
    elif row['Berapa Umur Anda'] == '25-45 tahun: Usia dewasa':
        row['Berapa Umur Anda'] = 2
    else:
        row['Berapa Umur Anda'] = 3
    
    daerah_mapping = {
        'Dekat pusat kota atau area komersial': 1,
        'Dekat area perumahan atau pemukiman padat': 2,
        'Dekat fasilitas umum (sekolah, kantor, rumah sakit)': 3,
        'Dekat dengan akses transportasi umum (misalnya, stasiun kereta, halte bus)': 4,
        'Dekat dengan area hijau, seperti taman atau ruang terbuka': 5
    }
    row['Bagaimana karakteristik umum lingkungan di sekitar tempat tinggal Anda?'] = daerah_mapping.get(row['Bagaimana karakteristik umum lingkungan di sekitar tempat tinggal Anda?'], 0)

    frekuensi_mapping = {
        'Sering': 4,
        '3-5 Kali seminggu': 3,
        '1-2 Kali seminggu': 2,
        'Jarang': 1
    }
    row['Frekuensi Berkendara di jalan raya'] = frekuensi_mapping.get(row['Frekuensi Berkendara di jalan raya'], 0)
    
    kendaraan_mapping = {
        'Kendaraan Roda 2': 1,
        'Kendaraan Roda 4': 2,
        'Yang lain': 3
    }
    row['Jenis Kendaraan yang Sering Anda Gunakan'] = kendaraan_mapping.get(row['Jenis Kendaraan yang Sering Anda Gunakan'], 3)

    perilaku_mapping = {
        'Sangat sering': 4,
        'Sering': 3,
        'Jarang': 2,
        'Tidak Pernah': 1
    }

    perilaku_mapping2 = {
        'Sangat Sering': 4,
        'Sering': 3,
        'Jarang': 2,
        'Tidak Pernah': 1
    }

    perilaku_mapping3 = {
        'Selalu': 4,
        'Sering': 3,
        'Jarang': 2,
        'Tidak Pernah': 1
    }
    perilaku_mapping4 = {
        'sangat sering': 4,
        'Sering': 3,
        'Jarang': 2,
        'Tidak Pernah': 1
    }
    row['Seberapa sering Anda menggunakan ponsel (untuk telepon, chatting, atau navigasi) saat berkendara?'] = perilaku_mapping.get(row['Seberapa sering Anda menggunakan ponsel (untuk telepon, chatting, atau navigasi) saat berkendara?'], 0)
    row['Seberapa sering Anda menaati batas kecepatan yang ditentukan (Dalam Perkotaan 50 Km/Jam)?'] = perilaku_mapping.get(row['Seberapa sering Anda menaati batas kecepatan yang ditentukan (Dalam Perkotaan 50 Km/Jam)?'], 0)
    row['Seberapa sering Anda menerobos lampu merah atau melanggar aturan lalu lintas lainnya?'] = perilaku_mapping2.get(row['Seberapa sering Anda menerobos lampu merah atau melanggar aturan lalu lintas lainnya?'], 0)
    row['Apakah Anda selalu menggunakan helm/sabuk pengaman saat berkendara?'] = perilaku_mapping3.get(row['Apakah Anda selalu menggunakan helm/sabuk pengaman saat berkendara?'], 0)

    efektif = {
        'Sangat Efektif': 4,
        'Efektif': 3,
        'Tidak Efektif': 2,
        'Sangat Tidak Efektif': 1
    }
    row['Seberapa efektif rambu-rambu keselamatan di daerah Anda dalam mengurangi risiko kecelakaan? '] = efektif.get(row['Seberapa efektif rambu-rambu keselamatan di daerah Anda dalam mengurangi risiko kecelakaan? '], 0)

    baik = {
        'Sangat Baik': 4,
        'Baik': 3,
        'Tidak Baik': 2,
        'Sangat Tidak Baik': 1
    }
    row['Bagaimana penilaian Anda terhadap kecepatan respons pihak berwenang dalam menangani insiden lalu lintas? '] = baik.get(row['Bagaimana penilaian Anda terhadap kecepatan respons pihak berwenang dalam menangani insiden lalu lintas? '], 0)
    row['Bagaimana penilaian Anda terhadap sistem rekayasa lalu lintas (seperti penutupan jalan atau pengaturan satu arah) dalam mengurangi kemacetan?  '] = baik.get(row['Bagaimana penilaian Anda terhadap sistem rekayasa lalu lintas (seperti penutupan jalan atau pengaturan satu arah) dalam mengurangi kemacetan?  '], 0)

    keselamatan_mapping = {
        'Sangat Aman': 4,
        'Aman': 3,
        'Tidak Aman': 2,
        'Sangat Tidak Aman': 1
    }

    keselamatan_mapping2 = {
        'Sangat Memadai': 4,
        'Memadai': 3,
        'Kurang memadai': 2,
        'Sangat Tidak Memadai': 1
    }
    row['Apakah Anda merasa aman berkendara atau berjalan di jalan Kota Surabaya ?'] = keselamatan_mapping.get(row['Apakah Anda merasa aman berkendara atau berjalan di jalan Kota Surabaya ?'], 0)
    row['Apakah fasilitas keselamatan seperti rambu dan marka jalan sudah memadai?'] = keselamatan_mapping2.get(row['Apakah fasilitas keselamatan seperti rambu dan marka jalan sudah memadai?'], 0)
    row['Seberapa sering Anda merasa kondisi jalan yang Anda lewati berpotensi membahayakan pengguna jalan? '] = perilaku_mapping4.get(row['Seberapa sering Anda merasa kondisi jalan yang Anda lewati berpotensi membahayakan pengguna jalan? '], 0)

    manajemen_mapping = {
        'Konsisten': 4,
        'Cukup Konsisten': 3,
        'Kurang Konsisten': 2,
        'Tidak Konsisten': 1
    }
    row['Apakah Anda merasa polisi lalu lintas konsisten dalam menegakkan peraturan?'] = manajemen_mapping.get(row['Apakah Anda merasa polisi lalu lintas konsisten dalam menegakkan peraturan?'], 0)
    return row


# MAIN NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Validity Test", "Reliability Test", "Sampling", "Descriptive Statistics", "Regression Analysis", "Regression Diagnostics", "Factor Analysis"])

@st.cache_data
def load_data():
    df106 = pd.read_excel("survei.xlsx")
    df30 = pd.read_excel("bismillahfix.xlsx")
    return df106, df30

df, df2 = load_data()

df = df.apply(convert_answers, axis=1)
df = df.drop(columns=['Timestamp', 'Email Address', 'Apa saran Anda untuk mengurangi Kemacetan Di Surabaya? ', 'Jenis Kelamin', 'Berapa Umur Anda', 'Bagaimana karakteristik umum lingkungan di sekitar tempat tinggal Anda?', 'Frekuensi Berkendara di jalan raya', 'Jenis Kendaraan yang Sering Anda Gunakan'])

df2 = df2.apply(convert_answers, axis=1)
df2['total Perilaku Pengguna Jalan '] = df2['Seberapa sering Anda menggunakan ponsel (untuk telepon, chatting, atau navigasi) saat berkendara?'] + df2['Seberapa sering Anda menaati batas kecepatan yang ditentukan (Dalam Perkotaan 50 Km/Jam)?'] + df2['Seberapa sering Anda menerobos lampu merah atau melanggar aturan lalu lintas lainnya?'] + df2['Apakah Anda selalu menggunakan helm/sabuk pengaman saat berkendara?']
df2['total Persepsi Keselamatan dan Kecelakaan'] = df2['Apakah Anda merasa aman berkendara atau berjalan di jalan Kota Surabaya ?'] + df2['Seberapa sering Anda merasa kondisi jalan yang Anda lewati berpotensi membahayakan pengguna jalan? '] + df2['Apakah fasilitas keselamatan seperti rambu dan marka jalan sudah memadai?'] + df2['Seberapa efektif rambu-rambu keselamatan di daerah Anda dalam mengurangi risiko kecelakaan? ']
df2['total Efektivitas Manajemen Lalu Lintas dan Kemacetan'] = df2['Bagaimana penilaian Anda terhadap sistem rekayasa lalu lintas (seperti penutupan jalan atau pengaturan satu arah) dalam mengurangi kemacetan?  '] + df2['Bagaimana penilaian Anda terhadap kecepatan respons pihak berwenang dalam menangani insiden lalu lintas? '] + df2['Apakah Anda merasa polisi lalu lintas konsisten dalam menegakkan peraturan?']
df2 = df2.drop(columns=['Timestamp', 'Email Address', 'Apa saran Anda untuk mengurangi Kemacetan Di Surabaya? ', 'Jenis Kelamin', 'Berapa Umur Anda', 'Bagaimana karakteristik umum lingkungan di sekitar tempat tinggal Anda?', 'Frekuensi Berkendara di jalan raya', 'Jenis Kendaraan yang Sering Anda Gunakan'])


# VALIDITY TEST PAGE
if page == "Validity Test":
    st.title("Traffic Safety Perception Analysis - Validity Test")
    
    def lwise_corr_pvalues(df, method="pearson"):
        df = df.dropna(how='any')._get_numeric_data()
        dfcols = pd.DataFrame(columns=df.columns)
        rvalues = dfcols.transpose().join(dfcols, how='outer')
        pvalues = dfcols.transpose().join(dfcols, how='outer')
        length = str(len(df))

        if method is None or method == "pearson":
            test = stats.pearsonr
            test_name = "Pearson"
        elif method == "spearman":
            test = stats.spearmanr
            test_name = "Spearman Rank"
        elif method == "kendall":
            test = stats.kendalltau
            test_name = "Kendall's Tau-b"
        else:
            raise ValueError(f"Method {method} not recognized. Please use 'pearson', 'spearman', or 'kendall'.")

        for r in df.columns:
            for c in df.columns:
                rvalues.loc[r, c] = round(test(df[r], df[c])[0], 4)
                pvalues.loc[r, c] = float(format(test(df[r], df[c])[1], '.4f'))
        
        return rvalues, pvalues, length, test_name
    
    def plot_correlation_heatmap(df, method="pearson"):
        df_cor = df.copy()
        df_cor.columns = [f"x{i+1}" for i in range(len(df_cor.columns))]
        
        if method == "pearson":
            corr_matrix = df_cor.corr()
        elif method == "spearman":
            corr_matrix = df_cor.corr(method='spearman')
        else:
            corr_matrix = df_cor.corr(method='kendall')
        
        fig = plt.figure(figsize=(30, 25))
        sns.heatmap(corr_matrix, 
                    cmap='coolwarm',
                    linewidths=0.75, 
                    linecolor='black', 
                    cbar=True, 
                    vmin=-1, 
                    vmax=1, 
                    annot=True, 
                    annot_kws={'size': 10, 'color': 'black'})
        
        plt.tick_params(labelsize=10, rotation=45)
        plt.title(f'{method.capitalize()} Correlation of Questionnaire Instrument', size=14)
        
        return fig
    
    st.subheader("Data Preview")
    st.dataframe(df2)
    
    correlation_method = st.selectbox(
        "Select correlation method:",
        ["pearson", "spearman", "kendall"],
        index=0
    )
    
    if st.button("Calculate Correlation"):
        numeric_df = df2._get_numeric_data()
        if numeric_df.empty:
            st.error("No numeric columns found in the dataset!")
        else:
            rvalues, pvalues, length, test_name = lwise_corr_pvalues(df2, method=correlation_method)
            
            st.subheader("Results")
            st.write(f"Correlation test conducted using list-wise deletion")
            st.write(f"Total observations used: {length}")
            st.write(f"Test method: {test_name}")
            
            st.write("**Correlation Heatmap**")
            fig = plot_correlation_heatmap(df2, method=correlation_method)
            st.pyplot(fig)
            
            st.write("**P-Values** (highlighted yellow if < 0.05)")
            def highlight_significant(val):
                color = 'yellow' if val < 0.05 else ''
                return f'background-color: {color}'
            
            st.dataframe(
                pvalues.style.applymap(highlight_significant)
            )


# RELIABILITY TEST PAGE
elif page == "Reliability Test":
    st.title("Traffic Safety Perception Analysis - Reliability Test")
    
    cronbach_alpha = pg.cronbach_alpha(data=df2)
    st.write("Cronbach's Alpha:", cronbach_alpha)

    for col in df2.columns[2:-1]:
        df2[col] = df2[col].astype('category')

    sd = df2.describe(include='category')
    st.dataframe(sd)


# SAMPLING PAGE
elif page == "Sampling":
    st.title("Traffic Safety Perception Analysis - Sampling")
    
    def find_optimal_sample_size(data, target_column, min_size=10, max_size=100, step=5):
        X = data.drop(columns=[target_column])
        y = data[target_column]
        sample_sizes = range(min_size, min(max_size, len(data)), step)
        r2_scores = []

        for size in sample_sizes:
            X_sample, _, y_sample, _ = train_test_split(X, y, train_size=size, random_state=42)
            model = LinearRegression()
            model.fit(X_sample, y_sample)
            y_pred = model.predict(X)
            score = r2_score(y, y_pred)
            r2_scores.append(score)

        st.subheader("Model Performance vs Sample Size")
        plt.figure(figsize=(10, 6))
        plt.plot(sample_sizes, r2_scores, marker='o')
        plt.title("Model Performance vs Sample Size")
        plt.xlabel("Sample Size")
        plt.ylabel("R-squared Score")
        plt.grid(True)
        st.pyplot(plt.gcf())

        return {"sample_sizes": list(sample_sizes), "r2_scores": r2_scores}
    
    def calculate_sample_size(population_size, sample_size):
        return int(population_size * sample_size)

    def perform_random_sampling(data, sample_size):
        return data.sample(n=sample_size, random_state=42)
    
    sample_size_info = find_optimal_sample_size(df, 'Apakah Anda merasa aman berkendara atau berjalan di jalan Kota Surabaya ?')

    total_population = len(df)
    sample_population = 0.4  # 40%
    required_sample_size = calculate_sample_size(total_population, sample_population)

    st.write(f"Total population size: {total_population}")
    st.write(f"Required sample size: {required_sample_size}")


# DESCRIPTIVE STATISTICS PAGE
elif page == "Descriptive Statistics":
    st.title("Traffic Safety Perception Analysis - Descriptive Statistics")
    
    # SAMPLED DF
    sampled_df = df.sample(frac=0.4, random_state=42)
    Y = sampled_df['Apakah Anda merasa aman berkendara atau berjalan di jalan Kota Surabaya ?']
    X = sampled_df[[
        'Seberapa sering Anda menggunakan ponsel (untuk telepon, chatting, atau navigasi) saat berkendara?',
        'Seberapa sering Anda menaati batas kecepatan yang ditentukan (Dalam Perkotaan 50 Km/Jam)?',
        'Seberapa sering Anda menerobos lampu merah atau melanggar aturan lalu lintas lainnya?',
        'Apakah Anda selalu menggunakan helm/sabuk pengaman saat berkendara?',
        'Seberapa efektif rambu-rambu keselamatan di daerah Anda dalam mengurangi risiko kecelakaan? ',
        'Bagaimana penilaian Anda terhadap kecepatan respons pihak berwenang dalam menangani insiden lalu lintas? ',
        'Bagaimana penilaian Anda terhadap sistem rekayasa lalu lintas (seperti penutupan jalan atau pengaturan satu arah) dalam mengurangi kemacetan?  ',
        'Apakah fasilitas keselamatan seperti rambu dan marka jalan sudah memadai?',
        'Seberapa sering Anda merasa kondisi jalan yang Anda lewati berpotensi membahayakan pengguna jalan? ',
        'Apakah Anda merasa polisi lalu lintas konsisten dalam menegakkan peraturan?'
    ]]
    
    st.subheader("Sampling Validation")
    st.write(f"Original data shape: {df.shape}")
    st.write(f"Sampled data shape: {sampled_df.shape}")

    st.subheader("Descriptive Statistics Comparison")
    st.write("**Original Data Summary:**")
    st.write(df[X.columns].describe())
    
    st.write("**Sampled Data Summary:**")
    st.write(X.describe())

    st.subheader("Distribution Comparison")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.histplot(df['Apakah Anda merasa aman berkendara atau berjalan di jalan Kota Surabaya ?'], bins=20, ax=ax[0])
    ax[0].set_title('Original Data Distribution')
    ax[0].set_xlabel('Safety Perception')

    sns.histplot(Y, bins=20, ax=ax[1])
    ax[1].set_title('Sampled Data Distribution')
    ax[1].set_xlabel('Safety Perception')

    plt.tight_layout()
    st.pyplot(fig)


# REGRESSION ANALYSIS PAGE
elif page == "Regression Analysis":
    st.title("Traffic Safety Perception Analysis - Regression Analysis")

    # SAMPLED DF
    sampled_df = df.sample(frac=0.4, random_state=42)
    Y = sampled_df['Apakah Anda merasa aman berkendara atau berjalan di jalan Kota Surabaya ?']
    X = sampled_df[[
        'Seberapa sering Anda menggunakan ponsel (untuk telepon, chatting, atau navigasi) saat berkendara?',
        'Seberapa sering Anda menaati batas kecepatan yang ditentukan (Dalam Perkotaan 50 Km/Jam)?',
        'Seberapa sering Anda menerobos lampu merah atau melanggar aturan lalu lintas lainnya?',
        'Apakah Anda selalu menggunakan helm/sabuk pengaman saat berkendara?',
        'Seberapa efektif rambu-rambu keselamatan di daerah Anda dalam mengurangi risiko kecelakaan? ',
        'Bagaimana penilaian Anda terhadap kecepatan respons pihak berwenang dalam menangani insiden lalu lintas? ',
        'Bagaimana penilaian Anda terhadap sistem rekayasa lalu lintas (seperti penutupan jalan atau pengaturan satu arah) dalam mengurangi kemacetan?  ',
        'Apakah fasilitas keselamatan seperti rambu dan marka jalan sudah memadai?',
        'Seberapa sering Anda merasa kondisi jalan yang Anda lewati berpotensi membahayakan pengguna jalan? ',
        'Apakah Anda merasa polisi lalu lintas konsisten dalam menegakkan peraturan?'
    ]]

    # OLS REGRESSION SUMMARY
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train_sm = sm.add_constant(X_train, has_constant='add')
    X_test_sm = sm.add_constant(X_test, has_constant='add')
    model_sm = sm.OLS(y_train, X_train_sm).fit()

    st.subheader("OLS Regression Summary")
    st.text(model_sm.summary())

    # REGRESSION MODEL
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    var_names = {
        'X0': 'Penggunaan ponsel',
        'X1': 'Kepatuhan batas kecepatan',
        'X2': 'Pelanggaran lampu merah',
        'X3': 'Penggunaan helm/sabuk',
        'X4': 'Efektivitas rambu',
        'X5': 'Respons pihak berwenang',
        'X6': 'Sistem rekayasa lalu lintas',
        'X7': 'Fasilitas keselamatan',
        'X8': 'Kondisi jalan berbahaya',
        'X9': 'Konsistensi polisi'
    }

    st.subheader("Regression Model")
    regression_eq = "Y = "
    for i, coef in enumerate(model.coef_):
        if i > 0:
            regression_eq += " + " if coef >= 0 else " - "
        regression_eq += f"{abs(coef):.6f}(X{i})"
    regression_eq += f" + {model.intercept_:.6f}"
    st.text(regression_eq)

    st.subheader("Variable Interpretation")
    var_df = pd.DataFrame(list(var_names.items()), columns=['Variable', 'Interpretasi'])
    st.table(var_df)

    coef_df = pd.DataFrame({
        'Variabel': list(var_names.values()),
        'Koefisien': model.coef_,
        'Pengaruh': ['Positif' if c > 0 else 'Negatif' for c in model.coef_]
    })
    coef_df['Abs_Koefisien'] = abs(coef_df['Koefisien'])
    coef_df = coef_df.sort_values('Abs_Koefisien', ascending=False)

    st.subheader("Summary of Variable Effects")
    st.table(coef_df[['Variabel', 'Koefisien', 'Pengaruh']])

    # PERFORMANCE METRICS
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.subheader("Model Performance Metrics:")
    metrics_df = pd.DataFrame({
        'Metric': ['R-squared Score', 'Mean Squared Error', 'Root Mean Squared Error'],
        'Value': [f"{r2:.4f}", f"{mse:.4f}", f"{rmse:.4f}"]
    })
    st.table(metrics_df)

    # FEATURE IMPORTANCE ANALYSIS
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

    st.subheader("Feature Importance:")
    st.write(feature_importance)

    st.subheader("Feature Importance (Absolute Coefficient Values)")
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Abs_Coefficient', y='Feature', data=feature_importance)
    plt.title('Feature Importance (Absolute Coefficient Values)')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # PREDICTED VS ACTUAL VISUALIZATION
    st.subheader("Actual vs Predicted Safety Perception")
    plt.figure(figsize=(14, 8))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Safety Perception')
    plt.ylabel('Predicted Safety Perception')
    plt.title('Actual vs Predicted Safety Perception')
    plt.tight_layout()
    st.pyplot(plt.gcf())


# REGRESSION DIAGNOSTICS PAGE
elif page == "Regression Diagnostics":
    st.title("Traffic Safety Perception Analysis - Regression Diagnostics")

    # SAMPLED DF
    sampled_df = df.sample(frac=0.4, random_state=42)
    Y = sampled_df['Apakah Anda merasa aman berkendara atau berjalan di jalan Kota Surabaya ?']
    X = sampled_df[[
        'Seberapa sering Anda menggunakan ponsel (untuk telepon, chatting, atau navigasi) saat berkendara?',
        'Seberapa sering Anda menaati batas kecepatan yang ditentukan (Dalam Perkotaan 50 Km/Jam)?',
        'Seberapa sering Anda menerobos lampu merah atau melanggar aturan lalu lintas lainnya?',
        'Apakah Anda selalu menggunakan helm/sabuk pengaman saat berkendara?',
        'Seberapa efektif rambu-rambu keselamatan di daerah Anda dalam mengurangi risiko kecelakaan? ',
        'Bagaimana penilaian Anda terhadap kecepatan respons pihak berwenang dalam menangani insiden lalu lintas? ',
        'Bagaimana penilaian Anda terhadap sistem rekayasa lalu lintas (seperti penutupan jalan atau pengaturan satu arah) dalam mengurangi kemacetan?  ',
        'Apakah fasilitas keselamatan seperti rambu dan marka jalan sudah memadai?',
        'Seberapa sering Anda merasa kondisi jalan yang Anda lewati berpotensi membahayakan pengguna jalan? ',
        'Apakah Anda merasa polisi lalu lintas konsisten dalam menegakkan peraturan?'
    ]]

    def regression_diagnostics(X, y):
        st.subheader("Normality Test")
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        residuals = model.resid

        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        plt.title("Q-Q Plot Residual")
        st.pyplot(fig)
        
        stat, p_value = stats.shapiro(residuals)
        st.write(f"Shapiro-Wilk test: statistic={stat:.4f}, p-value={p_value:.4f}")
        st.write("Kesimpulan: Residual", "berdistribusi normal" if p_value > 0.05 else "tidak berdistribusi normal")
        
        st.subheader("Multicolinearity Test")
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.table(vif_data)
        st.write("Kesimpulan: VIF > 10 mengindikasikan multikolinearitas")
        
        st.subheader("Heteroscedasticity Test")
        bp_test = het_breuschpagan(residuals, X_with_const)
        st.write(f"Breusch-Pagan test: statistic={bp_test[0]:.4f}, p-value={bp_test[1]:.4f}")
        st.write("Kesimpulan: Data", "homoskedastis" if bp_test[1] > 0.05 else "heteroskedastis")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(model.fittedvalues, residuals)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual vs Fitted Values Plot")
        st.pyplot(fig)
        
        st.subheader("Autocorrelation Test")
        dw_stat = durbin_watson(residuals)
        st.write(f"Durbin-Watson statistic: {dw_stat:.4f}")
        st.write("Kesimpulan: Nilai mendekati 2 mengindikasikan tidak ada autokorelasi")

    regression_diagnostics(X, Y)


# FACTOR ANALYSIS PAGE
elif page == "Factor Analysis":
    st.title("Traffic Safety Perception Analysis - Factor Analysis")
    
    # SAMPLED DF
    sampled_df = df.sample(frac=0.4, random_state=42)
    Y = sampled_df['Apakah Anda merasa aman berkendara atau berjalan di jalan Kota Surabaya ?']
    X = sampled_df[[
        'Seberapa sering Anda menggunakan ponsel (untuk telepon, chatting, atau navigasi) saat berkendara?',
        'Seberapa sering Anda menaati batas kecepatan yang ditentukan (Dalam Perkotaan 50 Km/Jam)?',
        'Seberapa sering Anda menerobos lampu merah atau melanggar aturan lalu lintas lainnya?',
        'Apakah Anda selalu menggunakan helm/sabuk pengaman saat berkendara?',
        'Seberapa efektif rambu-rambu keselamatan di daerah Anda dalam mengurangi risiko kecelakaan? ',
        'Bagaimana penilaian Anda terhadap kecepatan respons pihak berwenang dalam menangani insiden lalu lintas? ',
        'Bagaimana penilaian Anda terhadap sistem rekayasa lalu lintas (seperti penutupan jalan atau pengaturan satu arah) dalam mengurangi kemacetan?  ',
        'Apakah fasilitas keselamatan seperti rambu dan marka jalan sudah memadai?',
        'Seberapa sering Anda merasa kondisi jalan yang Anda lewati berpotensi membahayakan pengguna jalan? ',
        'Apakah Anda merasa polisi lalu lintas konsisten dalam menegakkan peraturan?'
    ]]
    feature_names = [
        'Seberapa sering Anda menggunakan ponsel (untuk telepon, chatting, atau navigasi) saat berkendara?',
        'Seberapa sering Anda menaati batas kecepatan yang ditentukan (Dalam Perkotaan 50 Km/Jam)?',
        'Seberapa sering Anda menerobos lampu merah atau melanggar aturan lalu lintas lainnya?',
        'Apakah Anda selalu menggunakan helm/sabuk pengaman saat berkendara?',
        'Seberapa efektif rambu-rambu keselamatan di daerah Anda dalam mengurangi risiko kecelakaan? ',
        'Bagaimana penilaian Anda terhadap kecepatan respons pihak berwenang dalam menangani insiden lalu lintas? ',
        'Bagaimana penilaian Anda terhadap sistem rekayasa lalu lintas (seperti penutupan jalan atau pengaturan satu arah) dalam mengurangi kemacetan?  ',
        'Apakah fasilitas keselamatan seperti rambu dan marka jalan sudah memadai?',
        'Seberapa sering Anda merasa kondisi jalan yang Anda lewati berpotensi membahayakan pengguna jalan? ',
        'Apakah Anda merasa polisi lalu lintas konsisten dalam menegakkan peraturan?'
    ]
    
    def find_surrogate_variables(fa, feature_names, n_factors):
        loadings = pd.DataFrame(
            fa.loadings_,
            columns=[f'Factor{i+1}' for i in range(n_factors)],
            index=feature_names
        )
        surrogate_vars = {}
        
        # For each factor, find the variable with the highest absolute loading
        for factor in loadings.columns:
            abs_loadings = loadings[factor].abs()
            surrogate = abs_loadings.idxmax()
            loading_value = loadings.loc[surrogate, factor]
            surrogate_vars[factor] = {
                'variable': surrogate,
                'loading': loading_value
            }
        return surrogate_vars

    def calculate_model_fit(X, fa):
        original_corr = np.corrcoef(X.T)
        loadings = fa.loadings_
        reproduced_corr = np.dot(loadings, loadings.T)
        residuals = original_corr - reproduced_corr
        rmse = np.sqrt(mean_squared_error(original_corr.flatten(), reproduced_corr.flatten()))
        n_residuals = np.sum(np.abs(residuals) > 0.05)
        total_elements = len(residuals.flatten())
        percent_large_residuals = (n_residuals / total_elements) * 100
        fit_stats = {
            'rmse': rmse,
            'n_large_residuals': n_residuals,
            'percent_large_residuals': percent_large_residuals
        }
        return fit_stats

    def perform_complete_factor_analysis(X, feature_names, n_factors=5):
        fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
        fa.fit(X)
        surrogate_vars = find_surrogate_variables(fa, feature_names, n_factors)
        fit_stats = calculate_model_fit(X, fa)
        loadings_df = pd.DataFrame(
            fa.loadings_,
            columns=[f'Factor{i+1}' for i in range(n_factors)],
            index=feature_names
        )
        return {
            'factor_analyzer': fa,
            'loadings': loadings_df,
            'surrogate_variables': surrogate_vars,
            'model_fit': fit_stats
        }
    
    st.subheader("Factor Analysis Results")
    
    chi_square_value, p_value = calculate_bartlett_sphericity(X)
    st.write(f"Chi-Square Value: {chi_square_value:.2f}, p-value: {p_value:.4f}")

    kmo_all, kmo_model = calculate_kmo(X)
    st.write(f"KMO Model Value: {kmo_model:.4f}")

    fa = FactorAnalyzer(n_factors=25, rotation=None)
    fa.fit(X)

    ev, v = fa.get_eigenvalues()
    st.write("Eigenvalues:")
    st.write(ev)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ev) + 1), ev, marker='o')
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.grid()
    st.pyplot(plt)

    fa_varimax_6 = FactorAnalyzer(n_factors=6, rotation="varimax")
    fa_varimax_6.fit(X)
    st.write("Factor Loadings for 6 Factors with Varimax Rotation:")
    st.write(fa_varimax_6.loadings_)

    fa_varimax_5 = FactorAnalyzer(n_factors=5, rotation="varimax")
    fa_varimax_5.fit(X)
    st.write("Factor Loadings for 5 Factors with Varimax Rotation:")
    st.write(fa_varimax_5.loadings_)
    
    results = perform_complete_factor_analysis(X, feature_names, n_factors=5)
    
    st.subheader("Factor Loadings")
    st.dataframe(results['loadings'])

    st.subheader("Surrogate Variables")
    surrogate_df = pd.DataFrame.from_dict(results['surrogate_variables'], orient='index')
    st.table(surrogate_df)

    st.subheader("Model Fit Statistics")
    fit_stats_df = pd.DataFrame([results['model_fit']])
    st.table(fit_stats_df)