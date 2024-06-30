#!/usr/bin/env python
# coding: utf-8

# # Final Project IDX Partners Data Science
# 
# - Nama: Dimas Aji Haritson
# - Data: loan_data_2007_2014

# # Business Understanding
# 
# IDX Partners adala perusahaan yang bergerak pada bidang konsultasi yang analisa data dan keputusan yang mengintegrasikan resiko manajemen. Perusahaan IDX Partners ingin meningkatkan keakuratan dalam menilai dan mengelola risiko kredit untuk mengoptimalkan keputusan bisnis dan mengurangi potensi kerugian. 

# ### Permasalahan
# 
# Perusahaan ingin meningkatkan keakuratan dalam menilai dan mengelola risiko kredit untuk mengoptimalkan keputusan bisnis dan mengurangi potensi kerugian. Dengan memanfaatkan data historis pinjaman yang disetujui dan ditolak, tujuan utama proyek ini adalah untuk membangun model prediksi yang dapat secara akurat menilai risiko kredit calon peminjam.

# ### Tujuan
# 
# Membuat model prediksi risiko kredit untuk membantu mengurangi tingkat kredit macet dengan mengidentifikasi calon peminjam yang berisiko tinggi. 

# ### Analytical Approach
# 
# - Data Understanding: Memahami struktur data, mengidentifikasi missing values, dan menganalisis distribusi serta outliers
# - Exploratory Data Analysis (EDA): Analisis korelasi, visualisasi kategori, dan pemahaman distribusi variabel target
# - Data Preparation: Pembersihan data, encoding variabel kategorikal, skala data numerik, dan pembagian data menjadi set pelatihan dan pengujian
# - Data Modelling: Membangun beberapa model machine learning seperti Logistic Regression, Random Forest, dan K-Nearest Neighbors.
# - Evaluation: Mengevaluasi kinerja model dengan set data pengujian dan membandingkan metrik kinerja dari beberapa model

# ## LIBRARIES and Load Dataset

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('loan_data_2007_2014.csv')


# ## Data Understanding

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


# Melihat Missing Value

listItem = []

for col in df.columns:
    listItem.append([col, df[col].dtype, df[col].isnull().sum(), round((df[col].isnull().sum()/len(df[col]))*100, 2), df[col].nunique(), list(df[col].drop_duplicates().sample(5,replace=True).values)]);

df_desc = pd.DataFrame(columns=['Column', 'Dtype', 'null count', 'null perc.', 'unique count', 'unique sample'],
                     data=listItem)
df_desc


# In[8]:


# Melihat duplikasi data

df.duplicated().sum()


# Dari hasil diatas kita dapat melihat  data memiliki jumlah baris 466285 sebanyak dan kolom sebanyak 75. kita juga akan mencari kolom yang memiliki missing value untuk nantinya akan dibersihkan. setelahnya kita mengecek duplikasi data, dan hasilnya 0

# # Exploratory Data Analysis (EDA)

# In[9]:


# Menentukan kategori untuk pinjaman "Excellent" dan "Bad"

status_pinjaman_baik = ['Current', 'Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']
status_pinjaman_buruk = ['Charged Off', 'Late (31-120 days)', 'In Grace Period', 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Charged Off']


# In[10]:


# Membuat kolom baru 'loan_category' untuk mengklasifikasikan pinjaman

df['loan_category'] = df['loan_status'].apply(lambda x: 'Excellent' if x in status_pinjaman_baik else 'Bad')


# In[12]:


# Menghitung jumlah setiap kategori pinjaman
jumlah_kategori_pinjaman = df['loan_category'].value_counts()

# Menentukan warna untuk setiap kategori pinjaman
warna_kategori = ['green', 'red']

# Membuat plot distribusi kategori pinjaman
plt.figure(figsize=(8, 6))
plt.bar(jumlah_kategori_pinjaman.index, jumlah_kategori_pinjaman.values, color=warna_kategori)
plt.xlabel('Kategori Pinjaman')
plt.ylabel('Jumlah')
plt.title('Distribusi Kategori Pinjaman')


# In[13]:


# Histogram untuk distribusi loan_amnt

df['loan_amnt'].hist()
plt.title('Distribusi Loan Amount')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()


# In[14]:


# Boxplot untuk melihat distribusi interest rate berdasarkan grade

sns.boxplot(x=df['grade'], y=df['int_rate'])
plt.title('Interest Rate by Grade')
plt.xlabel('Grade')
plt.ylabel('Interest Rate')
plt.show()


# # Data Preparation

# In[16]:


# Menghapus Kolom yang memiliki missing value

threshold = len(df) * 0.7
df_clean = df.dropna(axis=1, thresh=threshold)


# In[17]:


df_clean.shape


# In[18]:


# Melakukan pengecekan Kolom yang sudah dibersihkan

df_clean.columns


# In[19]:


for column in df_clean.columns:
    value_counts = df_clean[column].value_counts()
    print(f"Value counts for {column}:\n{value_counts}\n")


# In[20]:


# Menghapus Kolom Yang tidak diperlukan

unused_col = ['policy_code', 'application_type', 'Unnamed: 0', 'id', 'member_id','issue_d', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                   'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                   'last_pymnt_d', 'last_pymnt_amnt', 'zip_code', 'title', 'emp_title','loan_status', 'url']

drop_data = df_clean[unused_col]

df_clean.drop(columns=unused_col, axis=1, inplace=True)


# In[21]:


df_clean.head()


# In[22]:


# Mengidentifikasi outlier menggunakan IQR

Q1 = df_clean['loan_amnt'].quantile(0.25)
Q3 = df_clean['loan_amnt'].quantile(0.75)
IQR = Q3 - Q1
df_cleansing = df_clean[~((df_clean['loan_amnt'] < (Q1 - 1.5 * IQR)) | (df_clean['loan_amnt'] > (Q3 + 1.5 * IQR)))]


# In[23]:


# Melakukan Frequency Encoding untuk kolom kategorikal

categorical_columns = df_cleansing.select_dtypes(include=['object']).columns

for column in categorical_columns:
    frequency_encoding = df_cleansing[column].value_counts().to_dict()
    df_cleansing[column] = df_cleansing[column].map(frequency_encoding)

# Mengisi nilai yang hilang dengan median setelah encoding
df_cleansing.fillna(df_cleansing.median(), inplace=True)

print(df_cleansing.head())


# In[24]:


scaler = StandardScaler()
numerical_columns = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment', 'annual_inc', 'dti', 'revol_bal', 'revol_util', 'total_acc']
df_cleansing[numerical_columns] = scaler.fit_transform(df_cleansing[numerical_columns])


# In[25]:


# Memilih hanya kolom numerik

numerical_columns = df_cleansing.select_dtypes(include=['float64', 'int64']).columns
df_numerical = df_cleansing[numerical_columns]


# In[26]:


# Menghitung matriks korelasi

corr_matrix = df_numerical.corr()


# In[27]:


# Membuat heatmap korelasi

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi')
plt.show()


# In[28]:


X = df_cleansing.drop('loan_category', axis=1)
Y = df_cleansing['loan_category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[29]:


# Normalize features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[30]:


# Pelatihan model Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)


# In[31]:


# Pelatihan model Random Forest

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)


# In[32]:


# Pelatihan model K-Nearest Neighbors

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)


# # Evaluation

# In[33]:


# Inisialisasi model yang berbeda
results = {}
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
}

# Inisialisasi dictionary untuk laporan klasifikasi
classification_reports = {}
model_names = []
accuracies = []

# Melakukan Train and evaluate untuk tiap model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, Y_train)

    print(f"Evaluating {model_name}...")
    Y_pred = model.predict(X_test)

    confusion = confusion_matrix(Y_test, Y_pred)
    classification_rep = classification_report(
        Y_test, Y_pred, target_names=['Good', 'Bad'], zero_division=1  # Handle zero division
    )

    # Menyimpan laporan classification di dictionary
    classification_reports[model_name] = classification_rep

    accuracy = accuracy_score(Y_test, Y_pred)

    model_names.append(model_name)
    accuracies.append(accuracy)

    print("\nClassification Report:")
    print(classification_rep)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print("=" * 50)


# In[34]:


# Create a bar plot to visualize accuracies

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Tingkat Akurasi Tiap Model')
plt.ylim(0, 1)  # Set y-axis limits to 0-1 for accuracy percentage
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()


# In[35]:


X_o = df_cleansing.drop(['loan_category'], axis=1)
y_o = df_cleansing['loan_category']


# In[36]:


oversample = RandomOverSampler(sampling_strategy = 'not majority')
X_over, y_over = oversample.fit_resample(X_o, y_o)


# In[37]:


X_train_over, X_test_over, Y_train_over, Y_test_over = train_test_split(X_over, y_over, test_size=0.2, random_state=42)


# In[38]:


scaler = StandardScaler()
X_train_over = scaler.fit_transform(X_train_over)
X_test_over = scaler.fit_transform(X_test_over)


# In[39]:


# Inisialisasi model yang berbeda
results = {}
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
}

# Inisialisasi dictionary untuk laporan klasifikasi
classification_reports = {}
model_names_over = []
accuracies_over = []

# Melakukan Train and evaluate untuk tiap model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_over, Y_train_over)

    print(f"Evaluating {model_name}...")
    Y_pred = model.predict(X_test_over)

    confusion = confusion_matrix(Y_test_over, Y_pred)
    classification_rep = classification_report(
        Y_test_over, Y_pred, target_names=['Good', 'Bad'], zero_division=1  # Handle zero division
    )

    # Menyimpan laporan classification di dictionary
    classification_reports[model_name] = classification_rep

    accuracy = accuracy_score(Y_test_over, Y_pred)

    model_names_over.append(model_name)
    accuracies_over.append(accuracy)

    print("\nClassification Report:")
    print(classification_rep)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print("=" * 50)


# In[40]:


# Create a bar plot to visualize accuracies

plt.figure(figsize=(10, 6))
plt.bar(model_names_over, accuracies_over, color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Tingkat Akurasi Tiap Model')
plt.ylim(0, 1)  # Set y-axis limits to 0-1 for accuracy percentage
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()


# # KESIMPULAN
# 
# Berdasarkan hasil analisis yang telah saya lakukan menggunakna 3 model machine learning yang berbeda yaitu Logistic Regression, Random Forest, dan K-Nearest Neighbors, didapati perbandingan hasil dari Logistic Regression 0.6418, Random Forest 0.9866, K-Nearest Neighbors 0.8268. sehingga dapat disimpulakn, bahwa model machine learning dengan tingkat akurasi tertinggi adalah Random Forest sebesar 0.9866
