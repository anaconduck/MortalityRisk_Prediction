# -*- coding: utf-8 -*-
"""Mortality Risk Prediction_COVID 19.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mF-V0p4RaK-F_UTjwI570Je2Op9qnCSS

# Import Library
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
# %matplotlib inline
import seaborn as sns
import plotly.express as px

import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score

"""# Load Data"""

# Menampilkan fitur lengkap
df = pd.read_excel(r'https://github.com/anaconduck/Dataset/blob/main/data_asli.xlsx?raw=true')
df_eda = pd.read_excel(r'https://github.com/anaconduck/Dataset/blob/main/data_d.xlsx?raw=true')

# Hanya menampilkan fitur gejala
df_gejala = pd.read_excel(r'https://github.com/anaconduck/Dataset/blob/main/gejala.xlsx?raw=true')

df.head()

"""Berdasarkan info diatas, dataset memiliki 179 baris dan 52 kolom (fitur)."""

df.info()

"""Dari info tersebut, dapat ditemukan beberapa kolom dengan nilai kosong (missing value). Hal ini dapat dilihat dari jumlah baris yang < 179. Selanjutnya akan dicari tahu untuk lebih detail pada tahap EDA.

# EDA (Exploratory Data Analysis)

## Missing Values
"""

# Cek Data Kosong
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df

sns.heatmap(df.isnull(), cbar=False)

"""Dari info tersebut, hampir semua fitur memiliki missing value kecuali fitur nama.

## Duplicates
"""

# Mengecek Duplikat
df.duplicated().sum()

"""Tidak terdapat duplikat pada dataset

## Data Statistics
"""

# Statistik Data
df.describe()

df_dm1 = df[df['DM'] == 1]

df_dm1.describe()

"""## Uni-variate Analysis

### Jenis Kelamin
"""

plt.figure(figsize=(12,8))
plt.title("Jenis Kelamin")
circle = plt.Circle((0, 0), 0.5, color='white')
g = plt.pie(df['Jenis Kelamin'].value_counts(),
            explode=(0.025,0.025),
            labels=['Laki-Laki','Perempuan'],
            colors=['skyblue','navajowhite'],
            autopct='%1.1f%%',
            startangle=180);
plt.legend()
p = plt.gcf()
p.gca().add_artist(circle)
plt.show()

"""Dari pie-chart tersebut, dapat dikatakan bahwa pasien laki - laki dan perempuan jumlahnya sama. Hal ini berarti data seimbang.

### Usia
"""

plt.figure(figsize=(25,15))
sns.countplot(x=df["Usia"])
plt.title("COUNT OF PATIENTS AGE",fontsize=20)
plt.xlabel("AGE",fontsize=20)
plt.ylabel("COUNT",fontsize=20)
plt.show()

sns.displot(data=df, x="Usia", color="magenta")
plt.xlabel("DISTRIBUTION OF AGE")
plt.show()

"""Dari info diatas, dapat diketahui bahwa umur pasien kebanyakan dikisaran 60 - 70 tahun.

### BMI
"""

sns.displot(data=df, x="BMI", color="magenta")
plt.xlabel("DISTRIBUTION OF BMI")
plt.show()

"""Beberapa kategori BMI, yaitu :
- < 18.5 = berat tidak proporsional
- Antara 18,5 dan 22,9 = berat badan normal
- Antara 23 dan 29,9 = kelebihan berat badan (berpotensi obesitas)
- Lebih dari 30 = Obesitas

Berdasarkan diagram dan informasi mengenai kategori BMI, maka mayoritas BMI pasien berada antara angka 20 - 25 yang berarti belum mencapai obesitas. Hanya beberapa pasien yang mengalami obesitas.

### Gejala
"""

df_gejala.hist(bins=15,
        color='steelblue',
        edgecolor='black',
        linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(0, 0, 1.2, 1.2))

"""Dari diagram diatas, gejala batuk, demam, dan sesak merupakan hal yang umum terjadi ketika pasien ingin melakukan pemeriksaan. Sedangkan, gejala lainnya hanya beberapa pasien yang mengalami.

### Keparahan Infeksi
"""

c=df["Keparahan Infeksi "].value_counts().reset_index()
plt.figure(figsize=(20,10))
sns.barplot(x=c["index"],y=c["Keparahan Infeksi "])
plt.title("KEPARAHAN INFEKSI",fontsize=20)
plt.xlabel("TYPE",fontsize=20)
plt.ylabel("COUNT",fontsize=20)
plt.show()

"""### Leukosit"""

sns.displot(data=df, x="Leukosit", color="magenta")
plt.xlabel("DISTRIBUTION OF LEUKOCYTES")
plt.show()

"""### NLR, PT, INR, Fibrinogen, D-dimer, SGOT, LDH, CRP, TNF α, MCP-1, IP-10, IL-4"""

df1 = df[['NLR', 'PT', 'INR', 'Fibrinogen', 'D-dimer', 'SGOT', 'LDH', 'CRP', 'MCP-1', 'IL1B','TNF α', 'IP-10', 'IL-4' ]]

df1.hist(bins=15,
        color='steelblue',
        edgecolor='black',
        linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(0, 0, 1.2, 1.2))

"""### Penderita Diabetes Melitus"""

plt.figure(figsize=(12,8))
plt.title("diabetes")
circle = plt.Circle((0, 0), 0.5, color='white')
g = plt.pie(df.DM.value_counts(),
            explode=(0.025,0.025),
            labels=['Non DM','DM'],
            colors=['skyblue','b'],
            autopct='%1.1f%%',
            startangle=180);
plt.legend()
p = plt.gcf()
p.gca().add_artist(circle)
plt.show()

# Laki laki = 1.0, Perempuan = 0.0
gender_DM = pd.crosstab(df['DM'], (df['Jenis Kelamin']))

gender_DM

"""Dari info diatas, maka didapatkan informasi bahwa kebanyakan pasien pada dataset tidak menderita diabetes melitus, dengan jumlah 37 orang perempuan dan 44 orang laki - laki.

### Mortalitas
"""

plt.figure(figsize=(12,8))
plt.title("Mortalitas")
circle = plt.Circle((0, 0), 0.5, color='white')
g = plt.pie(df.Mortalitas.value_counts(),
            explode=(0.025,0.025),
            labels=['Alive','Death'],
            colors=['#4F6272', '#B7C3F3'],
            autopct='%1.1f%%',
            startangle=180);
plt.legend()
p = plt.gcf()
p.gca().add_artist(circle)
plt.show()

"""Kurang dari 40% pasien meninggal pada dataset, sisa pasien masih hidup

## Bi-Variate Analysis
"""

# Hubungan mortalitas dengan usia
plt.figure(figsize=(12,10))
sns.set_theme(style="darkgrid", color_codes=True)
ax = sns.countplot(y="Usia", hue="Mortalitas", data=df, palette="PuBu")
plt.title("Hubungan Mortalitas dengan Usia")
plt.show()

"""Dari info tersebut, jumlah paling banyak meninggal pada usia pasien 67 - 70 tahun."""

# Hubungan mortalitas dengan tingkat keparahan infeksi
plt.figure(figsize=(10,10))
sns.set_theme(style="darkgrid", color_codes=True)
ax = sns.countplot(y="Keparahan Infeksi ", hue="Mortalitas", data=df, palette="PuBu")
plt.title("Hubungan Mortalitas dengan Keparahan Infeksi")
plt.show()

"""Dari info tersebut, didapatkan informasi bahwa semakin tinggi angka keparahan infeksi maka kemungkinan pasien meninggal lebih tinggi juga begitu pun sebaliknya."""

# Hubungan mortalitas dengan tingkat keparahan infeksi
plt.figure(figsize=(10,10))
sns.set_theme(style="darkgrid", color_codes=True)
ax = sns.countplot(y="DM", hue="Mortalitas", data=df, palette="PuBu")
plt.title("Hubungan Mortalitas dengan Diabetes Melitus")
plt.show()

"""Dari info tersebut, hubungan antara penyakit diabetes melitus dengan tingkat mortalitas tidak terlalu berpengaruh. Hal ini dikarenakan tingkat kehidupan dan kematian pada pasien DM dan non DM hampir sama.

## Multi-Variate Analysis

Untuk tahap ini akan menggunakan heatmap, yang mana merupakan sebuah visualisasi untuk menggambarkan matriks data dalam bentuk warna dengan memanfaatkan skala warna yang berbeda untuk merepresentasikan nilai-nilai dalam matriks tersebut. Dalam konteks analisis data, heatmap sering digunakan untuk memvisualisasikan matriks korelasi antara variabel-variabel dalam dataset. Dengan menggunakan heatmap, kita dapat dengan cepat melihat hubungan antara variabel-variabel tersebut, di mana warna yang lebih terang menunjukkan korelasi yang lebih tinggi, sedangkan warna yang lebih gelap menunjukkan korelasi yang lebih rendah.

### Keseluruhan Atribut / Variabel
"""

# tambahkan info + perjelas
corr = df.corr()
plt.figure(figsize=(50,50))
# sns.heatmap(corr, cmap='PuBu', annot=True)
sns.heatmap(corr, annot=True, cmap='PuBu',
            annot_kws={'fontsize':11},
            linewidths=0.01,linecolor="white");

"""### Variabel Penting"""

corr_eda = df_eda.corr()
plt.figure(figsize=(32,32))
sns.heatmap(corr_eda, cmap='PuBu', annot=True,
            linewidths=0.01,linecolor="white")

"""Dari info korelasi diatas, terdapat beberapa fitur yang berhubungan dengan tingkat mortalitas, yaitu
Usia,
BMI,
Keparahan Infeksi,
Leukosit,
NLR,
SGOT,
PT,
INR,
Fibrinogen,
D-dimer,
LDH,
CRP,
TNF α,
MCP-1,
IP-10,
IL-4,
IL-6. Oleh karena itu, pada tahap selanjutnya dilakukan penghapusan pada fitur yang memiliki korelasi rendah terhadap mortalitas.

### Penderita Diabetes
"""

df_dm = df_eda[df_eda['DM'] == 1]

corr_dm = df_dm.corr()
plt.figure(figsize=(32,32))
sns.heatmap(corr_dm, cmap='PuBu', annot=True,
            linewidths=0.01,linecolor="white")

"""Variabel berpengaruh dengan mortalitas pada pasien DM : keparahan infeksi, leukosit, neutrofil, PT, INR, Fibrinogen, SGOT, LDH, CRP, MCP-1, IL1B, TNF a, IP-10, IL-4, IL-6, IFN y.

# Preprocessing
"""

# Data dengan fitur yang penting (korelasi tinggi dengan penyebab mortalitas)
df_prepro = pd.read_excel(r'https://github.com/anaconduck/Dataset/blob/main/data_prepro.xlsx?raw=true')

df_prepro.head()

"""## Handle Null Data

Pada tahap ini dilakukan pengisian nilai kosong pada data dengan menggunakan model machine learning yaitu random forest.
"""

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

df_null = df_prepro[['Usia', 'BMI', 'Keparahan-Infeksi ', 'Leukosit', 'NLR', 'PT', 'Fibrinogen', 'D-dimer', 'SGOT', 'LDH', 'CRP','Procalcitonin','MCP-1','TNF α', 'IP-10', 'IL-4','IL-6','DM' ]]

imptr = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)
data = pd.DataFrame(imptr.fit_transform(df_null), columns = df_null.columns)
data.head()

df['Mortalitas'] = df['Mortalitas'].fillna(df['Mortalitas'].mode()[0])

df['Mortalitas']

"""## Handle Outliers"""

# Q1
q1 = data.quantile(0.25)
# Q3
q3 = data.quantile(0.75)
# IQR
IQR = q3 - q1
# Outlier range
upper = q3 + IQR * 1.5
lower = q1 - IQR * 1.5
upper_dict = dict(upper)
lower_dict = dict(lower)

for i,v in data.items():
    v_col = v[( v<= lower_dict[i]) | (v >= upper_dict[i])]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
    print("Column {} outliers = {} => {}%".format(i,len(v_col),round((perc),3)))

"""## Split Dataset

Split dataset dengan rasio 75% data train dan 25% data test.
"""

X = data
Y = df['Mortalitas']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state=42)

"""## Scaling

Pada tahap ini dilakukan scaling dengan menerapkan robust scaler.Robust Scaler adalah sebuah metode dalam preprocessing data yang digunakan untuk menangani outlier dalam dataset. Cara kerja Robust Scaler secara singkat adalah sebagai berikut:

1. Menghitung Median: Pertama, Robust Scaler menghitung median dari setiap fitur dalam dataset. Median adalah nilai tengah dalam urutan data yang diurutkan.

2. Menghitung Interquartile Range (IQR): Selanjutnya, Robust Scaler menghitung Interquartile Range (IQR) dari setiap fitur. IQR adalah perbedaan antara quartil ketiga (Q3) dan quartil pertama (Q1) dalam urutan data yang diurutkan.

3. Scaling dengan Median dan IQR: Setelah itu, Robust Scaler menggunakan median dan IQR untuk melakukan scaling pada setiap fitur. Nilai dari setiap titik data di-fitur tersebut dikurangi dengan median dan kemudian dibagi dengan IQR.

Dengan tahap ini, outlier dapat diatasi karena median dan IQR lebih tahan terhadap nilai ekstrem dibandingkan dengan mean dan standar deviasi yang digunakan dalam metode-scaling lainnya seperti Standard Scaler.Penerapan Robust Scaler pada dataset membantu dalam mempertahankan distribusi asli data dan mengurangi efek dari outlier pada hasil analisis.


Berikut adalah contoh hasil Robust Scaler pada sebuah dataset:

- Dataset Input:
[2, 5, 10, 12, 20, 25, 30]

- Hasil setelah diaplikasikan Robust Scaler:
[-0.85714286, -0.42857143, 0.28571429, 0.57142857, 1.14285714, 1.42857143, 1.71428571]
"""

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""# Modeling

## Logistic Regression

Model
"""

lr = LogisticRegression(solver='liblinear', random_state=0)
lr.fit(X_train_scaled,y_train)
y_predt= lr.predict(X_test_scaled)

"""Evaluation"""

accuracy_lr = accuracy_score(y_test, y_predt)
precision_lr = precision_score(y_test, y_predt)
recall_lr = recall_score(y_test, y_predt)
f1_score_lr = f1_score(y_test, y_predt)

print("Logistic Regression:")
print('Accuracy: %.3f' % accuracy_score(y_test, y_predt))
print('F1 Score: %.3f' % f1_score(y_test, y_predt))
print('Precision: %.3f' % precision_score(y_test, y_predt))
print('Recall: %.3f' % recall_score(y_test, y_predt))

"""Confusion Matrix"""

print(confusion_matrix(y_test, y_predt))

"""AUC Score"""

plt.figure(figsize=(10, 6))

# Logistic Regression
y_lr_scores = lr.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_lr_scores)
auc_lr = roc_auc_score(y_test, y_lr_scores)
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = {:.2f})'.format(auc_lr))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression')
plt.legend(loc='lower right')
plt.show()

"""## Decision Tree

Model with Hyperparameter Tuning
"""

dt_clf = DecisionTreeClassifier()
dt_param_grid = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
dt_grid_search = GridSearchCV(dt_clf, dt_param_grid, cv=5)

dt_grid_search.fit(X_train_scaled, y_train)

dt_best_params = dt_grid_search.best_params_
dt_best_model = dt_grid_search.best_estimator_
dt_y_pred = dt_best_model.predict(X_test_scaled)

accuracy_dt = accuracy_score(y_test, dt_y_pred)
f1_score_dt = f1_score(y_test, dt_y_pred, average='weighted')
precision_dt = precision_score(y_test, dt_y_pred, average='weighted')
recall_dt = recall_score(y_test, dt_y_pred, average='weighted')

"""Evaluation"""

print("Decision Tree:")
print("Best Parameters:", dt_best_params)
print("Accuracy:", accuracy_dt)
print("F1-score:", f1_score_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)

"""Confusion Matrix"""

print(confusion_matrix(y_test, dt_y_pred))

"""AUC Score"""

plt.figure(figsize=(10, 6))

# Decision Tree
y_dt_scores = dt_best_model.predict_proba(X_test_scaled)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_dt_scores)
auc_dt = roc_auc_score(y_test, y_dt_scores)
plt.plot(fpr_dt, tpr_dt, label='Decision Tree (AUC = {:.2f})'.format(auc_dt))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree')
plt.legend(loc='lower right')
plt.show()

from scipy.stats import chi2_contingency
import numpy as np

# Confusion matrix
confusion_matrix = np.array([[28, 5], [8, 4]])

# Hitung nilai p-value dengan uji chi-square
chi2, p, _, _ = chi2_contingency(confusion_matrix)

# Tampilkan nilai p-value
print("Nilai p-value:", p)

"""## Naive Bayes

Model with Hyperparameter Tuning
"""

nb_clf = GaussianNB()
nb_param_grid = {}
nb_grid_search = GridSearchCV(nb_clf, nb_param_grid, cv=5)

nb_grid_search.fit(X_train_scaled, y_train)

nb_best_params = nb_grid_search.best_params_
nb_best_model = nb_grid_search.best_estimator_
nb_y_pred = nb_best_model.predict(X_test_scaled)

accuracy_nb = accuracy_score(y_test, nb_y_pred)
f1_score_nb = f1_score(y_test, nb_y_pred, average='weighted')
precision_nb = precision_score(y_test, nb_y_pred, average='weighted')
recall_nb = recall_score(y_test, nb_y_pred, average='weighted')

"""Evaluation"""

print("Naive Bayes:")
print("Best Parameters:", nb_best_params)
print("Accuracy:", accuracy_nb)
print("F1-score:", f1_score_nb)
print("Precision:", precision_nb)
print("Recall:", recall_nb)

"""Confusion Matrix"""

print(confusion_matrix(y_test, nb_y_pred))

"""AUC Score"""

plt.figure(figsize=(10, 6))

# Naive Bayes
y_nb_scores = nb_best_model.predict_proba(X_test_scaled)[:, 1]
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_nb_scores)
auc_nb = roc_auc_score(y_test, y_nb_scores)
plt.plot(fpr_nb, tpr_nb, label='Naive Bayes (AUC = {:.2f})'.format(auc_nb))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes')
plt.legend(loc='lower right')
plt.show()

from scipy.stats import chi2_contingency

# Confusion matrix
confusion_matrix = np.array([[30, 3], [3, 9]])

# Hitung nilai p-value dengan uji chi-square
chi2, p, _, _ = chi2_contingency(confusion_matrix)

# Tampilkan nilai p-value
print("Nilai p-value:", p)

"""## SVM

Model with Hyperparameter Tuning
"""

svm = SVC(probability=True)

svm_param_grid = {'C': [0.1, 1, 10],
                  'gamma': [1, 0.1, 0.01],
                  'kernel': ['linear', 'rbf']}

svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5)
svm_grid_search.fit(X_train_scaled, y_train)

svm_best_params = svm_grid_search.best_params_
svm_best_model = svm_grid_search.best_estimator_
svm_y_pred = svm_grid_search.predict(X_test_scaled)

svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_f1_score = f1_score(y_test, svm_y_pred)
svm_precision = precision_score(y_test, svm_y_pred)
svm_recall = recall_score(y_test, svm_y_pred)

"""Evaluation"""

print("SVM:")
print("Best Hyperparameters:", svm_best_params)
print("Accuracy:", svm_accuracy)
print("F1-score:", svm_f1_score)
print("Precision:", svm_precision)
print("Recall:", svm_recall)

"""Confusion Matrix"""

print(confusion_matrix(y_test, svm_y_pred))

"""AUC Score"""

plt.figure(figsize=(10, 6))

# Naive Bayes
y_svm_scores = svm_best_model.predict_proba(X_test_scaled)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_svm_scores)
auc_svm = roc_auc_score(y_test, y_svm_scores)
plt.plot(fpr_svm, tpr_svm, label='SVM (AUC = {:.2f})'.format(auc_svm))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM')
plt.legend(loc='lower right')
plt.show()

"""# Comparison Model

AUC Score
"""

plt.figure(figsize=(10, 6))

# Logistic Regression
y_lr_scores = lr.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_lr_scores)
auc_lr = roc_auc_score(y_test, y_lr_scores)
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = {:.2f})'.format(auc_lr))

# Decision Tree
y_dt_scores = dt_best_model.predict_proba(X_test_scaled)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_dt_scores)
auc_dt = roc_auc_score(y_test, y_dt_scores)
plt.plot(fpr_dt, tpr_dt, label='Decision Tree (AUC = {:.2f})'.format(auc_dt))

# Naive Bayes
y_nb_scores = nb_best_model.predict_proba(X_test_scaled)[:, 1]
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_nb_scores)
auc_nb = roc_auc_score(y_test, y_nb_scores)
plt.plot(fpr_nb, tpr_nb, label='Naive Bayes (AUC = {:.2f})'.format(auc_nb))

# SVM
y_svm_scores = svm_best_model.predict_proba(X_test_scaled)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_svm_scores)
auc_svm = roc_auc_score(y_test, y_svm_scores)
plt.plot(fpr_svm, tpr_svm, label='SVM (AUC = {:.2f})'.format(auc_svm))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison AUC Score')
plt.legend(loc='lower right')
plt.show()

"""Vertical Bar"""

# evaluation metrics for logistic regression model
lr_acc = accuracy_lr
lr_precision = precision_lr
lr_recall = recall_lr
lr_f1 = f1_score_lr

# evaluation metrics for decision tree model
dt_acc = accuracy_dt
dt_precision = precision_dt
dt_recall = recall_dt
dt_f1 = f1_score_dt

# evaluation metrics for random forest model
rf_acc = accuracy_nb
rf_precision = precision_nb
rf_recall = recall_nb
rf_f1 = f1_score_nb

# evaluation metrics for SVM model
svm_acc = svm_accuracy
svm_precision = svm_precision
svm_recall = svm_recall
svm_f1 = svm_f1_score

# create lists for evaluation metrics and model names
#metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
accuracy = [lr_acc, dt_acc, rf_acc, svm_acc]
precision = [lr_precision, dt_precision, rf_precision, svm_precision]
recall = [lr_recall, dt_recall, rf_recall, svm_recall]
f1_score = [lr_f1, dt_f1, rf_f1, svm_f1]
models = ['Logistic Regression','Decision Tree', 'Naive Bayes', 'SVM']

# Create the plot
x = range(len(models))
width = 0.2

fig, ax = plt.subplots()
ax.bar(x, accuracy, width, label='Accuracy')
ax.bar([val + width for val in x], precision, width, label='Precision')
ax.bar([val + 2*width for val in x], recall, width, label='Recall')
ax.bar([val + 3*width for val in x], f1_score, width, label='F1-Score')

# Add labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Evaluation Metrics')
ax.set_xticks([val + width for val in x])
ax.set_xticklabels(models)
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()

"""# Deployment"""

#import joblib as joblib

# Save the model to a file
#joblib.dump(nb_best_model, 'nb_model.pkl')
#joblib.dump(scaler, 'scaler.save')