import streamlit as st
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Adult Income Prediction", layout="wide")

REPO_URL = "https://github.com/wulanlumbantoruan/Capstone-Project-PSD"
FITUR_WAJIB = ['education.num', 'age', 'hours.per.week', 'capital.gain', 'sex', 'relationship']


@st.cache_resource
def load_artifact():
    with open("generate_income.pkl", "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_eval():
    with open("eval_data.pkl", "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("adult.csv")
    except FileNotFoundError:
        st.error(f"File adult.csv tidak ditemukan. Silakan akses data di repositori: {REPO_URL}")
        st.stop()


artifact = load_artifact()
model = artifact['model']
scaler = artifact['scaler']
mappings = artifact['mappings']
features = artifact['features']
threshold = artifact['threshold']
ev = load_eval()


def page_home():
    st.title("Capstone Project - Kelompok 6 PSD")
    st.header("Adult Income Prediction")
    st.markdown(f"**GitHub:** [{REPO_URL}]({REPO_URL})")

    st.subheader("Informasi dataset")
    st.write("Dataset Adult Census Income berisi data sensus penduduk Amerika Serikat tahun 1994. "
             "Tujuannya memprediksi apakah pendapatan tahunan seseorang **di atas 50K USD (>50K)** "
             "atau **tidak (<=50K)** berdasarkan atribut demografi dan pekerjaan.")

    df = load_dataset()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah data", f"{len(df):,}")
    c2.metric("Jumlah fitur", df.shape[1] - 1)
    c3.metric("Jumlah kelas", df['income'].nunique())
    c4.metric("Fitur dipakai model", len(FITUR_WAJIB))
    st.caption("Setelah 24 data duplikat dihapus, 32.537 data digunakan: 26.029 data latih dan "
               "6.508 data uji (rasio 80:20).")

    st.write("Jumlah data per kelas:")
    st.bar_chart(df['income'].value_counts())

    st.subheader("Algoritma yang digunakan")
    st.write("Empat algoritma dibandingkan: **Logistic Regression**, **Decision Tree**, "
             "**Random Forest**, dan **MLP (Neural Network)**, masing-masing dengan hyperparameter tuning. "
             "Model terbaik yang dipakai pada aplikasi ini adalah **Random Forest**.")

    st.subheader("Ringkasan performa model (Random Forest)")
    rf = ev['models']['Random Forest']
    t = rf['at_threshold']
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC", f"{rf['auc']:.3f}")
    m2.metric("Akurasi", f"{t['accuracy']:.3f}")
    m3.metric("Recall (>50K)", f"{t['recall']:.3f}")
    m4.metric("F1-score (>50K)", f"{t['f1']:.3f}")
    st.caption(f"Dihitung pada data uji dengan threshold terbaik {rf['threshold']:.3f}. "
               "Detail perbandingan semua model ada di halaman Model Evaluation.")


def page_home():
    st.title("Capstone Project - Kelompok 6 PSD")
    st.header("Adult Income Prediction")
    st.markdown(f"**GitHub:** [{REPO_URL}]({REPO_URL})")

    st.subheader("Informasi dataset")
    st.write("Dataset Adult Census Income berisi data sensus penduduk Amerika Serikat tahun 1994. "
             "Tujuannya memprediksi apakah pendapatan tahunan seseorang **di atas 50K USD (>50K)** "
             "atau **tidak (<=50K)** berdasarkan atribut demografi dan pekerjaan.")

    df = load_dataset()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah data", f"{len(df):,}")
    c2.metric("Jumlah fitur", df.shape[1] - 1)
    c3.metric("Jumlah kelas", df['income'].nunique())
    c4.metric("Fitur dipakai model", len(FITUR_WAJIB))
    st.caption("Setelah 24 data duplikat dihapus, 32.537 data digunakan: 26.029 data latih dan "
               "6.508 data uji (rasio 80:20).")

    st.write("Jumlah data per kelas:")
    st.bar_chart(df['income'].value_counts())

    st.subheader("Algoritma yang digunakan")
    st.write("Empat algoritma dibandingkan: **Logistic Regression**, **Decision Tree**, "
             "**Random Forest**, dan **MLP (Neural Network)**, masing-masing dengan hyperparameter tuning. "
             "Model terbaik yang dipakai pada aplikasi ini adalah **Random Forest**.")

    st.subheader("Ringkasan performa model (Random Forest)")
    rf = ev['models']['Random Forest']
    t = rf['at_threshold']
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC", f"{rf['auc']:.3f}")
    m2.metric("Akurasi", f"{t['accuracy']:.3f}")
    m3.metric("Recall (>50K)", f"{t['recall']:.3f}")
    m4.metric("F1-score (>50K)", f"{t['f1']:.3f}")
    st.caption(f"Dihitung pada data uji dengan threshold terbaik {rf['threshold']:.3f}. "
               "Detail perbandingan semua model ada di halaman Model Evaluation.")


def hasil_prediksi(input_df):
    X_in = input_df[features]
    X_scaled = scaler.transform(X_in)
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, pred


def page_prediction():
    st.title("Prediction")
    st.write("Masukkan data seseorang untuk memprediksi kelas pendapatannya.")
    tab_manual, tab_csv = st.tabs(["Input manual", "Unggah CSV"])

    with tab_manual:
        c1, c2 = st.columns(2)
        with c1:
            education_num = st.slider("Tingkat pendidikan (education.num)", 1, 16, 10)
            st.caption("Contoh: 9 = HS-grad, 10 = Some-college, 13 = Bachelors, 14 = Masters, 16 = Doctorate")
            age = st.slider("Usia (age)", 17, 90, 35)
            hours = st.slider("Jam kerja per minggu (hours.per.week)", 1, 99, 40)
        with c2:
            capital_gain = st.number_input("Capital gain (USD)", min_value=0, max_value=99999, value=0)
            sex_label = st.selectbox("Jenis kelamin (sex)", ('Perempuan', 'Pria'))
            rel_label = st.selectbox("Peran dalam keluarga (relationship)", list(mappings['relationship'].keys()))

        if st.button("Predict!"):
            input_df = pd.DataFrame({
                'education.num': [education_num],
                'age': [age],
                'hours.per.week': [hours],
                'capital.gain': [capital_gain],
                'sex': [0 if sex_label == 'Perempuan' else 1],
                'relationship': [mappings['relationship'][rel_label]],
            })
            proba, pred = hasil_prediksi(input_df)

            st.subheader("Hasil prediksi")
            if pred[0] == 1:
                st.success("Kelas: **>50K** (pendapatan tahunan di atas 50.000 USD)")
            else:
                st.info("Kelas: **<=50K** (pendapatan tahunan 50.000 USD atau kurang)")
            st.write(f"Probabilitas kelas >50K: **{proba[0]:.2f}**")
            st.progress(float(proba[0]))

            st.subheader("Keterangan hasil klasifikasi")
            st.write(f"Model Random Forest menghitung probabilitas kelas >50K, lalu memprediksi **>50K** "
                     f"jika probabilitasnya mencapai threshold **{threshold:.3f}** (bukan 0,5). "
                     "Threshold diturunkan agar lebih banyak orang berpenghasilan >50K terdeteksi "
                     "(recall naik), dengan konsekuensi precision lebih rendah. "
                     "Data latih berasal dari sensus 1994, sehingga hasil ini bersifat indikatif.")
            st.write("Data yang dimasukkan:")
            st.dataframe(pd.DataFrame({
                'Fitur': ['education.num', 'age', 'hours.per.week', 'capital.gain', 'sex', 'relationship'],
                'Nilai': [education_num, age, hours, capital_gain, sex_label, rel_label]}))

    with tab_csv:
        template = pd.DataFrame({
            'education.num': [13, 9],
            'age': [39, 50],
            'hours.per.week': [40, 45],
            'capital.gain': [0, 5000],
            'sex': ['Male', 'Female'],
            'relationship': ['Husband', 'Not-in-family']
        })
        st.download_button("Download template CSV", template.to_csv(index=False),
                           file_name="template_input.csv", mime="text/csv")
        st.caption(f"Nilai relationship yang valid: {list(mappings['relationship'].keys())}")
        uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            kurang = [c for c in FITUR_WAJIB if c not in input_df.columns]
            if kurang:
                st.error(f"Kolom berikut belum ada di CSV: {kurang}. Gunakan template CSV di atas.")
                return
            if not pd.api.types.is_numeric_dtype(input_df['sex']):
                input_df['sex'] = input_df['sex'].map({'Female': 0, 'Male': 1})
            if not pd.api.types.is_numeric_dtype(input_df['relationship']):
                input_df['relationship'] = input_df['relationship'].map(mappings['relationship'])
            if input_df[FITUR_WAJIB].isnull().any().any():
                st.error("Ada nilai kosong atau kategori tidak dikenali pada kolom 'sex' atau 'relationship'.")
                return

            if st.button("Predict CSV!"):
                proba, pred = hasil_prediksi(input_df)
                out = input_df.copy()
                out['probabilitas >50K'] = proba.round(3)
                out['prediksi'] = ['>50K' if p == 1 else '<=50K' for p in pred]
                st.dataframe(out)


def page_evaluation():
    st.title("Model Evaluation")
    st.write(f"Evaluasi pada data uji ({ev['n_test']:,} data). Metrik precision, recall, dan F1-score "
             "dihitung untuk kelas >50K, karena data target tidak seimbang (76% <=50K dan 24% >50K).")
    nama_model = list(ev['models'].keys())

    st.subheader("1. Perbandingan model sebelum dan sesudah hyperparameter tuning")
    baris = []
    for n in nama_model:
        b = ev['before'][n]
        a = ev['models'][n]['default']
        baris.append({'Model': n,
                      'Akurasi (sebelum)': b['accuracy'], 'Akurasi (sesudah)': a['accuracy'],
                      'Recall (sebelum)': b['recall'], 'Recall (sesudah)': a['recall'],
                      'F1 (sebelum)': b['f1'], 'F1 (sesudah)': a['f1']})
    st.dataframe(pd.DataFrame(baris).round(3).set_index('Model'))
    st.caption("Sebelum dan sesudah tuning memakai threshold bawaan 0,5.")

    st.subheader("2. Parameter terbaik (GridSearchCV, cv=5, scoring=roc_auc)")
    st.dataframe(pd.DataFrame({
        'Model': nama_model,
        'Best params': [str(ev['models'][n]['best_params']) for n in nama_model],
        'AUC validasi silang': [round(ev['models'][n]['cv_auc'], 3) for n in nama_model]}).set_index('Model'))

    st.subheader("3. ROC dan AUC")
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in nama_model:
        ax.plot(ev['models'][n]['fpr'], ev['models'][n]['tpr'], label=f"{n} (AUC = {ev['models'][n]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for Income Prediction Models')
    ax.legend()
    st.pyplot(fig)

    st.subheader("4. Threshold terbaik (Youden's Index) dan performanya")
    baris = []
    for n in nama_model:
        m = ev['models'][n]
        t = m['at_threshold']
        baris.append({'Model': n, 'AUC': m['auc'], 'Threshold terbaik': m['threshold'],
                      'Akurasi': t['accuracy'], 'Precision': t['precision'],
                      'Recall': t['recall'], 'F1': t['f1']})
    st.dataframe(pd.DataFrame(baris).round(3).set_index('Model'))

    st.subheader("5. Confusion matrix Random Forest (threshold terbaik)")
    cm = ev['models']['Random Forest']['cm']
    st.dataframe(pd.DataFrame(cm, index=['Aktual <=50K', 'Aktual >50K'],
                              columns=['Prediksi <=50K', 'Prediksi >50K']))

    st.subheader("6. Kesimpulan")
    rf = ev['models']['Random Forest']
    t = rf['at_threshold']
    st.write(f"Model **Random Forest** dipilih karena memiliki AUC-ROC tertinggi ({rf['auc']:.3f}) "
             f"dan performa paling seimbang pada threshold terbaik ({rf['threshold']:.3f}): "
             f"recall {t['recall']:.2f}, precision {t['precision']:.2f}, F1-score {t['f1']:.2f}, "
             f"dan akurasi {t['accuracy']:.2f}. Pada threshold bawaan 0,5, recall kelas >50K hanya "
             f"{rf['default']['recall']:.2f}, sehingga threshold diturunkan agar lebih banyak responden "
             "berpenghasilan >50K terdeteksi, dengan konsekuensi precision lebih rendah.")


FITUR_DESKRIPSI = pd.DataFrame([
    ['age', 'Numerik', 'Usia responden (tahun)', 'Ya'],
    ['workclass', 'Kategorik', 'Sektor tempat bekerja (Private, Self-emp, Federal-gov, dst.)', 'Tidak'],
    ['fnlwgt', 'Numerik', 'Bobot sampel sensus: perkiraan jumlah penduduk yang diwakili responden', 'Tidak'],
    ['education', 'Kategorik', 'Jenjang pendidikan tertinggi (teks)', 'Tidak (diwakili education.num)'],
    ['education.num', 'Numerik (ordinal)', 'Jenjang pendidikan dalam angka 1-16 (9 = HS-grad, 13 = Bachelors, 16 = Doctorate)', 'Ya'],
    ['marital.status', 'Kategorik', 'Status pernikahan', 'Tidak'],
    ['occupation', 'Kategorik', 'Jenis pekerjaan', 'Tidak'],
    ['relationship', 'Kategorik', 'Peran responden dalam keluarga (Husband, Wife, Own-child, dst.)', 'Ya'],
    ['race', 'Kategorik', 'Ras', 'Tidak'],
    ['sex', 'Kategorik', 'Jenis kelamin', 'Ya'],
    ['capital.gain', 'Numerik', 'Keuntungan modal dari investasi di luar gaji (USD)', 'Ya'],
    ['capital.loss', 'Numerik', 'Kerugian modal (USD)', 'Tidak'],
    ['hours.per.week', 'Numerik', 'Jam kerja per minggu', 'Ya'],
    ['native.country', 'Kategorik', 'Negara asal', 'Tidak'],
    ['income', 'Kategorik (target)', 'Kelas pendapatan tahunan: <=50K atau >50K', 'Target'],
], columns=['Fitur', 'Tipe', 'Deskripsi', 'Dipakai model'])


def page_dataset():
    st.title("Dataset Information")
    st.markdown(f"Akses data di repositori GitHub: [{REPO_URL}]({REPO_URL})")
    df = load_dataset()
    st.write(f"Dataset memiliki **{df.shape[0]:,} baris** dan **{df.shape[1]} kolom**. "
             "Nilai tidak diketahui pada data asli ditulis dengan tanda '?' (kolom workclass, "
             "occupation, dan native.country).")

    st.subheader("Preview dataset")
    n = st.slider("Jumlah baris ditampilkan", 5, 100, 10)
    st.dataframe(df.head(n))

    st.subheader("Nama dan deskripsi fitur")
    st.dataframe(FITUR_DESKRIPSI.set_index('Fitur'))


st.sidebar.title("Menu")
halaman = st.sidebar.radio("Pilih halaman",
                           ["Home / Dashboard", "Prediction", "Model Evaluation", "Dataset Information"])

if halaman == "Home / Dashboard":
    page_home()
elif halaman == "Prediction":
    page_prediction()
elif halaman == "Model Evaluation":
    page_evaluation()
else:
    page_dataset()
