import os
import re
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Multi Dataset Classification App", layout="wide")
np.random.seed(42)
tf.random.set_seed(42)

IMG_SIZE = (64, 64)
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


# =========================================================
# COMMON HELPERS
# =========================================================
def evaluate_classification(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }


def make_confusion_df(y_true, y_pred, labels, class_names=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if class_names is None:
        return pd.DataFrame(cm, index=labels, columns=labels)
    return pd.DataFrame(cm, index=class_names, columns=class_names)


# =========================================================
# DINOSAUR FUNCTIONS
# =========================================================
def parse_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", s)
    if not matches:
        return np.nan
    try:
        return float(matches[0])
    except Exception:
        return np.nan


def parse_year(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    years = re.findall(r"(18\d{2}|19\d{2}|20\d{2})", s)
    if not years:
        return np.nan
    try:
        return float(years[0])
    except Exception:
        return np.nan


def normalize_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = s.replace("-", " ")
    s = " ".join(s.split())
    if s in ["", "nan", "none", "unknown", "?", "n/a"]:
        return np.nan
    return s


def clean_diet(x):
    s = normalize_text(x)
    if pd.isna(s):
        return np.nan
    if "herb" in s:
        return "Herbivore"
    if "carn" in s:
        return "Carnivore"
    if "omni" in s:
        return "Omnivore"
    return np.nan


def clean_locomotion(x):
    s = normalize_text(x)
    if pd.isna(s):
        return "Unknown"
    if ("biped" in s or "bi pedal" in s) and ("quad" in s):
        return "Mixed"
    if "biped" in s or "bi pedal" in s:
        return "Bipedal"
    if "quad" in s or "quadriped" in s:
        return "Quadrupedal"
    if "aquatic" in s:
        return "Aquatic"
    return "Other"


def clean_intelligence(x):
    s = normalize_text(x)
    if pd.isna(s):
        return "Unknown"
    if s in ["tiny", "very small", "small", "low"]:
        return "Low"
    if s in ["avg", "med", "medium", "possibly medium"]:
        return "Medium"
    if s in ["high", "large", "very large", "huge"]:
        return "High"
    return "Unknown"


def clean_period(x):
    s = normalize_text(x)
    if pd.isna(s):
        return "Unknown"
    if "cretaceous" in s:
        return "Cretaceous"
    if "jurassic" in s:
        return "Jurassic"
    if "triassic" in s:
        return "Triassic"
    return "Other"


def clean_region(x):
    s = normalize_text(x)
    if pd.isna(s):
        return "Unknown"
    return str(x).strip().title()


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_dino_data(df):
    data = df.copy()

    data["diet_clean"] = data["diet"].apply(clean_diet)

    data["length_m_clean"] = data["length_m"].apply(parse_number)
    data["weight_kg_clean"] = data["weight_kg"].apply(parse_number)
    data["height_m_clean"] = data["height_m"].apply(parse_number)
    data["first_discovered_year"] = data["first_discovered"].apply(parse_year)

    data["locomotion_clean"] = data["locomotion"].apply(clean_locomotion)
    data["intelligence_clean"] = data["intelligence_level"].apply(clean_intelligence)
    data["geological_period_clean"] = data["geological_period"].apply(clean_period)
    data["lived_in_clean"] = data["lived_in"].apply(clean_region)

    for col in ["length_m_clean", "weight_kg_clean", "height_m_clean"]:
        data.loc[data[col] < 0, col] = np.nan

    data.loc[
        (data["first_discovered_year"] < 1800) | (data["first_discovered_year"] > 2030),
        "first_discovered_year"
    ] = np.nan

    data["size_index"] = data["length_m_clean"] * data["height_m_clean"]
    data["mass_per_meter"] = data["weight_kg_clean"] / (data["length_m_clean"] + 1)

    data = data.drop_duplicates()
    data = data.dropna(subset=["diet_clean"])

    class_counts = data["diet_clean"].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    data = data[data["diet_clean"].isin(valid_classes)].copy()

    numeric_features = [
        "length_m_clean",
        "weight_kg_clean",
        "height_m_clean",
        "first_discovered_year",
        "size_index",
        "mass_per_meter"
    ]

    categorical_features = [
        "locomotion_clean",
        "intelligence_clean",
        "geological_period_clean",
        "lived_in_clean"
    ]

    X = data[numeric_features + categorical_features]
    y = data["diet_clean"]

    report = {
        "rows_before": len(df),
        "rows_after": len(data),
        "class_distribution": data["diet_clean"].value_counts().to_dict()
    }

    return data, X, y, numeric_features, categorical_features, report


def build_dino_nn(input_dim, n_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.30),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(32, activation="relu"),
        Dropout(0.20),

        Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_dinosaur_mode(test_size, epochs, batch_size):
    st.title("Dinosaur Dataset: Ensemble vs Neural Network")
    st.write("ใช้ไฟล์ dinoDatasetCSV_dirty.csv เพื่อจำแนกประเภทอาหารของไดโนเสาร์ (diet)")

    uploaded_file = st.sidebar.file_uploader("Upload Dinosaur CSV", type=["csv"], key="dino_csv")
    use_local = st.sidebar.checkbox("Use local dinosaur file: dinoDatasetCSV_dirty.csv", value=False)

    df = None
    source_name = None

    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            source_name = uploaded_file.name
        elif use_local:
            df = pd.read_csv("dinoDatasetCSV_dirty.csv")
            source_name = "dinoDatasetCSV_dirty.csv"
    except Exception as e:
        st.error(f"อ่านไฟล์ Dinosaur ไม่ได้: {e}")

    if df is None:
        st.info("กรุณาอัปโหลดไฟล์ dinoDatasetCSV_dirty.csv หรือเลือก use local dinosaur file")
        st.stop()

    st.subheader(" Raw Dataset")
    st.write(f"Source: **{source_name}**")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    c1, c2 = st.columns(2)
    with c1:
        st.write("Columns")
        st.write(list(df.columns))
    with c2:
        st.write("Missing Values")
        st.dataframe(
            df.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing_count"}),
            use_container_width=True
        )

    st.subheader(" Data Cleaning")
    clean_df, X, y, numeric_features, categorical_features, report = prepare_dino_data(df)

    st.write(f"จำนวนแถวก่อน clean: {report['rows_before']}")
    st.write(f"จำนวนแถวหลัง clean: {report['rows_after']}")
    st.dataframe(
        pd.DataFrame(list(report["class_distribution"].items()), columns=["diet", "count"]),
        use_container_width=True
    )
    st.dataframe(clean_df.head(), use_container_width=True)

    csv_data = clean_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned Dinosaur CSV", csv_data, "dino_cleaned_dataset.csv", "text/csv")

    st.subheader(" Features and Target")
    st.write("Target: **diet_clean**")
    st.write("Numeric features:", numeric_features)
    st.write("Categorical features:", categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot())
        ]), categorical_features)
    ])

    st.subheader(" Train Models")

    ensemble_model = Pipeline([
        ("preprocess", preprocessor),
        ("model", VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=250, random_state=42)),
                ("et", ExtraTreesClassifier(n_estimators=250, random_state=42)),
                ("gb", GradientBoostingClassifier(random_state=42))
            ],
            voting="soft"
        ))
    ])

    with st.spinner("Training Dinosaur Ensemble model..."):
        ensemble_model.fit(X_train, y_train)
        ensemble_pred = ensemble_model.predict(X_test)
        ensemble_metrics = evaluate_classification(y_test, ensemble_pred)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    X_train_nn = preprocessor.fit_transform(X_train)
    X_test_nn = preprocessor.transform(X_test)

    if not isinstance(X_train_nn, np.ndarray):
        X_train_nn = X_train_nn.toarray()
    if not isinstance(X_test_nn, np.ndarray):
        X_test_nn = X_test_nn.toarray()

    nn_model = build_dino_nn(X_train_nn.shape[1], len(label_encoder.classes_))

    early_stop = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)

    with st.spinner("Training Dinosaur Neural Network model..."):
        history = nn_model.fit(
            X_train_nn,
            y_train_enc,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        nn_prob = nn_model.predict(X_test_nn, verbose=0)
        nn_pred_enc = np.argmax(nn_prob, axis=1)
        nn_pred = label_encoder.inverse_transform(nn_pred_enc)
        nn_metrics = evaluate_classification(y_test, nn_pred)

    st.subheader(" Evaluation Results")
    result_df = pd.DataFrame([
        {"Model": "Dinosaur Ensemble Voting Classifier", **ensemble_metrics},
        {"Model": "Dinosaur Custom Neural Network", **nn_metrics}
    ])
    st.dataframe(result_df, use_container_width=True)

   

    st.subheader(" Summary")

    st.markdown("### ภาพรวม")
    st.write("""
    งานนี้ใช้ชุดข้อมูลไดโนเสาร์จากไฟล์ dinoDatasetCSV_dirty.csv เพื่อพัฒนาโมเดลจำแนกประเภทอาหารของไดโนเสาร์
    โดยกำหนดตัวแปรเป้าหมายเป็น diet_clean ซึ่งแบ่งเป็นกลุ่ม เช่น Herbivore, Carnivore และ Omnivore
    การพัฒนาแบ่งเป็น 2 แนวทาง คือ Machine Learning และ Neural Network เพื่อเปรียบเทียบประสิทธิภาพของโมเดล
    """)

    st.markdown("### Machine Learning Model")
    st.write("""
    โมเดล Machine Learning ที่ใช้คือ Ensemble Voting Classifier ซึ่งรวมอัลกอริทึม Random Forest,
    Extra Trees และ Gradient Boosting เข้าด้วยกันแบบ soft voting เพื่อเพิ่มความแม่นยำและลดความลำเอียงของโมเดลเดี่ยว

    ขั้นตอนพัฒนาเริ่มจากการทำความสะอาดข้อมูล เช่น แปลงข้อความให้อยู่ในรูปแบบมาตรฐาน
    ดึงค่าตัวเลขจากคอลัมน์ความยาว น้ำหนัก ความสูง และปีที่ค้นพบ
    จากนั้นสร้างตัวแปรใหม่ เช่น size_index และ mass_per_meter แล้วแยกข้อมูลเป็นตัวแปรเชิงตัวเลขและเชิงหมวดหมู่

    ก่อนฝึกโมเดล มีการเติมค่าที่หายไปด้วย SimpleImputer
    ปรับมาตรฐานข้อมูลตัวเลขด้วย StandardScaler
    และแปลงข้อมูลหมวดหมู่ด้วย OneHotEncoder
    จากนั้นจึงแบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบเพื่อใช้สร้างและประเมินโมเดล
    """)

    st.markdown("### Neural Network Model")
    st.write("""
    โมเดล Neural Network ที่ใช้เป็นโครงข่ายประสาทเทียมแบบ Dense Neural Network หรือ Multilayer Perceptron
    ประกอบด้วยชั้น Dense หลายชั้นร่วมกับ Batch Normalization และ Dropout
    เพื่อช่วยให้การเรียนรู้มีเสถียรภาพและลดปัญหา overfitting

    ข้อมูลที่ใช้กับ Neural Network ผ่านขั้นตอน preprocessing แบบเดียวกับ Machine Learning
    โดยแปลงข้อมูลทั้งหมดให้อยู่ในรูปตัวเลขก่อน
    จากนั้นแปลง label ของประเภทอาหารให้เป็นรหัสตัวเลขด้วย LabelEncoder
    แล้วนำไปฝึกโมเดลด้วยฟังก์ชันสูญเสีย sparse_categorical_crossentropy และ optimizer แบบ Adam

    ในระหว่างการฝึก มีการใช้ EarlyStopping เพื่อหยุดการฝึกเมื่อโมเดลไม่พัฒนาต่อ
    และ ReduceLROnPlateau เพื่อลดค่า learning rate อัตโนมัติเมื่อผลลัพธ์เริ่มคงที่
    """)

    st.markdown("### การประเมินผล")
    st.write("""
   จากผลการทดลองพบว่าโมเดล Dinosaur Ensemble Voting Classifier มีประสิทธิภาพดีกว่า Dinosaur Custom Neural Network ในทุกตัวชี้วัด ได้แก่ Accuracy, Precision, Recall และ F1-score จึงสรุปได้ว่าโมเดล Ensemble เหมาะสมกับชุดข้อมูลไดโนเสาร์นี้มากกว่า
    """)

    st.markdown("### แหล่งข้อมูล")
    st.write("""
    ข้อมูลที่ใช้ในส่วนนี้มาจากไฟล์ https://www.kaggle.com/datasets/canozensoy/dinosaur-genera-dataset ที่นำมาทำให้ไม่สะอาดโดย chat gpt
    ซึ่งเป็นชุดข้อมูลหลักสำหรับการทดลองและพัฒนาโมเดลจำแนกประเภทอาหารของไดโนเสาร์
    """)


# =========================================================
# JELLYFISH FUNCTIONS
# =========================================================
def extract_zip_to_temp(zip_source):
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "dataset.zip")

    if isinstance(zip_source, str):
        with open(zip_source, "rb") as src, open(zip_path, "wb") as dst:
            dst.write(src.read())
    else:
        with open(zip_path, "wb") as dst:
            dst.write(zip_source.read())

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    return temp_dir


def find_image_files(root_dir):
    image_paths = []
    labels = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            ext = Path(file).suffix
            if ext in VALID_EXT:
                full_path = os.path.join(root, file)
                label = Path(full_path).parent.name

                if label.lower() in ["train", "test", "valid", "val"]:
                    continue

                image_paths.append(full_path)
                labels.append(label)

    return pd.DataFrame({
        "image_path": image_paths,
        "label": labels
    })


def try_find_dataset_root(root_dir):
    df = find_image_files(root_dir)
    if not df.empty:
        return root_dir, df

    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for sub in subdirs:
        df = find_image_files(sub)
        if not df.empty:
            return sub, df

    return root_dir, pd.DataFrame(columns=["image_path", "label"])


def load_image_for_cnn(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    return np.array(img).astype("float32") / 255.0


def extract_color_features(path, bins=8):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)

    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))

    hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 255), density=True)
    hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 255), density=True)
    hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 255), density=True)

    return np.concatenate([mean_rgb, std_rgb, hist_r, hist_g, hist_b])


def build_feature_table(df):
    return np.array([extract_color_features(p) for p in df["image_path"]])


def build_jelly_nn(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_jellyfish_mode(test_size, epochs, batch_size):
    st.title("Jellyfish Dataset: Ensemble vs Neural Network")
    st.write("ใช้รูปภาพแมงกะพรุนเพื่อทำ classification, species identification และ color analysis")

    uploaded_zip = st.sidebar.file_uploader("Upload Jellyfish ZIP", type=["zip"], key="jelly_zip")
    use_local_zip = st.sidebar.checkbox("Use local jellyfish ZIP: jellyfish.zip", value=False)
    # use_local_folder = st.sidebar.checkbox("Use local jellyfish folder: jellyfish_dataset", value=False)

    df = None
    source_name = None

    try:
        if uploaded_zip is not None:
            root_dir = extract_zip_to_temp(uploaded_zip)
            root_dir, df = try_find_dataset_root(root_dir)
            source_name = "uploaded jellyfish ZIP"

        elif use_local_zip:
            local_zip_path = "jellyfish.zip"
            if os.path.exists(local_zip_path):
                root_dir = extract_zip_to_temp(local_zip_path)
                root_dir, df = try_find_dataset_root(root_dir)
                source_name = "jellyfish.zip"
            else:
                st.error("ไม่พบไฟล์ jellyfish.zip")
                st.stop()

        

    except Exception as e:
        st.error(f"โหลด Jellyfish dataset ไม่ได้: {e}")
        st.stop()

    if df is None or df.empty:
        st.info("กรุณาอัปโหลด jellyfish.zip หรือเลือก use local jellyfish file/folder")
        st.stop()

    st.subheader(" Raw Dataset")
    st.write(f"Source: **{source_name}**")
    st.write("จำนวนรูปทั้งหมด:", len(df))
    st.write("จำนวน species:", df["label"].nunique())

    c1, c2 = st.columns(2)
    with c1:
        class_df = df["label"].value_counts().reset_index()
        class_df.columns = ["species", "count"]
        st.dataframe(class_df, use_container_width=True)
    with c2:
        st.write("Species labels")
        st.write(sorted(df["label"].unique().tolist()))

    st.bar_chart(df["label"].value_counts())

    sample_n = min(4, len(df))
    sample_rows = df.sample(sample_n, random_state=42)
    cols = st.columns(sample_n)
    for i, (_, row) in enumerate(sample_rows.iterrows()):
        with cols[i]:
            st.image(row["image_path"], caption=row["label"], use_container_width=True)

    st.subheader(" Color Analysis")
    color_rows = []
    for _, row in df.iterrows():
        img = Image.open(row["image_path"]).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img)
        color_rows.append({
            "label": row["label"],
            "mean_red": arr[:, :, 0].mean(),
            "mean_green": arr[:, :, 1].mean(),
            "mean_blue": arr[:, :, 2].mean()
        })

    color_df = pd.DataFrame(color_rows)
    species_color = color_df.groupby("label")[["mean_red", "mean_green", "mean_blue"]].mean().reset_index()
    st.dataframe(species_color, use_container_width=True)
    st.bar_chart(species_color.set_index("label"))

    st.subheader(" Features and Target")
    st.write("Target: **species label ของรูปภาพ**")
    st.write("Model 1 ใช้ color-based features")
    st.write("Model 2 ใช้รูปภาพโดยตรงด้วย CNN")

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df["label"])
    class_names = list(label_encoder.classes_)

    train_df, test_df, y_train, y_test = train_test_split(
        df, y_all, test_size=test_size, random_state=42, stratify=y_all
    )

    st.subheader(" Train Models")

    with st.spinner("Extracting color features for Jellyfish Ensemble..."):
        X_train_feat = build_feature_table(train_df)
        X_test_feat = build_feature_table(test_df)

    ensemble_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=250, random_state=42)),
                ("et", ExtraTreesClassifier(n_estimators=250, random_state=42)),
                ("gb", GradientBoostingClassifier(random_state=42))
            ],
            voting="soft"
        ))
    ])

    with st.spinner("Training Jellyfish Ensemble model..."):
        ensemble_model.fit(X_train_feat, y_train)
        ensemble_pred = ensemble_model.predict(X_test_feat)
        ensemble_metrics = evaluate_classification(y_test, ensemble_pred)

    with st.spinner("Loading images for Jellyfish Neural Network..."):
        X_train_img = np.array([load_image_for_cnn(p) for p in train_df["image_path"]])
        X_test_img = np.array([load_image_for_cnn(p) for p in test_df["image_path"]])

    nn_model = build_jelly_nn(len(class_names))
    early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)

    with st.spinner("Training Jellyfish Neural Network model..."):
        history = nn_model.fit(
            X_train_img,
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        nn_prob = nn_model.predict(X_test_img, verbose=0)
        nn_pred = np.argmax(nn_prob, axis=1)
        nn_metrics = evaluate_classification(y_test, nn_pred)

    st.subheader(" Evaluation Results")
    result_df = pd.DataFrame([
        {"Model": "Jellyfish Ensemble Voting Classifier", **ensemble_metrics},
        {"Model": "Jellyfish Custom CNN", **nn_metrics}
    ])
    st.dataframe(result_df, use_container_width=True)

    

    st.subheader(" Species Identification Summary")
    st.write("""
    ในงานนี้ jellyfish classification และ species identification เป็นงานเดียวกัน
    เพราะ label ของแต่ละรูปคือชื่อ species ของแมงกะพรุนโดยตรง
    """)

    st.subheader(" Try Your Own Prediction")
    uploaded_image = st.file_uploader("อัปโหลดรูปแมงกะพรุน 1 รูป", type=["jpg", "jpeg", "png"], key="jelly_predict")
    if uploaded_image is not None:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

        img_array = np.array(pil_img.resize(IMG_SIZE)).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        temp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        pil_img.save(temp_img_file.name)

        color_feat = extract_color_features(temp_img_file.name)
        color_feat = np.expand_dims(color_feat, axis=0)

        ensemble_idx = ensemble_model.predict(color_feat)[0]
        ensemble_result = label_encoder.inverse_transform([ensemble_idx])[0]

        nn_idx = np.argmax(nn_model.predict(img_array, verbose=0), axis=1)[0]
        nn_result = label_encoder.inverse_transform([nn_idx])[0]

        r1, r2 = st.columns(2)
        with r1:
            st.metric("Ensemble Prediction", ensemble_result)
        with r2:
            st.metric("Neural Network Prediction", nn_result)

    st.subheader(" Summary")

    st.markdown("### ภาพรวม")
    st.write("""
    งานนี้ใช้ชุดข้อมูลภาพแมงกะพรุนเพื่อพัฒนาโมเดลจำแนกชนิดของแมงกะพรุน
    โดยกำหนด label จากชื่อโฟลเดอร์ของแต่ละ species
    และพัฒนาโมเดล 2 แนวทาง คือ Machine Learning และ Neural Network
    เพื่อเปรียบเทียบประสิทธิภาพในการจำแนกภาพ
    """)

    st.markdown("### Machine Learning Model")
    st.write("""
    โมเดล Machine Learning ที่ใช้คือ Ensemble Voting Classifier
    ซึ่งรวม Random Forest, Extra Trees และ Gradient Boosting แบบ soft voting
    โดยโมเดลนี้ไม่ได้ใช้ภาพโดยตรง แต่ใช้คุณลักษณะที่สกัดจากสีของภาพแทน

    ขั้นตอนเตรียมข้อมูลเริ่มจากการอ่านไฟล์ภาพทั้งหมดจาก ZIP หรือโฟลเดอร์
    แล้วใช้ชื่อโฟลเดอร์เป็น label ของแต่ละภาพ
    จากนั้นปรับขนาดภาพให้เป็นขนาดเดียวกัน และสกัดคุณลักษณะด้านสี เช่น
    ค่าเฉลี่ย RGB, ส่วนเบี่ยงเบนมาตรฐานของแต่ละช่องสี และ histogram ของสี
    คุณลักษณะเหล่านี้ถูกนำมาใช้เป็น input ของโมเดล Machine Learning

    ก่อนฝึกโมเดล มีการเติมค่าที่หายไปด้วย SimpleImputer
    และปรับมาตรฐานข้อมูลด้วย StandardScaler
    แล้วจึงฝึก Ensemble Model เพื่อจำแนก species ของแมงกะพรุน
    """)

    st.markdown("### Neural Network Model")
    st.write("""
    โมเดล Neural Network ที่ใช้เป็น Convolutional Neural Network (CNN)
    ซึ่งเหมาะกับงานจำแนกภาพ เพราะสามารถเรียนรู้ลักษณะสำคัญของภาพได้โดยตรง
    เช่น รูปร่าง ลวดลาย สี และโครงสร้างของแมงกะพรุน

    โครงสร้างของโมเดลประกอบด้วยชั้น Conv2D, MaxPooling2D, Flatten, Dense และ Dropout
    โดยรับภาพขนาด 128x128 พิกเซลแบบ RGB เป็นข้อมูลนำเข้า
    จากนั้นแปลงค่าพิกเซลให้อยู่ในช่วง 0 ถึง 1 เพื่อช่วยให้โมเดลเรียนรู้ได้ดีขึ้น

    ในระหว่างการฝึก มีการใช้ EarlyStopping เพื่อป้องกัน overfitting
    และ ReduceLROnPlateau เพื่อลด learning rate เมื่อผลลัพธ์บน validation set ไม่ดีขึ้น
    แล้วนำผลลัพธ์ที่ได้ไปเปรียบเทียบกับโมเดล Machine Learning
    """)

    st.markdown("### การประเมินผล")
    st.write("""
    จากผลการทดลองพบว่าโมเดล Jellyfish Ensemble Voting Classifier ให้ผลใกล้เคียง Jellyfish Custom CNN ต่างแค่เล็กน้อยในทุกตัวชี้วัด ได้แก่ Accuracy, Precision, Recall และ F1-score อย่างไรก็ตาม ค่าประสิทธิภาพของทั้งสองโมเดลอยู่ในระดับสูงและมีความใกล้เคียงกันมาก แสดงให้เห็นว่าทั้งแนวทาง Machine Learning และ Neural Network สามารถจำแนกชนิดของแมงกะพรุนได้อย่างมีประสิทธิภาพ
    """)

    st.markdown("### แหล่งข้อมูล")
    st.write("""
    ข้อมูลที่ใช้ในส่วนนี้มาจากhttps://www.kaggle.com/datasets/anshtanwar/jellyfish-types
    โดยใช้ภาพของแมงกะพรุนแต่ละชนิดเป็นข้อมูลหลักในการฝึกและทดสอบโมเดล
    """)


# =========================================================
# MAIN APP
# =========================================================
st.sidebar.header("Global Settings")
dataset_mode = st.sidebar.selectbox(
    "Choose dataset mode",
    ["Dinosaur", "Jellyfish"]
)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
epochs = st.sidebar.slider("NN Epochs", 5, 200, 30, 5)
batch_size = st.sidebar.selectbox("NN Batch size", [8, 16, 32, 64], index=2)

if dataset_mode == "Dinosaur":
    run_dinosaur_mode(test_size, epochs, batch_size)
else:
    run_jellyfish_mode(test_size, epochs, batch_size)