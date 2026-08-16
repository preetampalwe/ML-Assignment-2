import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Spambase ML Classifier",
    page_icon="📧",
    layout="wide"
)


# ============================================================
# Expected Spambase Feature Columns
# ============================================================

EXPECTED_FEATURES = [
    'word_freq_make',
    'word_freq_address',
    'word_freq_all',
    'word_freq_3d',
    'word_freq_our',
    'word_freq_over',
    'word_freq_remove',
    'word_freq_internet',
    'word_freq_order',
    'word_freq_mail',
    'word_freq_receive',
    'word_freq_will',
    'word_freq_people',
    'word_freq_report',
    'word_freq_addresses',
    'word_freq_free',
    'word_freq_business',
    'word_freq_email',
    'word_freq_you',
    'word_freq_credit',
    'word_freq_your',
    'word_freq_font',
    'word_freq_000',
    'word_freq_money',
    'word_freq_hp',
    'word_freq_hpl',
    'word_freq_george',
    'word_freq_650',
    'word_freq_lab',
    'word_freq_labs',
    'word_freq_telnet',
    'word_freq_857',
    'word_freq_data',
    'word_freq_415',
    'word_freq_85',
    'word_freq_technology',
    'word_freq_1999',
    'word_freq_parts',
    'word_freq_pm',
    'word_freq_direct',
    'word_freq_cs',
    'word_freq_meeting',
    'word_freq_original',
    'word_freq_project',
    'word_freq_re',
    'word_freq_edu',
    'word_freq_table',
    'word_freq_conference',
    'char_freq_;',
    'char_freq_(',
    'char_freq_[',
    'char_freq_!',
    'char_freq_$',
    'char_freq_#',
    'capital_run_length_average',
    'capital_run_length_longest',
    'capital_run_length_total'
]

EXPECTED_COLUMNS = EXPECTED_FEATURES + ['spam']


# ============================================================
# Dataset Validation Function
# ============================================================

def validate_dataset(df):

    errors = []

    # --------------------------------------------------------
    # Check number of columns
    # --------------------------------------------------------

    if df.shape[1] != 58:
        errors.append(
            f"Expected 58 columns, but found {df.shape[1]}."
        )

    # --------------------------------------------------------
    # Check for missing columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Check for unexpected columns
    # --------------------------------------------------------

    unexpected_columns = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    if unexpected_columns:
        errors.append(
            "Unexpected columns found: "
            + ", ".join(unexpected_columns)
        )

    # --------------------------------------------------------
    # Stop further validation if column structure is invalid
    # --------------------------------------------------------

    if missing_columns or unexpected_columns:
        return errors

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    missing_values = int(df.isnull().sum().sum())

    if missing_values > 0:
        errors.append(
            f"Dataset contains {missing_values} missing values."
        )

    # --------------------------------------------------------
    # Check that feature columns are numeric
    # --------------------------------------------------------

    non_numeric_features = [
        column
        for column in EXPECTED_FEATURES
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_features:
        errors.append(
            "The following feature columns are not numeric: "
            + ", ".join(non_numeric_features)
        )

    # --------------------------------------------------------
    # Check target values
    # --------------------------------------------------------

    target_values = set(df["spam"].dropna().unique())

    if not target_values.issubset({0, 1}):
        errors.append(
            "The 'spam' column must contain only 0 and 1."
        )

    # --------------------------------------------------------
    # Check that both classes are present
    # --------------------------------------------------------

    if df["spam"].nunique() < 2:
        errors.append(
            "The test dataset must contain both classes: "
            "Non-Spam (0) and Spam (1)."
        )

    return errors



# ============================================================
# Load Saved Models
# ============================================================

@st.cache_resource
def load_models():

    model_directory = "model"

    models = {}

    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "K-Nearest Neighbors": "knn.pkl",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
        "Random Forest": "random_forest.pkl"
    }

    for model_name, filename in model_files.items():

        model_path = os.path.join(
            model_directory,
            filename
        )

        models[model_name] = joblib.load(model_path)

    return models



# ============================================================
# Generate Predictions
# ============================================================

def generate_predictions(model_bundle, X):

    model = model_bundle["model"]
    scaler = model_bundle["scaler"]

    # Apply scaler only for models that used scaling
    if scaler is not None:
        X_processed = scaler.transform(X)
    else:
        X_processed = X

    predictions = model.predict(X_processed)

    # Get probability of Spam (class 1)
    probabilities = model.predict_proba(X_processed)[:, 1]

    return predictions, probabilities



# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(model_bundle, X, y):

    predictions, probabilities = generate_predictions(
        model_bundle,
        X
    )

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, probabilities),
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
        "F1 Score": f1_score(y, predictions),
        "MCC": matthews_corrcoef(y, predictions)
    }

    return metrics, predictions, probabilities



# ============================================================
# Application Title
# ============================================================

st.title("📧 Spambase ML Classification")

st.write(
    "Machine Learning classification of emails as "
    "Spam or Non-Spam."
)


# ============================================================
# Dataset Upload
# ============================================================

st.header("📁 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload the Spambase test dataset (CSV)",
    type=["csv"]
)


# ============================================================
# Process Uploaded Dataset
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        df = pd.read_csv(uploaded_file)

        # ----------------------------------------------------
        # Validate Dataset
        # ----------------------------------------------------

        validation_errors = validate_dataset(df)

        # ----------------------------------------------------
        # Display validation result
        # ----------------------------------------------------

        if validation_errors:

            st.error(
                "❌ Dataset validation failed."
            )

            st.subheader("Validation Errors")

            for error in validation_errors:
                st.write(f"• {error}")

            st.stop()

        # ----------------------------------------------------
        # Dataset is valid
        # ----------------------------------------------------

        st.success(
            "✅ Dataset uploaded and validated successfully!"
        )

        # ----------------------------------------------------
        # Dataset Information
        # ----------------------------------------------------

        st.subheader("Dataset Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Number of Rows",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Number of Features",
                len(EXPECTED_FEATURES)
            )

        with col3:
            st.metric(
                "Non-Spam",
                int((df["spam"] == 0).sum())
            )

        with col4:
            st.metric(
                "Spam",
                int((df["spam"] == 1).sum())
            )


        # ----------------------------------------------------
        # Preview Dataset
        # ----------------------------------------------------

        st.subheader("Preview of Uploaded Data")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        # ====================================================
        # Prepare Features and Target
        # ====================================================

        X = df[EXPECTED_FEATURES]
        y = df["spam"]

        # ====================================================
        # Load Models
        # ====================================================

        models = load_models()

        st.success(
            "All five trained models loaded successfully! ✅"
        )

        # ====================================================
        # Evaluate All Models
        # ====================================================

        results = []
        model_evaluation = {}

        for model_name, model_bundle in models.items():

            metrics, predictions, probabilities = evaluate_model(
                model_bundle,
                X,
                y
            )

            # Store complete evaluation information
            model_evaluation[model_name] = {
                "metrics": metrics,
                "predictions": predictions,
                "probabilities": probabilities
            }

            # Store metrics for comparison table
            results.append({
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "AUC": metrics["AUC"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1 Score"],
                "MCC": metrics["MCC"]
            })

        results_df = pd.DataFrame(results)

        # ====================================================
        # Format Model Comparison Results
        # ====================================================

        comparison_df = results_df.copy()

        metric_columns = [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]

        for column in metric_columns:
            comparison_df[column] = (
                comparison_df[column] * 100
            ).round(2)

        st.subheader("📈 Model Comparison")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # Best Model Summary
        # ====================================================

        best_accuracy_model = results_df.loc[
            results_df["Accuracy"].idxmax(),
            "Model"
        ]

        best_auc_model = results_df.loc[
            results_df["AUC"].idxmax(),
            "Model"
        ]

        best_precision_model = results_df.loc[
            results_df["Precision"].idxmax(),
            "Model"
        ]

        best_recall_model = results_df.loc[
            results_df["Recall"].idxmax(),
            "Model"
        ]

        best_f1_model = results_df.loc[
            results_df["F1 Score"].idxmax(),
            "Model"
        ]

        st.subheader("🏆 Best Performing Models")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Best Accuracy",
                best_accuracy_model
            )

        with col2:
            st.metric(
                "Best AUC",
                best_auc_model
            )

        with col3:
            st.metric(
                "Best Precision",
                best_precision_model
            )

        with col4:
            st.metric(
                "Best Recall",
                best_recall_model
            )

        with col5:
            st.metric(
                "Best F1 Score",
                best_f1_model
            )

        # ====================================================
        # Model Selection
        # ====================================================

        st.subheader("🤖 Select a Classification Model")

        selected_model = st.selectbox(
            "Choose a model for detailed evaluation:",
            list(models.keys())
        )

        st.divider()

        # ====================================================
        # Selected Model Results
        # ====================================================

        selected_results = model_evaluation[selected_model]

        selected_metrics = selected_results["metrics"]
        selected_predictions = selected_results["predictions"]
        selected_probabilities = selected_results["probabilities"]

        st.subheader(
            f"📊 Detailed Results — {selected_model}"
        )

        st.write(
            f"Detailed evaluation results for **{selected_model}**."
        )

        st.write(
            "The following metrics evaluate the selected model on "
            "the uploaded test dataset."
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                "Accuracy",
                f"{selected_metrics['Accuracy'] * 100:.2f}%"
            )

        with metric_col2:
            st.metric(
                "AUC",
                f"{selected_metrics['AUC'] * 100:.2f}%"
            )

        with metric_col3:
            st.metric(
                "Precision",
                f"{selected_metrics['Precision'] * 100:.2f}%"
            )

        metric_col4, metric_col5, metric_col6 = st.columns(3)

        with metric_col4:
            st.metric(
                "Recall",
                f"{selected_metrics['Recall'] * 100:.2f}%"
            )

        with metric_col5:
            st.metric(
                "F1 Score",
                f"{selected_metrics['F1 Score'] * 100:.2f}%"
            )

        with metric_col6:
            st.metric(
                "MCC",
                f"{selected_metrics['MCC'] * 100:.2f}%"
            )

        st.divider()

        # ====================================================
        # Confusion Matrix
        # ====================================================

        st.subheader("📊 Confusion Matrix")

        cm = confusion_matrix(
            y,
            selected_predictions
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Non-Spam", "Actual Spam"],
            columns=["Predicted Non-Spam", "Predicted Spam"]
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(
            f"Confusion Matrix — {selected_model}"
        )

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # ====================================================
        # Classification Report
        # ====================================================

        st.subheader("📋 Classification Report")

        report = classification_report(
            y,
            selected_predictions,
            target_names=["Non-Spam", "Spam"],
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"❌ Unable to read the uploaded CSV file: {e}"
        )

else:

    st.info(
        "Please upload the test_data.csv file to begin."
    )

st.divider()

st.caption(
    "Spambase Binary Classification | "
    "Logistic Regression • Decision Tree • KNN • "
    "Gaussian Naive Bayes • Random Forest"
)
