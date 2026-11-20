import json
import pandas as pd
import streamlit as st

from dataset import (
    load_dataset,
    validate_dataset,
    clean_dataset,
    get_column_information,
)
from model import train_models, predict_students
from recommender import generate_recommendation
from blockchain import Blockchain


st.set_page_config(
    page_title="Smart Education AI",
    page_icon="🎓",
    layout="wide",
)


defaults = {
    "dataset": None,
    "dataset_name": None,
    "training": None,
    "predictions": None,
    "selected_model": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource
def get_blockchain():
    return Blockchain()


blockchain = get_blockchain()


st.sidebar.title("🎓 Smart Education AI")
st.sidebar.caption("AI + Blockchain Research Prototype")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset Upload",
        "AI Training",
        "Student Predictions",
        "Blockchain Security",
        "Model Evaluation",
    ],
)
st.sidebar.divider()
st.sidebar.write("Research Project")
st.sidebar.caption("Python | Streamlit | Machine Learning")


if page == "Dashboard":
    st.title("🎓 Smart Education System")
    st.write(
        "An experimental framework combining Artificial Intelligence "
        "and Blockchain for personalized and secure education."
    )
    st.divider()

    df = st.session_state.dataset
    training = st.session_state.training

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df) if df is not None else 0)
    col2.metric("Dataset Features", len(df.columns) if df is not None else 0)
    col3.metric("Trained Models", len(training["models"]) if training else 0)
    col4.metric("Blockchain Blocks", len(blockchain.get_chain()))

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("🤖 Artificial Intelligence")
        st.write("- Student performance prediction")
        st.write("- Machine Learning model comparison")
        st.write("- Personalized recommendations")
    with right:
        st.subheader("🔐 Blockchain Security")
        st.write("- SHA-256 cryptographic hashing")
        st.write("- Hash-linked academic records")
        st.write("- Tampering detection")

    if df is None:
        st.info("Upload an educational dataset to begin.")


elif page == "Dataset Upload":
    st.title("📁 Dataset Management")
    uploaded = st.file_uploader("Upload educational dataset", type=["csv"])

    if uploaded is not None:
        try:
            new_df = load_dataset(uploaded)
            if st.session_state.dataset_name != uploaded.name:
                st.session_state.training = None
                st.session_state.predictions = None
                st.session_state.selected_model = None

            st.session_state.dataset = new_df
            st.session_state.dataset_name = uploaded.name
            st.success("Dataset uploaded successfully!")
        except Exception as error:
            st.error(f"Dataset error: {error}")

    df = st.session_state.dataset
    if df is not None:
        report = validate_dataset(df)
        st.divider()
        st.subheader("Dataset Overview")
        a, b, c, d = st.columns(4)
        a.metric("Rows", report["rows"])
        b.metric("Columns", report["columns"])
        c.metric("Missing Values", report["missing_values"])
        d.metric("Duplicate Rows", report["duplicate_rows"])

        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Column Information")
        st.dataframe(get_column_information(df), use_container_width=True)

        st.subheader("Prepare Dataset")
        remove_duplicates = st.checkbox(
            "Remove duplicate rows", value=True
        )
        cleaned = clean_dataset(df, remove_duplicates)

        st.write(
            f"Original records: {len(df)} | "
            f"Prepared records: {len(cleaned)}"
        )
        st.dataframe(cleaned.head(10), use_container_width=True)

        st.download_button(
            "Download Cleaned Dataset",
            data=cleaned.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_student_dataset.csv",
            mime="text/csv",
        )

        if st.button("Apply Cleaned Dataset"):
            st.session_state.dataset = cleaned
            st.session_state.training = None
            st.session_state.predictions = None
            st.success("Cleaned dataset applied!")
            st.rerun()


elif page == "AI Training":
    st.title("🤖 Machine Learning Training")
    df = st.session_state.dataset

    if df is None:
        st.warning("Upload a dataset first.")
    else:
        default_target_index = (
            list(df.columns).index("G3") if "G3" in df.columns else 0
        )
        target = st.selectbox(
            "Select prediction target",
            df.columns.tolist(),
            index=default_target_index,
        )

        threshold = 10.0
        if pd.api.types.is_numeric_dtype(df[target]):
            threshold = st.number_input(
                "Minimum score for On Track",
                min_value=0.0,
                value=10.0,
            )

        default_excluded = []
        if target == "G3":
            default_excluded = [
                col for col in ["G1", "G2"] if col in df.columns
            ]

        excluded = st.multiselect(
            "Exclude input columns",
            options=[col for col in df.columns if col != target],
            default=default_excluded,
        )
        test_size = st.slider(
            "Test data percentage",
            min_value=10,
            max_value=40,
            value=20,
        )

        if st.button("🚀 Train All Models", type="primary"):
            try:
                with st.spinner("Training machine learning models..."):
                    training = train_models(
                        df=df,
                        target=target,
                        threshold=threshold,
                        test_size=test_size / 100,
                        excluded_columns=excluded,
                    )
                st.session_state.training = training
                st.session_state.predictions = None
                st.success("Training completed!")
            except Exception as error:
                st.error(f"Training failed: {error}")

    training = st.session_state.training
    if training:
        st.subheader("Model Comparison")
        st.dataframe(
            training["metrics"].style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
            }),
            use_container_width=True,
        )
        best = training["metrics"].sort_values(
            "F1-Score", ascending=False
        ).iloc[0]
        st.success(f"Best model by weighted F1-score: {best['Model']}")


elif page == "Student Predictions":
    st.title("🎯 Student Predictions")
    df = st.session_state.dataset
    training = st.session_state.training

    if df is None or training is None:
        st.warning("Upload a dataset and train the models first.")
    else:
        model_name = st.selectbox(
            "Select trained model",
            list(training["models"].keys()),
        )

        if st.button("Generate Predictions"):
            try:
                model = training["models"][model_name]
                labels = predict_students(
                    model, df, training["features"], training["encoder"]
                )
                results = df.copy()
                results["AI Prediction"] = labels
                results["Learning Recommendation"] = [
                    " ".join(generate_recommendation(row, label))
                    for (_, row), label in zip(results.iterrows(), labels)
                ]
                st.session_state.predictions = results
                st.session_state.selected_model = model_name
                st.success("Predictions generated successfully!")
            except Exception as error:
                st.error(f"Prediction error: {error}")

        predictions = st.session_state.predictions
        if predictions is not None:
            st.subheader("Prediction Results")
            st.dataframe(predictions, use_container_width=True)
            st.caption(
                "These displayed predictions are generated for the uploaded "
                "dataset. Use the held-out test metrics under Model Evaluation "
                "to assess generalization; these are not independent predictions."
            )
            st.subheader("Prediction Distribution")
            st.bar_chart(predictions["AI Prediction"].value_counts())

            st.download_button(
                "Download Predictions",
                data=predictions.to_csv(index=False).encode("utf-8"),
                file_name="student_predictions.csv",
                mime="text/csv",
            )

            st.divider()
            st.subheader("Secure Records")
            if st.button("🔐 Add Records to Blockchain"):
                for index, row in predictions.iterrows():
                    record = {
                        "student_record": str(index),
                        "prediction": str(row["AI Prediction"]),
                        "recommendation": str(
                            row["Learning Recommendation"]
                        ),
                        "model": st.session_state.selected_model,
                        "dataset": st.session_state.dataset_name,
                    }
                    blockchain.add_block(record)

                st.success(
                    f"{len(predictions)} records added to the local hash chain."
                )


elif page == "Blockchain Security":
    st.title("🔐 Blockchain Security")
    chain = blockchain.get_chain()
    st.metric("Total Blocks", len(chain))

    if chain:
        valid, message = blockchain.verify_chain()
        if valid:
            st.success(message)
        else:
            st.error(message)

        st.subheader("Blockchain Records")
        for block in reversed(chain):
            with st.expander(
                f"Block {block['index']} | {block['timestamp']}"
            ):
                st.json(block)

        if st.button("Verify Blockchain"):
            valid, message = blockchain.verify_chain()
            if valid:
                st.success(message)
            else:
                st.error(message)

        st.download_button(
            "Download Blockchain JSON",
            data=json.dumps(chain, indent=4, ensure_ascii=False),
            file_name="blockchain_records.json",
            mime="application/json",
        )
    else:
        st.info("No blockchain records available.")

    st.caption(
        "This prototype uses a locally stored hash-linked chain. "
        "It does not implement distributed consensus."
    )


elif page == "Model Evaluation":
    st.title("📊 Model Evaluation")
    training = st.session_state.training

    if training is None:
        st.warning("Train the models first.")
    else:
        metrics = training["metrics"]
        st.subheader("Evaluation Results")
        st.dataframe(metrics, use_container_width=True)

        for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            st.subheader(f"{metric} Comparison")
            st.bar_chart(metrics.set_index("Model")[metric])

        st.download_button(
            "Download Evaluation Results",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="model_evaluation.csv",
            mime="text/csv",
        )
