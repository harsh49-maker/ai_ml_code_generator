import os
import re
import json
import pandas as pd
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool

load_dotenv()

st.set_page_config(
    page_title="AI ML Code Generator",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #f7f8fa;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 48px;
    font-size: 16px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">🤖 AI ML Code Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent Python Machine Learning Pipeline Creator</div>',
    unsafe_allow_html=True
)

st.divider()

if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

if "agent_output" not in st.session_state:
    st.session_state.agent_output = ""

if "logs" not in st.session_state:
    st.session_state.logs = []


@tool
def inspect_dataset(file_path: str) -> str:
    """
    Analyze a CSV dataset and return useful information for machine
    learning code generation.

    Args:
        file_path: Path to the CSV dataset.

    Returns:
        Dataset shape, columns, data types, missing values,
        numerical columns, categorical columns and sample rows.
    """

    try:

        df = pd.read_csv(file_path)

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        missing_values = df.isnull().sum()

        missing_values = {
            column: int(value)
            for column, value in missing_values.items()
            if value > 0
        }

        information = {
            "rows": int(df.shape[0]),
            "columns_count": int(df.shape[1]),
            "columns": df.columns.tolist(),

            "data_types": {
                column: str(dtype)
                for column, dtype in df.dtypes.items()
            },

            "numeric_columns": numeric_columns,

            "categorical_columns": categorical_columns,

            "missing_values": missing_values,

            "duplicate_rows": int(df.duplicated().sum()),

            "sample_rows": df.head(5).to_dict(
                orient="records"
            )
        }

        return json.dumps(
            information,
            indent=2,
            default=str
        )

    except Exception as e:

        return f"ERROR: {str(e)}"

def create_agent():

    api_key = "AQ.Ab8RN6KKwn-FkMU18A2a1dGbz9K_kMYNXOAut3fxWzTN0r5vhQ"

    model = LiteLLMModel(
        model_id="gemini/gemini-3.6-flash",
        api_key=api_key,
        temperature=0.1,
        max_tokens=5000
    )

    agent = CodeAgent(
        tools=[
            inspect_dataset
        ],
        model=model,
        max_steps=8,
        verbosity_level=1
    )

    return agent

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


with left_column:

    st.markdown(
        "### 📥 1. Configuration & Data Input"
    )

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx"],
        help="Upload your CSV or Excel dataset."
    )

    dataframe = None
    temp_file_path = None

    if uploaded_file:

        try:

            if uploaded_file.name.endswith(".csv"):

                dataframe = pd.read_csv(
                    uploaded_file
                )

            else:

                dataframe = pd.read_excel(
                    uploaded_file
                )

            st.success(
                f"Dataset loaded: {uploaded_file.name}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Rows",
                    dataframe.shape[0]
                )

            with col2:
                st.metric(
                    "Columns",
                    dataframe.shape[1]
                )

            with col3:
                st.metric(
                    "Missing Values",
                    int(dataframe.isnull().sum().sum())
                )

            with st.expander(
                "👀 Preview Dataset"
            ):

                st.dataframe(
                    dataframe.head(10),
                    use_container_width=True
                )

            os.makedirs(
                "generated",
                exist_ok=True
            )

            temp_file_path = os.path.join(
                "generated",
                uploaded_file.name
            )

            dataframe.to_csv(
                temp_file_path,
                index=False
            )

        except Exception as e:

            st.error(
                f"Unable to read dataset: {e}"
            )

    st.markdown(
        "### 🧠 2. Enter Machine Learning Task"
    )

    prompt = st.text_area(
        "Describe what you want the AI agent to build:",
        placeholder=(
            "Example:\n"
            "Create a step-by-step Python machine learning "
            "pipeline for linear regression using sklearn. "
            "Automatically identify the target column, "
            "perform preprocessing, train the model, "
            "evaluate it using appropriate metrics and "
            "show feature importance."
        ),
        height=180
    )

    st.markdown("#### 💡 Example prompts")

    example1 = st.button(
        "Linear Regression"
    )

    example2 = st.button(
        "Classification Model"
    )

    example3 = st.button(
        "Complete ML Pipeline"
    )

    if example1:

        prompt = """
Create a complete single linear regression machine
learning pipeline using scikit-learn.

Automatically inspect the dataset and identify a suitable
target column.

Include:

1. Data loading
2. Data preprocessing
3. Train-test split
4. LinearRegression
5. Model training
6. Predictions
7. MAE
8. MSE
9. RMSE
10. R2 score
11. Actual vs predicted visualization

Generate clean production-style Python code.
"""

    elif example2:

        prompt = """
Build a classification machine learning pipeline.

Automatically inspect the dataset and identify a suitable
target column.

Perform:

1. Data preprocessing
2. Missing value handling
3. Encoding categorical variables
4. Train-test split
5. Classification model
6. Model training
7. Predictions
8. Accuracy
9. Precision
10. Recall
11. F1 score
12. Confusion matrix

Use scikit-learn.
"""

    elif example3:

        prompt = """
Create a complete end-to-end machine learning pipeline.

Analyze the uploaded dataset first.

Automatically determine whether the problem is:

- regression
- binary classification
- multiclass classification

Then generate code containing:

1. Data loading
2. EDA
3. Missing value handling
4. Duplicate handling
5. Feature preprocessing
6. Encoding
7. Train-test split
8. Model selection
9. Model training
10. Evaluation
11. Visualization
12. Model saving

Use clean production-ready Python code.
"""

    generate_button = st.button(
        "🚀 Generate ML Analytics",
        type="primary"
    )


with right_column:

    st.markdown(
        "### 💻 3. Agent Execution & Generated Script"
    )

    if generate_button:

        if uploaded_file is None:

            st.warning(
                "Please upload a CSV or Excel dataset first."
            )

        elif not prompt.strip():

            st.warning(
                "Please enter an ML task."
            )

        else:

            st.session_state.logs = []

            try:


                st.session_state.logs.append(
                    "📂 Dataset received"
                )


                st.session_state.logs.append(
                    "🔍 Inspecting dataset..."
                )


                agent = create_agent()

                st.session_state.logs.append(
                    "🤖 AI agent initialized"
                )

           
                agent_prompt = f"""
You are an expert Machine Learning Engineer and
Python code-generation agent.

The user uploaded this dataset:

FILE PATH:
{temp_file_path}

USER REQUEST:
{prompt}

Your workflow:

STEP 1
Use the inspect_dataset tool to inspect the uploaded
dataset.

STEP 2
Understand the dataset columns, data types,
missing values and structure.

STEP 3
Determine the appropriate machine learning approach
from the user's request.

STEP 4
Generate complete Python code.

IMPORTANT:

The generated code must:

- use pandas
- use numpy when required
- use scikit-learn
- load the dataset using:

    pd.read_csv("DATASET_PATH")

- automatically handle missing values where appropriate
- encode categorical columns when required
- split data into training and testing sets
- train the requested ML model
- evaluate the model
- print useful metrics
- include useful visualizations
- use clear comments
- be executable Python code

IMPORTANT FOR TARGET:

If the user explicitly specifies a target column,
use that target.

If the user does not specify a target column,
infer a reasonable target based on the dataset and
EXPLAIN your assumption.

IMPORTANT:

Return the final answer as Python code only.
Do not use markdown explanations outside the code.

Do not use databases.

Do not use SQLite.

Do not invent columns that do not exist.
"""

                st.session_state.logs.append(
                    "🧠 Agent analyzing ML task..."
                )

                result = agent.run(
                    agent_prompt
                )

                st.session_state.logs.append(
                    "⚙️ ML pipeline generated"
                )

                generated_code = str(result)

                # Remove markdown code fences
                generated_code = re.sub(
                    r"```python",
                    "",
                    generated_code,
                    flags=re.IGNORECASE
                )

                generated_code = re.sub(
                    r"```",
                    "",
                    generated_code
                )

                generated_code = generated_code.strip()

                st.session_state.generated_code = (
                    generated_code
                )

                st.session_state.agent_output = (
                    str(result)
                )

                st.session_state.logs.append(
                    "✅ Generation completed"
                )

            except Exception as e:

                st.error(
                    f"Agent error: {str(e)}"
                )

                st.session_state.logs.append(
                    f"❌ Error: {str(e)}"
                )

    if st.session_state.logs:

        st.markdown(
            "#### 📋 Execution Logs"
        )

        for log in st.session_state.logs:

            st.write(log)


    if st.session_state.generated_code:

        st.markdown(
            "#### 🧠 AI Workspace Output"
        )

        st.code(
            st.session_state.generated_code,
            language="python",
            line_numbers=True
        )

        st.download_button(
            label="⬇️ Download Generated Python Code",
            data=st.session_state.generated_code,
            file_name="generated_ml_pipeline.py",
            mime="text/x-python"
        )


        st.success(
            "ML pipeline generated successfully!"
        )


st.divider()

st.markdown(
    """
<center>

**🤖 AI ML Code Generator**

Built with Streamlit + Python + smolagents + LLM + Scikit-learn

</center>
""",
    unsafe_allow_html=True
)