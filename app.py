{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7a899ace-7487-4a39-ad58-4ace99e54856",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pandas in /opt/anaconda3/lib/python3.12/site-packages (2.2.2)\n",
      "Requirement already satisfied: numpy in /opt/anaconda3/lib/python3.12/site-packages (1.26.4)\n",
      "Requirement already satisfied: scikit-learn in /opt/anaconda3/lib/python3.12/site-packages (1.4.2)\n",
      "Requirement already satisfied: matplotlib in /opt/anaconda3/lib/python3.12/site-packages (3.8.4)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: scipy>=1.6.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.13.1)\n",
      "Requirement already satisfied: joblib>=1.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.4.2)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (2.2.0)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (1.2.0)\n",
      "Requirement already satisfied: cycler>=0.10 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (4.51.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (1.4.4)\n",
      "Requirement already satisfied: packaging>=20.0 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (23.2)\n",
      "Requirement already satisfied: pillow>=8 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (10.3.0)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (3.0.9)\n",
      "Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install pandas numpy scikit-learn matplotlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cec6d609-b999-42f6-9c69-ce95fcc9b470",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Generate 6 months of fake data\n",
    "np.random.seed(42)\n",
    "dates = pd.date_range(\"2023-01-01\", \"2023-06-30\")\n",
    "data = {\n",
    "    \"date\": dates,\n",
    "    \"temperature\": np.random.normal(25, 5, len(dates)),  # Average 25°C\n",
    "    \"rainfall\": np.random.poisson(2, len(dates)),        # Avg 2mm rain\n",
    "    \"weekend\": (dates.weekday >= 5).astype(int),         # 1 if weekend\n",
    "    \"holiday\": np.random.choice([0, 1], len(dates), p=[0.9, 0.1]),  # 10% holidays\n",
    "    \"sales\": np.random.poisson(100, len(dates))         # Base sales\n",
    "}\n",
    "\n",
    "# Adjust sales based on factors\n",
    "data[\"sales\"] = (data[\"sales\"] \n",
    "                 + 20 * data[\"weekend\"]                  # More sales on weekends\n",
    "                 - 15 * data[\"rainfall\"]                 # Less sales if raining\n",
    "                 + 10 * (data[\"temperature\"] > 25)      # More sales if hot\n",
    "                 - 30 * data[\"holiday\"])                # Closed on holidays\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "df.to_csv(\"coffee_shop_sales.csv\", index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "559b8bb5-f560-464e-8ac5-aa4918358b77",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Score: 0.7647860739819707\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Load data\n",
    "df = pd.read_csv(\"coffee_shop_sales.csv\")\n",
    "df[\"date\"] = pd.to_datetime(df[\"date\"])\n",
    "\n",
    "# Feature engineering\n",
    "X = df[[\"temperature\", \"rainfall\", \"weekend\", \"holiday\"]]\n",
    "y = df[\"sales\"]\n",
    "\n",
    "# Train/test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n",
    "\n",
    "# Train model\n",
    "model = RandomForestRegressor()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Evaluate\n",
    "print(\"Model Score:\", model.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "879c96ee-f998-4b36-88e6-6f9a4358a48b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Next week's predicted sales: [110.1   68.74  78.74  92.06 111.28 101.65 128.62]\n"
     ]
    }
   ],
   "source": [
    "# Forecast next week (example inputs)\n",
    "next_week = pd.DataFrame({\n",
    "    \"temperature\": [28, 26, 24, 22, 30, 29, 27],\n",
    "    \"rainfall\": [0, 3, 1, 0, 0, 2, 0],\n",
    "    \"weekend\": [0, 0, 0, 0, 0, 1, 1],\n",
    "    \"holiday\": [0, 0, 0, 0, 0, 0, 0]\n",
    "})\n",
    "\n",
    "predicted_sales = model.predict(next_week)\n",
    "print(\"Next week's predicted sales:\", predicted_sales)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "ae53ced5-2624-4df8-ba63-84c29fba1f95",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-26 21:44:44.061 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "st.title(\"☕ Coffee Sales Predictor\")\n",
    "st.write(\"Predict how much coffee you’ll sell tomorrow!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "857f35b9-532f-4507-b8f6-b7632a7bd6af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in /opt/anaconda3/lib/python3.12/site-packages (1.32.0)\n",
      "Requirement already satisfied: pandas in /opt/anaconda3/lib/python3.12/site-packages (2.2.2)\n",
      "Requirement already satisfied: scikit-learn in /opt/anaconda3/lib/python3.12/site-packages (1.4.2)\n",
      "Requirement already satisfied: altair<6,>=4.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (5.0.1)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (1.6.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (5.3.3)\n",
      "Requirement already satisfied: click<9,>=7.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: numpy<2,>=1.19.3 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (1.26.4)\n",
      "Requirement already satisfied: packaging<24,>=16.8 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (23.2)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (10.3.0)\n",
      "Collecting protobuf<5,>=3.20 (from streamlit)\n",
      "  Downloading protobuf-4.25.6-cp37-abi3-macosx_10_9_universal2.whl.metadata (541 bytes)\n",
      "Requirement already satisfied: pyarrow>=7.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (14.0.2)\n",
      "Requirement already satisfied: requests<3,>=2.27 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (2.32.2)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (13.3.5)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (8.2.2)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (4.11.0)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (3.1.37)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (0.8.0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (6.4.1)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: scipy>=1.6.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.13.1)\n",
      "Requirement already satisfied: joblib>=1.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.4.2)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (2.2.0)\n",
      "Requirement already satisfied: jinja2 in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
      "Requirement already satisfied: jsonschema>=3.0 in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
      "Requirement already satisfied: toolz in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.7)\n",
      "Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2.2.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2024.7.4)\n",
      "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/anaconda3/lib/python3.12/site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: smmap<5,>=3.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/lib/python3.12/site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: mdurl~=0.1 in /opt/anaconda3/lib/python3.12/site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Downloading protobuf-4.25.6-cp37-abi3-macosx_10_9_universal2.whl (394 kB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m394.2/394.2 kB\u001b[0m \u001b[31m7.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
      "\u001b[?25hInstalling collected packages: protobuf\n",
      "  Attempting uninstall: protobuf\n",
      "    Found existing installation: protobuf 5.29.3\n",
      "    Uninstalling protobuf-5.29.3:\n",
      "      Successfully uninstalled protobuf-5.29.3\n",
      "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
      "ortools 9.12.4544 requires protobuf<5.30,>=5.29.3, but you have protobuf 4.25.6 which is incompatible.\u001b[0m\u001b[31m\n",
      "\u001b[0mSuccessfully installed protobuf-4.25.6\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit pandas scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "b5354bb5-9cc6-4661-89bc-55a46bc18600",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "import numpy as np\n",
    "\n",
    "# --- App Config ---\n",
    "st.set_page_config(\n",
    "    page_title=\"Coffee Demand AI\",\n",
    "    page_icon=\"☕\",\n",
    "    layout=\"wide\"\n",
    ")\n",
    "\n",
    "# --- Model Loader with Error Handling ---\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    try:\n",
    "        # Generate better synthetic data\n",
    "        np.random.seed(42)\n",
    "        dates = pd.date_range(\"2023-01-01\", periods=180)\n",
    "        data = {\n",
    "            \"temperature\": np.clip(np.random.normal(25, 7, 180), 5, 40),\n",
    "            \"rainfall\": np.random.poisson(3, 180),\n",
    "            \"weekend\": (dates.weekday >= 5).astype(int),\n",
    "            \"holiday\": np.random.binomial(1, 0.05, 180),\n",
    "            \"sales\": np.clip(80 + 30*(dates.weekday >=5) - 10*np.random.poisson(3, 180), 0, 300)\n",
    "        }\n",
    "        df = pd.DataFrame(data)\n",
    "        \n",
    "        model = RandomForestRegressor(n_estimators=100)\n",
    "        model.fit(df[[\"temperature\", \"rainfall\", \"weekend\", \"holiday\"]], df[\"sales\"])\n",
    "        return model\n",
    "    except Exception as e:\n",
    "        st.error(f\"Model failed to load: {str(e)}\")\n",
    "        return None\n",
    "\n",
    "# --- UI Layout ---\n",
    "col1, col2 = st.columns([1, 2])\n",
    "\n",
    "with col1:\n",
    "    st.header(\"Input Parameters\")\n",
    "    temp = st.slider(\"Temperature (°C)\", 0, 40, 25, help=\"Higher temps usually increase sales\")\n",
    "    rain = st.select_slider(\"Rainfall\", options=[\"None\", \"Light\", \"Medium\", \"Heavy\"], \n",
    "                          help=\"Rain reduces foot traffic\")\n",
    "    is_weekend = st.toggle(\"Weekend\")\n",
    "    is_holiday = st.toggle(\"Holiday\")\n",
    "    \n",
    "    # Convert rainfall to mm\n",
    "    rain_mm = {\"None\": 0, \"Light\": 2, \"Medium\": 5, \"Heavy\": 10}[rain]\n",
    "\n",
    "with col2:\n",
    "    st.header(\"Prediction\")\n",
    "    model = load_model()\n",
    "    if model:\n",
    "        input_data = pd.DataFrame({\n",
    "            \"temperature\": [temp],\n",
    "            \"rainfall\": [rain_mm],\n",
    "            \"weekend\": [int(is_weekend)],\n",
    "            \"holiday\": [int(is_holiday)]\n",
    "        })\n",
    "        \n",
    "        prediction = model.predict(input_data)[0]\n",
    "        st.metric(\"Expected Sales\", f\"{max(0, int(prediction))} cups\", \n",
    "                delta=f\"{int((prediction-100)/100*100)}% vs average\")\n",
    "        \n",
    "        # Visual feedback\n",
    "        if prediction > 150:\n",
    "            st.success(\"🔥 Hot day ahead! Prepare extra staff and ingredients.\")\n",
    "        elif prediction < 70:\n",
    "            st.warning(\"Slow day expected. Consider reducing temporary staff.\")\n",
    "    else:\n",
    "        st.error(\"Model unavailable - please refresh the page\")\n",
    "\n",
    "# --- Business Insights Section ---\n",
    "st.divider()\n",
    "with st.expander(\"💡 Business Recommendations\"):\n",
    "    st.write(\"\"\"\n",
    "    **How to use these predictions:**\n",
    "    - 📈 **High sales forecast:** Schedule more staff, prep extra inventory\n",
    "    - 📉 **Low sales forecast:** Reduce perishable orders, plan maintenance\n",
    "    - ☔ **Rainy days:** Promote delivery/takeout options\n",
    "    \"\"\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
