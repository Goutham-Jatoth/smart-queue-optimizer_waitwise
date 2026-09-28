# smart_queue_final.py
import streamlit as st
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Smart Queue Optimizer",
    page_icon="⏱️",
    layout="wide"
)

# -------------------------------
# College Logo / Header
# -------------------------------
st.image("https://upload.wikimedia.org/wikipedia/commons/0/0a/ACE_Logo.png", width=200)  # Replace with your college logo
st.title("🟢 Smart Queue Optimizer")
st.write("Predict waiting time & manage queue in real-time for hospitals, banks, and offices.")

# -------------------------------
# Generate Sample Queue Dataset
# -------------------------------
num_entries = 50  # Total people in queue
data = {
    "token_number": list(range(1, num_entries + 1)),
    "service_time": [random.randint(3, 10) for _ in range(num_entries)]  # minutes per person
}
df = pd.DataFrame(data)

# -------------------------------
# Train ML Model (Optional)
# -------------------------------
# Using token_number as "customers ahead", counters=1, service_time as features
X = df[["token_number", "service_time"]]
y = df["service_time"].cumsum()  # Approx total waiting time until this token
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# Sidebar - User Input
# -------------------------------
st.sidebar.header("Queue Info / Your Token")
your_token = st.sidebar.number_input("Enter Your Token Number", min_value=1, max_value=num_entries, value=1)
counters = st.sidebar.slider("Number of Counters", 1, 5, 1)

# -------------------------------
# CSV Download Preparation
# -------------------------------
csv_data = df.copy()
csv_data["Approx_wait_time"] = df["service_time"].cumsum()
st.sidebar.download_button(
    label="Download Token Dataset CSV",
    data=csv_data.to_csv(index=False).encode('utf-8'),
    file_name='queue_token_dataset.csv',
    mime='text/csv'
)

# -------------------------------
# Function: Calculate Waiting Time for a Token
# -------------------------------
def calculate_wait(token_num, counters, df):
    remaining_tokens = df[df["token_number"] >= token_num].copy().reset_index(drop=True)
    total_wait = 0
    approx_waits = []
    for idx, row in remaining_tokens.iterrows():
        wait = row["service_time"] / counters
        total_wait += wait
        approx_waits.append(round(total_wait, 2))
    return approx_waits

# -------------------------------
# Real-Time Queue Simulation
# -------------------------------
st.subheader("⏳ Queue Simulation")
simulate_button = st.button("Start Queue Simulation")

queue_placeholder = st.empty()

if simulate_button:
    for current_token in range(1, num_entries + 1):
        with queue_placeholder.container():
            st.write(f"🎫 **Current Token Being Served:** {current_token}")
            tokens_left = num_entries - current_token
            st.write(f"📌 Tokens remaining in queue: {tokens_left}")
            
            if your_token >= current_token:
                approx_waits = calculate_wait(your_token, counters, df)
                predicted_wait = approx_waits[0]

                # Color-coded waiting time
                if predicted_wait < 30:
                    st.success(f"⏱ Your token: {your_token} | Approx waiting time: {predicted_wait} minutes (✅ Optimal)")
                    st.balloons()  # 🎈 Fun effect
                elif predicted_wait <= 60:
                    st.warning(f"⏱ Your token: {your_token} | Approx waiting time: {predicted_wait} minutes (⚠️ Moderate)")
                else:
                    st.error(f"⏱ Your token: {your_token} | Approx waiting time: {predicted_wait} minutes (❌ High!)")
            else:
                st.success(f"✅ Your token {your_token} has been served!")

            # Upcoming tokens preview
            st.write("Next tokens in queue:", list(range(current_token+1, min(current_token+6, num_entries+1))))

            # Bar chart: current token vs remaining
            fig, ax = plt.subplots()
            ax.bar(["Current Token", "Remaining Tokens"], [1, tokens_left], color=["green", "orange"])
            ax.set_ylabel("Tokens")
            ax.set_title("Queue Progress")
            st.pyplot(fig)

            # Pause to simulate real service time
            time.sleep(df.loc[current_token-1, "service_time"] / counters)

# -------------------------------
# Queue Dataset Visualization
# -------------------------------
st.subheader("📈 Queue Dataset Scatter Plot")
fig2, ax2 = plt.subplots()
scatter = ax2.scatter(df["token_number"], df["service_time"].cumsum(), 
                      c=df["service_time"].cumsum(), cmap="RdYlGn_r", s=80)
ax2.set_xlabel("Token Number")
ax2.set_ylabel("Approx Waiting Time (minutes)")
ax2.set_title("Queue Waiting Time Analysis")
plt.colorbar(scatter, label="Waiting Time")
st.pyplot(fig2)

# -------------------------------
# Dataset Preview
# -------------------------------
st.subheader("📋 Sample Token Dataset")
st.dataframe(df.head(15))
