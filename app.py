import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq

# Page config
st.set_page_config(
    page_title="AI Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/churn_model.pkl')
    features = joblib.load('models/feature_names.pkl')
    return model, features

model, feature_names = load_model()

# AI explanation - Single Customer
def get_ai_explanation(customer_data, churn_prob, top_factors):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        factors_text = "\n".join([f"- {f[0]}: {f[1]:.3f} importance" for f in top_factors])
        prompt = f"""You are a customer retention specialist. A customer has a {churn_prob:.1%} churn probability.

Top risk factors:
{factors_text}

Customer profile:
- Tenure: {customer_data.get('tenure', 'N/A')} months
- Monthly Charges: £{customer_data.get('MonthlyCharges', 'N/A')}
- Contract: {customer_data.get('Contract', 'N/A')}

Write a concise business analysis (3 paragraphs):
1. Why this customer is at risk
2. Key warning signals
3. Specific retention recommendations

Keep it professional and actionable. Max 200 words."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

# AI Executive Summary - Batch Mode
def get_batch_ai_summary(total, high, medium, low, avg_prob, top_factors):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        factors_text = "\n".join([f"- {f[0]}: {f[1]:.3f} importance" for f in top_factors])
        prompt = f"""You are a senior customer retention analyst presenting to the board.

Batch analysis results:
- Total customers analysed: {total}
- High risk (>70% churn probability): {high} ({high/total:.1%})
- Medium risk (40-70%): {medium} ({medium/total:.1%})
- Low risk (<40%): {low} ({low/total:.1%})
- Average churn probability: {avg_prob:.1%}

Top churn drivers across the customer base:
{factors_text}

Write a concise executive summary (3 paragraphs) for the board:
1. Overall churn risk situation and business impact
2. Key patterns and root causes driving churn
3. Strategic retention recommendations with priority actions

Keep it professional, data-driven and actionable. Max 250 words."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI summary unavailable: {str(e)}"

# Preprocess uploaded CSV
def preprocess_uploaded(df):
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

# Header
st.title("🔮 AI Customer Churn Predictor")
st.markdown("*Predict customer churn risk and get AI-powered retention recommendations*")
st.markdown("---")

# Mode selector
mode = st.radio(
    "Choose prediction mode:",
    ["👤 Single Customer", "📂 Batch Upload (CSV)"],
    horizontal=True
)

st.markdown("---")

# ─────────────────────────────────────────
# MODE 1: SINGLE CUSTOMER
# ─────────────────────────────────────────
if mode == "👤 Single Customer":

    st.sidebar.header("👤 Customer Profile")

    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges (£)", 18, 120, 65)
    total_charges = monthly_charges * tenure

    contract = st.sidebar.selectbox("Contract Type",
        ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Internet Service",
        ["DSL", "Fiber optic", "No"])
    payment_method = st.sidebar.selectbox("Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    senior_citizen = st.sidebar.checkbox("Senior Citizen")
    partner = st.sidebar.checkbox("Has Partner")
    dependents = st.sidebar.checkbox("Has Dependents")
    phone_service = st.sidebar.checkbox("Phone Service", value=True)
    multiple_lines = st.sidebar.checkbox("Multiple Lines")
    online_security = st.sidebar.checkbox("Online Security")
    online_backup = st.sidebar.checkbox("Online Backup")
    device_protection = st.sidebar.checkbox("Device Protection")
    tech_support = st.sidebar.checkbox("Tech Support")
    streaming_tv = st.sidebar.checkbox("Streaming TV")
    streaming_movies = st.sidebar.checkbox("Streaming Movies")
    paperless_billing = st.sidebar.checkbox("Paperless Billing", value=True)

    def encode_inputs():
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
        payment_map = {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3
        }
        return {
            'SeniorCitizen': int(senior_citizen),
            'Partner': int(partner),
            'Dependents': int(dependents),
            'tenure': tenure,
            'PhoneService': int(phone_service),
            'MultipleLines': int(multiple_lines),
            'InternetService': internet_map[internet_service],
            'OnlineSecurity': int(online_security),
            'OnlineBackup': int(online_backup),
            'DeviceProtection': int(device_protection),
            'TechSupport': int(tech_support),
            'StreamingTV': int(streaming_tv),
            'StreamingMovies': int(streaming_movies),
            'Contract': contract_map[contract],
            'PaperlessBilling': int(paperless_billing),
            'PaymentMethod': payment_map[payment_method],
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'gender': 0
        }

    if st.sidebar.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):

        customer_data = encode_inputs()
        input_df = pd.DataFrame([customer_data])[feature_names]

        churn_prob = model.predict_proba(input_df)[0][1]

        if churn_prob >= 0.7:
            risk_level = "🔴 HIGH RISK"
            risk_color = "#FF4444"
            recommendation = "Immediate intervention required"
        elif churn_prob >= 0.4:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "#FFA500"
            recommendation = "Proactive retention recommended"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "#00CC00"
            recommendation = "Monitor and maintain satisfaction"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Churn Probability", f"{churn_prob:.1%}")
        with col2:
            st.metric("Risk Level", risk_level)
        with col3:
            st.metric("Tenure", f"{tenure} months")
        with col4:
            st.metric("Monthly Charges", f"£{monthly_charges}")

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=churn_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk Score", 'font': {'size': 20}},
                number={'suffix': "%", 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 40], 'color': "#E8F5E9"},
                        {'range': [40, 70], 'color': "#FFF9C4"},
                        {'range': [70, 100], 'color': "#FFEBEE"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_right:
            importances = model.feature_importances_
            feat_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(8)

            fig_bar = px.bar(
                feat_imp, x='Importance', y='Feature',
                orientation='h', title='Top Risk Factors',
                color='Importance', color_continuous_scale='Reds'
            )
            fig_bar.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 AI Retention Analysis")

        top_factors = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:5]

        with st.spinner("Generating AI analysis..."):
            ai_analysis = get_ai_explanation(
                {'tenure': tenure, 'MonthlyCharges': monthly_charges, 'Contract': contract},
                churn_prob, top_factors
            )
        st.info(ai_analysis)

        st.markdown("---")
        st.subheader("📋 Recommended Action")
        if churn_prob >= 0.7:
            st.error(f"⚠️ **{recommendation}** — Contact this customer within 24 hours with a personalised retention offer.")
        elif churn_prob >= 0.4:
            st.warning(f"📞 **{recommendation}** — Schedule a check-in call and review their current plan.")
        else:
            st.success(f"✅ **{recommendation}** — Continue delivering great service to maintain loyalty.")

    else:
        st.info("👈 Configure customer profile in the sidebar and click **Predict Churn Risk** to get started.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🔮 ML Prediction")
            st.markdown("Random Forest model trained on 7,043 telecom customers with 76.4% accuracy")
        with col2:
            st.markdown("### 🤖 AI Analysis")
            st.markdown("Groq/Llama 3.3 explains WHY a customer is at risk in plain business English")
        with col3:
            st.markdown("### 📋 Action Plan")
            st.markdown("Colour-coded risk levels with specific retention recommendations")

# ─────────────────────────────────────────
# MODE 2: BATCH CSV UPLOAD
# ─────────────────────────────────────────
else:
    st.subheader("📂 Batch Churn Prediction")
    st.markdown("Upload a CSV file with customer data to predict churn risk for multiple customers at once.")

    # Download sample template
    sample_df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv').head(5)
    st.download_button(
        label="⬇️ Download Sample CSV Template",
        data=sample_df.to_csv(index=False),
        file_name="sample_customers.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload your customer CSV file",
        type=['csv'],
        help="CSV should have the same columns as the Telco dataset"
    )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded — {len(raw_df)} customers loaded")

        with st.expander("📋 Preview Raw Data"):
            st.dataframe(raw_df.head(10))

        try:
            processed_df = preprocess_uploaded(raw_df)
            input_df = processed_df[feature_names]

            probs = model.predict_proba(input_df)[:, 1]
            preds = model.predict(input_df)

            results_df = raw_df.copy()
            if 'customerID' in results_df.columns:
                id_col = results_df['customerID']
            else:
                id_col = pd.Series(range(len(results_df)), name='CustomerID')

            results_df['Churn_Probability'] = (probs * 100).round(1)
            results_df['Churn_Prediction'] = ['Will Churn' if p == 1 else 'Will Stay' for p in preds]
            results_df['Risk_Level'] = results_df['Churn_Probability'].apply(
                lambda x: '🔴 HIGH' if x >= 70 else ('🟡 MEDIUM' if x >= 40 else '🟢 LOW')
            )

            st.markdown("---")

            # Summary metrics
            high = (probs >= 0.7).sum()
            medium = ((probs >= 0.4) & (probs < 0.7)).sum()
            low = (probs < 0.4).sum()
            avg_prob = probs.mean()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", len(raw_df))
            with col2:
                st.metric("🔴 High Risk", high)
            with col3:
                st.metric("🟡 Medium Risk", medium)
            with col4:
                st.metric("🟢 Low Risk", low)

            st.markdown("---")

            # Charts
            col_left, col_right = st.columns(2)

            with col_left:
                risk_counts = pd.DataFrame({
                    'Risk Level': ['High Risk', 'Medium Risk', 'Low Risk'],
                    'Count': [high, medium, low]
                })
                fig_pie = px.pie(
                    risk_counts, values='Count', names='Risk Level',
                    title='Customer Risk Distribution',
                    color_discrete_map={
                        'High Risk': '#FF4444',
                        'Medium Risk': '#FFA500',
                        'Low Risk': '#00CC00'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_right:
                fig_hist = px.histogram(
                    x=probs * 100,
                    nbins=20,
                    title='Churn Probability Distribution',
                    labels={'x': 'Churn Probability (%)'},
                    color_discrete_sequence=['#FF6B6B']
                )
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")

            # Results table
            st.subheader("📊 Customer Results Table")
            display_cols = ['Churn_Probability', 'Churn_Prediction', 'Risk_Level']
            if 'customerID' in results_df.columns:
                display_cols = ['customerID'] + display_cols
            if 'tenure' in results_df.columns:
                display_cols.append('tenure')
            if 'MonthlyCharges' in results_df.columns:
                display_cols.append('MonthlyCharges')
            if 'Contract' in results_df.columns:
                display_cols.append('Contract')

            st.dataframe(
                results_df[display_cols].sort_values('Churn_Probability', ascending=False),
                use_container_width=True
            )

            # Download results
            st.download_button(
                label="⬇️ Download Results CSV",
                data=results_df.to_csv(index=False),
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

            # ─── AI EXECUTIVE SUMMARY ───
            st.markdown("---")
            st.subheader("🤖 AI Executive Summary")
            st.markdown("*Generate a board-ready AI analysis of your entire customer churn landscape*")

            top_factors = sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:5]

            if st.button("🤖 Generate AI Executive Summary", type="primary"):
                with st.spinner("AI is analysing your customer base and writing executive summary..."):
                    summary = get_batch_ai_summary(
                        total=len(raw_df),
                        high=int(high),
                        medium=int(medium),
                        low=int(low),
                        avg_prob=float(avg_prob),
                        top_factors=top_factors
                    )
                st.info(summary)

                # Download summary as text
                st.download_button(
                    label="⬇️ Download AI Executive Summary",
                    data=summary,
                    file_name="ai_executive_summary.txt",
                    mime="text/plain"
                )

            # ─── HIGH RISK CUSTOMERS ───
            st.markdown("---")
            st.subheader("🔴 High Risk Customers — Immediate Action Required")
            high_risk = results_df[probs >= 0.7]
            if len(high_risk) > 0:
                st.warning(f"⚠️ {len(high_risk)} customers need immediate retention intervention!")
                show_cols = [c for c in display_cols if c in high_risk.columns]
                st.dataframe(high_risk[show_cols], use_container_width=True)
            else:
                st.success("✅ No high risk customers found in this batch!")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Make sure your CSV has the same column structure as the Telco dataset. Download the sample template above.")

    else:
        st.info("👆 Upload a CSV file to get started, or download the sample template first.")