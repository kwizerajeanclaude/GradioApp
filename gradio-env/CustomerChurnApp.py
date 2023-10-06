import gradio as gr

# Define your function that processes the input
def process_input(
    CustomerID, Gender, SeniorCitizens, Partner, Tenure, PhoneService, Multiplelines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges
):
    # Perform some processing based on the inputs (you can define your logic here)
    result = f"CustomerID: {CustomerID}, Gender: {Gender}, Tenure: {Tenure}"
    return result

# Define input components using gr.components
input_components = [
    gr.components.Textbox(label="CustomerID"),
    gr.components.Radio(label="Gender", choices=["Male", "Female"]),
    gr.components.Radio(label="SeniorCitizens", choices=["Yes", "No"]),
    gr.components.Radio(label="Partner", choices=["Yes", "No"]),
    gr.components.Number(label="Tenure"),
    gr.components.Radio(label="PhoneService", choices=["Yes", "No"]),
    gr.components.Dropdown(
        label="MultipleLines",
        choices=["Unknown", "No", "Yes"]
    ),
    gr.components.Dropdown(label="InternetService", choices=["DSL", "Fiber optic", "No"]),
    gr.components.Dropdown(label="OnlineSecurity", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="OnlineBackup", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="DeviceProtection", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="TechSupport", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="StreamingTV", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="StreamingMovies", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="Contract", choices=["Month-to-month", "One year", "Two year"]),
    gr.components.Radio(label="PaperlessBilling", choices=["Yes", "No"]),
    gr.components.Dropdown(label="PaymentMethod", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
    gr.components.Number(label="MonthlyCharges"),
    gr.components.Number(label="TotalCharges")
]

# Create the Gradio interface
iface = gr.Interface(
    fn=process_input,
    inputs=input_components,
    outputs="text"  # You can specify the output type here.
)

# Launch the interface
iface.launch()
