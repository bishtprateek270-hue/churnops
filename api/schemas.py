"""
Pydantic schemas for FastAPI serving layer input/output validation.
"""

from pydantic import BaseModel, Field


class ChurnInput(BaseModel):
    gender: str = Field(..., json_schema_extra={"example": "Female"}, description="Gender of customer ('Male', 'Female')")
    SeniorCitizen: int = Field(..., ge=0, le=1, json_schema_extra={"example": 0}, description="Whether customer is a senior citizen (0 or 1)")
    Partner: str = Field(..., json_schema_extra={"example": "Yes"}, description="Whether customer has a partner ('Yes', 'No')")
    Dependents: str = Field(..., json_schema_extra={"example": "No"}, description="Whether customer has dependents ('Yes', 'No')")
    tenure: int = Field(..., ge=0, le=120, json_schema_extra={"example": 12}, description="Number of months customer has stayed with company")
    PhoneService: str = Field(..., json_schema_extra={"example": "Yes"}, description="Whether customer has phone service ('Yes', 'No')")
    MultipleLines: str = Field(..., json_schema_extra={"example": "No"}, description="Whether customer has multiple lines ('Yes', 'No', 'No phone service')")
    InternetService: str = Field(..., json_schema_extra={"example": "DSL"}, description="Customer's internet service provider ('DSL', 'Fiber optic', 'No')")
    OnlineSecurity: str = Field(..., json_schema_extra={"example": "No"}, description="Whether customer has online security ('Yes', 'No', 'No internet service')")
    OnlineBackup: str = Field(..., json_schema_extra={"example": "Yes"}, description="Whether customer has online backup ('Yes', 'No', 'No internet service')")
    DeviceProtection: str = Field(..., json_schema_extra={"example": "No"}, description="Whether customer has device protection ('Yes', 'No', 'No internet service')")
    TechSupport: str = Field(..., json_schema_extra={"example": "No"}, description="Whether customer has tech support ('Yes', 'No', 'No internet service')")
    StreamingTV: str = Field(..., json_schema_extra={"example": "No"}, description="Whether customer has streaming TV ('Yes', 'No', 'No internet service')")
    StreamingMovies: str = Field(..., json_schema_extra={"example": "No"}, description="Whether customer has streaming movies ('Yes', 'No', 'No internet service')")
    Contract: str = Field(..., json_schema_extra={"example": "Month-to-month"}, description="Contract term ('Month-to-month', 'One year', 'Two year')")
    PaperlessBilling: str = Field(..., json_schema_extra={"example": "Yes"}, description="Whether customer has paperless billing ('Yes', 'No')")
    PaymentMethod: str = Field(..., json_schema_extra={"example": "Electronic check"}, description="Payment method ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')")
    MonthlyCharges: float = Field(..., ge=0.0, le=300.0, json_schema_extra={"example": 65.50}, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0.0, json_schema_extra={"example": 786.00}, description="Total amount charged to customer")


class ChurnOutput(BaseModel):
    churn_prediction: int = Field(..., description="Binary churn label (1 = Churn, 0 = Stay)")
    churn_label: str = Field(..., description="Readable label ('Yes' or 'No')")
    churn_probability: float = Field(..., description="Probability of customer churning")
    model_version: str = Field(..., description="MLflow model version used for inference")
    timestamp: str = Field(..., description="ISO timestamp of prediction request")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_name: str = Field(..., description="Registered model name")
    model_stage: str = Field(..., description="MLflow stage loaded (e.g., 'Production' or 'Staging')")
    model_version: str = Field(..., description="Version of model loaded")
    timestamp: str = Field(..., description="Current server ISO timestamp")
