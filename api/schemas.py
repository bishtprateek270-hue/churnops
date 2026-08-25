"""
Pydantic schemas for FastAPI serving layer input/output validation.
"""


from pydantic import BaseModel, Field, field_validator


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

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('gender must be either Male or Female')
        return v

    @field_validator('Partner', 'Dependents', 'PhoneService', 'PaperlessBilling')
    @classmethod
    def validate_yes_no(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('This field must be either Yes or No')
        return v

    @field_validator('MultipleLines')
    @classmethod
    def validate_multiple_lines(cls, v):
        if v not in ['Yes', 'No', 'No phone service']:
            raise ValueError('MultipleLines must be Yes, No, or No phone service')
        return v

    @field_validator('InternetService')
    @classmethod
    def validate_internet_service(cls, v):
        if v not in ['DSL', 'Fiber optic', 'No']:
            raise ValueError('InternetService must be DSL, Fiber optic, or No')
        return v

    @field_validator('OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies')
    @classmethod
    def validate_service_options(cls, v):
        if v not in ['Yes', 'No', 'No internet service']:
            raise ValueError('This field must be Yes, No, or No internet service')
        return v

    @field_validator('Contract')
    @classmethod
    def validate_contract(cls, v):
        if v not in ['Month-to-month', 'One year', 'Two year']:
            raise ValueError('Contract must be Month-to-month, One year, or Two year')
        return v

    @field_validator('PaymentMethod')
    @classmethod
    def validate_payment_method(cls, v):
        if v not in ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']:
            raise ValueError('PaymentMethod must be one of the valid payment methods')
        return v


class ChurnOutput(BaseModel):
    churn_prediction: int = Field(..., description="Binary churn label (1 = Churn, 0 = Stay)")
    churn_label: str = Field(..., description="Readable label ('Yes' or 'No')")
    churn_probability: float = Field(..., description="Probability of customer churning")
    model_version: str = Field(..., description="MLflow model version used for inference")
    timestamp: str = Field(..., description="ISO timestamp of prediction request")
    request_id: str | None = Field(None, description="Unique request identifier for tracing")
    processing_time_ms: float | None = Field(None, description="Processing time in milliseconds")


class BatchChurnInput(BaseModel):
    customers: list[ChurnInput] = Field(..., min_length=1, max_length=100, description="List of customer records to predict")


class BatchChurnResult(BaseModel):
    index: int = Field(..., description="Index of customer in input list")
    churn_prediction: int = Field(..., description="Binary churn label (1 = Churn, 0 = Stay)")
    churn_label: str = Field(..., description="Readable label ('Yes' or 'No')")
    churn_probability: float = Field(..., description="Probability of customer churning")


class BatchChurnError(BaseModel):
    index: int = Field(..., description="Index of customer that failed")
    error: str = Field(..., description="Error message")


class BatchChurnOutput(BaseModel):
    results: list[BatchChurnResult] = Field(..., description="Successful predictions")
    errors: list[BatchChurnError] = Field(default_factory=list, description="Failed predictions")
    total_processed: int = Field(..., description="Number of successfully processed customers")
    total_errors: int = Field(..., description="Number of failed predictions")
    model_version: str = Field(..., description="MLflow model version used for inference")
    timestamp: str = Field(..., description="ISO timestamp of prediction request")
    request_id: str | None = Field(None, description="Unique request identifier for tracing")
    processing_time_ms: float | None = Field(None, description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    model_name: str = Field(..., description="Registered model name")
    model_stage: str = Field(..., description="MLflow stage loaded (e.g., 'Production' or 'Staging')")
    model_version: str = Field(..., description="Version of model loaded")
    model_loaded_at: str | None = Field(None, description="ISO timestamp when model was loaded")
    preprocessor_loaded: bool = Field(..., description="Whether preprocessor is loaded")
    timestamp: str = Field(..., description="Current server ISO timestamp")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    request_id: str | None = Field(None, description="Request identifier for tracing")
    timestamp: str = Field(..., description="ISO timestamp of error")
