from pydantic import BaseModel


class ErrorDetail(BaseModel):
    error_code: str
    message: str
    request_id: str
    context: dict = {}


class ErrorResponse(BaseModel):
    error: ErrorDetail
