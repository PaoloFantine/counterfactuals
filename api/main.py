from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.classification import router as classification_router
from api.regression import router as regression_router

app = FastAPI()
app.include_router(classification_router)
app.include_router(regression_router)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    # Map ValueError to HTTP 422 Unprocessable Entity
    return JSONResponse(
        status_code=422, content={"error": str(exc), "type": "ValueError"}
    )


@app.get("/")
async def root():
    return {
        "message": "API showing example of counterfactuals for classification and regression tasks"
    }
