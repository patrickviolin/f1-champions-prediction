from fastapi import FastAPI

from api.routers import predict_router

app = FastAPI(
    title='F1 Race Predictor API',
    description='Machine Learning API for F1 Race Results Predictor',
    version='1.0.0'
)

app.include_router(predict_router.router, prefix='/api/v1')


@app.get('/health', tags=['health'])
def health_check():
    return {'status': 'UP', 'message': 'F1 Predictor API is running.'}
