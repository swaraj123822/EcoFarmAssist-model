from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
    return {"what message do you want  to type in "}
