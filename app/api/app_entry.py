from fastapi import FastAPI

app = FastAPI(title="Fraud Detection SaaS API")


@app.get("/sanity")
def sanity():
    return {"ok": True}


@app.get("/admin/test")
def admin_test():
    return {"admin": "alive"}