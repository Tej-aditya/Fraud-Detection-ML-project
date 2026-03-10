from app.db import Base, engine
import app.models  # registers tables

Base.metadata.create_all(bind=engine)
print("✅ Database initialized successfully")