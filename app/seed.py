from app.db import SessionLocal
from app.models import Organization, APIKey
import secrets

def seed_initial_data():
    db = SessionLocal()

    # If org already exists, do nothing
    existing_org = db.query(Organization).first()
    if existing_org:
        print("[INFO] Seed data already exists")
        db.close()
        return

    # Create org
    org = Organization(name="Demo Fintech")
    db.add(org)
    db.commit()
    db.refresh(org)

    # Create API key
    api_key = APIKey(
        key="sk_" + secrets.token_hex(16),
        org_id=org.id,
        threshold=0.5,
        active=True
    )
    db.add(api_key)
    db.commit()

    print("[INFO] Seeding initial data...")
    print(f"[OK] Created demo org and API key: {api_key.key}")

    db.close()