from app.db import SessionLocal
from app.models import Organization, APIKey
import secrets

def seed_initial_data():
    db = SessionLocal()

    # Check if org already exists
    org = db.query(Organization).first()
    if org:
        db.close()
        print("ℹ️ Initial data already exists")
        return

    # Create default org
    org = Organization(name="Default Fintech Org")
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

    print("✅ Seeded default org and API key")
    print("🔑 API KEY:", api_key.key)

    db.close()