from sqlalchemy import Column, Integer, Float, Boolean, String
from .database import Base

class TransactionDB(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float)
    is_international = Column(Boolean)
    transaction_type = Column(String)
    device_type = Column(String)

    risk_score = Column(Integer)
    ml_probability = Column(Float)
    fraud_detected = Column(Boolean)
