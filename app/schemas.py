from pydantic import BaseModel


class Customer(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    poutcome: str
    campaign: int
    previous: int
    emp_var_rate: float
    cons_price_idx: float
    euribor3m: float
    nr_employed: float
