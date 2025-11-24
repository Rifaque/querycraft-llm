import json
import random
from datetime import datetime, timedelta
from tqdm import tqdm

random.seed(42)

# --- Tables & columns ---
tables = {
    "users": ["id", "name", "email", "registration_date", "country", "age", "status"],
    "products": ["id", "name", "category_id", "price", "stock", "created_at", "rating"],
    "orders": ["id", "user_id", "order_date", "status", "total_amount", "payment_method"],
    "order_items": ["id", "order_id", "product_id", "quantity", "unit_price"],
    "employees": ["id", "first_name", "last_name", "department_id", "hire_date", "salary"],
    "departments": ["id", "name", "manager_id", "budget"],
    "customers": ["id", "name", "email", "joined_date", "country", "phone"],
    "reviews": ["id", "product_id", "user_id", "rating", "comment", "created_at"],
    "payments": ["id", "order_id", "amount", "payment_date", "status"],
    "shipments": ["id", "order_id", "shipped_date", "delivery_date", "carrier", "status"],
    "categories": ["id", "name", "parent_id"],
    "suppliers": ["id", "name", "contact_email", "country", "phone"],
    "transactions": ["id", "user_id", "amount", "transaction_date", "status"],
    "posts": ["id", "user_id", "title", "body", "created_at", "views"],
    "comments": ["id", "post_id", "user_id", "comment", "created_at"]
}
table_names = list(tables.keys())

def random_date(start_year=2018, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")

def format_cols(cols):
    return ", ".join(cols)

# ----------------------------
# Schema-grounded example
# ----------------------------
def generate_example():
    t = random.choice(table_names)          # fixed table
    cols = tables[t]
    schema = f"{t}({', '.join(cols)})"

    # pick 1–4 columns for SELECT
    selected = random.sample(cols, k=random.randint(1, min(4, len(cols))))

    r = random.random()

    # --------------------------------------
    # SIMPLE SELECT (40%)
    # --------------------------------------
    if r < 0.40:
        question = f"Get {format_cols(selected)} from {t}"
        sql = f"SELECT {format_cols(selected)} FROM {t};"

    # --------------------------------------
    # WHERE clause (30%)
    # --------------------------------------
    elif r < 0.70:
        col = random.choice(cols)
        if "date" in col:
            val = random_date()
            question = f"Get {format_cols(selected)} from {t} where {col} > '{val}'"
            sql = f"SELECT {format_cols(selected)} FROM {t} WHERE {col} > '{val}';"
        else:
            question = f"Get {format_cols(selected)} from {t} where {col} IS NOT NULL"
            sql = f"SELECT {format_cols(selected)} FROM {t} WHERE {col} IS NOT NULL;"

    # --------------------------------------
    # GROUP BY MAX (30%)
    # --------------------------------------
    else:
        usable = [c for c in cols if c not in ("id", selected[0])]
        if not usable:
            usable = cols
        group_col = random.choice(usable)

        question = f"Get MAX of {selected[0]} grouped by {group_col} from {t}"
        sql = (
            f"SELECT {group_col}, MAX({selected[0]}) AS max_{selected[0]} "
            f"FROM {t} GROUP BY {group_col};"
        )

    prompt = f"Schema: {schema}\nQuestion: {question}"
    return {"prompt": prompt, "response": sql}


# ----------- Write dataset -----------
N = 10000
with open("data/dataset.jsonl", "w") as f:
    for _ in tqdm(range(N), desc="Building SQL dataset"):
        f.write(json.dumps(generate_example(), ensure_ascii=False) + "\n")

print("Dataset rebuilt with schema-grounded SQL.")
