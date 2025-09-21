# optimized_dataset_generator.py
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
    "comments": ["id", "post_id", "user_id", "comment", "created_at"],
    "movies": ["id", "title", "release_date", "rating", "duration"],
    "cast": ["id", "movie_id", "actor_id", "character_name"],
    "actors": ["id", "name", "birthdate", "nationality"]
}
table_names = list(tables.keys())

# --- Helpers ---
def sample_columns(table, minc=1, maxc=4):
    cols = tables[table]
    return random.sample(cols, k=random.randint(minc, min(maxc, len(cols))))

def format_cols(cols):
    return ", ".join(cols)

def random_date(start_year=2018, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")

def make_where_clause(table):
    col = random.choice(tables[table])
    if "date" in col or "created_at" in col or "join" in col:
        op = random.choice([">", "<", ">=", "<="])
        d = random_date()
        return f"WHERE {col} {op} '{d}'", f"{col} {op} {d}"
    if col in ("country", "status", "payment_method", "carrier", "name"):
        val = random.choice(["'US'", "'UK'", "'IN'", "'pending'", "'completed'", "'credit_card'", "'UPS'"])
        return f"WHERE {col} = {val}", f"{col} = {val}"
    if col in ("age", "price", "stock", "salary", "quantity", "rating", "views", "total_amount", "amount", "duration", "budget"):
        num = random.randint(1, 500)
        op = random.choice([">", "<", ">=", "<=", "="])
        return f"WHERE {col} {op} {num}", f"{col} {op} {num}"
    return f"WHERE {col} IS NOT NULL", f"{col} IS NOT NULL"

def make_join(table_a, table_b):
    # Try FK candidates
    candidates = [(a_col, b_col) for a_col in tables[table_a] for b_col in tables[table_b] if a_col.endswith("_id") and a_col[:-3] in table_b]
    if candidates:
        a_col, b_col = random.choice(candidates)
        return f"JOIN {table_b} ON {table_a}.{a_col} = {table_b}.id", f"{table_a}.{a_col} = {table_b}.id"
    # fallback
    return f"JOIN {table_b} ON {table_a}.id = {table_b}.{table_a}_id", f"{table_a}.id = {table_b}.{table_a}_id"

def make_aggregate_clause(table):
    agg = random.choice(["COUNT", "SUM", "AVG", "MAX", "MIN"])
    col = random.choice([c for c in tables[table] if c not in ("id","name","title")] or ["id"])
    return f"{agg}({col}) AS {agg.lower()}_{col}", agg, col

# --- Generate Examples ---
def generate_example():
    patterns = [
        "simple_select", "select_where", "select_order_limit", "join_select", 
        "aggregate_group", "aggregate_having", "multi_join", "subquery_in", 
        "between_dates", "count_distinct", "select_specific_columns"
    ]
    weights = [0.08, 0.20, 0.12, 0.18, 0.10, 0.05, 0.07, 0.07, 0.06, 0.03, 0.04]
    pat = random.choices(patterns, weights)[0]

    t = random.choice(table_names)
    if pat == "simple_select":
        cols = sample_columns(t)
        prompt = f"List {format_cols(cols)} from {t}"
        sql = f"SELECT {format_cols(cols)} FROM {t};"
    elif pat == "select_where":
        cols = sample_columns(t)
        where_clause, where_desc = make_where_clause(t)
        prompt = f"Get {format_cols(cols)} from {t} where {where_desc}"
        sql = f"SELECT {format_cols(cols)} FROM {t} {where_clause};"
    elif pat == "select_order_limit":
        cols = sample_columns(t)
        order_col = random.choice(tables[t])
        direction = random.choice(["ASC", "DESC"])
        limit = random.choice([5,10,20,50])
        prompt = f"Get {format_cols(cols)} from {t} ordered by {order_col} {direction} limit {limit}"
        sql = f"SELECT {format_cols(cols)} FROM {t} ORDER BY {order_col} {direction} LIMIT {limit};"
    elif pat == "join_select":
        a, b = random.sample(table_names, 2)
        cols_a = sample_columns(a)
        cols_b = sample_columns(b)
        join_clause, join_desc = make_join(a,b)
        prompt = f"Join {a} and {b} to get {format_cols(cols_a + cols_b)} where {join_desc}"
        sql = f"SELECT {format_cols([f'{a}.{c}' for c in cols_a] + [f'{b}.{c}' for c in cols_b])} FROM {a} {join_clause};"
    elif pat == "aggregate_group":
        agg_expr, agg_name, agg_col = make_aggregate_clause(t)
        group_col = random.choice([c for c in tables[t] if c not in (agg_col,"id")] or ["id"])
        prompt = f"Get {agg_name} of {agg_col} grouped by {group_col} from {t}"
        sql = f"SELECT {group_col}, {agg_expr} FROM {t} GROUP BY {group_col};"
    elif pat == "aggregate_having":
        agg_expr, agg_name, agg_col = make_aggregate_clause(t)
        group_col = random.choice([c for c in tables[t] if c not in (agg_col,"id")] or ["id"])
        comp = random.randint(1,50)
        prompt = f"Get {group_col} having {agg_name}({agg_col}) > {comp} from {t}"
        sql = f"SELECT {group_col}, {agg_expr} FROM {t} GROUP BY {group_col} HAVING {agg_name}({agg_col}) > {comp};"
    elif pat == "subquery_in":
        other = random.choice([x for x in table_names if x != t])
        col = random.choice(tables[t])
        subcol = random.choice(tables[other])
        prompt = f"Select records from {t} where {col} is in {other} {subcol}"
        sql = f"SELECT * FROM {t} WHERE {col} IN (SELECT {subcol} FROM {other});"
    elif pat == "between_dates":
        date_cols = [c for c in tables[t] if "date" in c or "created_at" in c or "hire_date" in c]
        date_col = random.choice(date_cols or tables[t])
        d1, d2 = sorted([random_date(2019,2023), random_date(2024,2025)])
        prompt = f"Get records from {t} where {date_col} between {d1} and {d2}"
        sql = f"SELECT * FROM {t} WHERE {date_col} BETWEEN '{d1}' AND '{d2}';"
    elif pat == "count_distinct":
        col = random.choice([c for c in tables[t] if c != "id"])
        prompt = f"Count distinct {col} in {t}"
        sql = f"SELECT COUNT(DISTINCT {col}) FROM {t};"
    else:
        cols = sample_columns(t)
        prompt = f"Get {format_cols(cols)} from {t}"
        sql = f"SELECT {format_cols(cols)} FROM {t};"

    return {"prompt": prompt, "response": sql}

# --- Generate Dataset ---
N = 10000
out_path = "data/dataset.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for _ in tqdm(range(N), desc="Generating dataset"):
        f.write(json.dumps(generate_example(), ensure_ascii=False) + "\n")

print(f"Dataset generated: {out_path} with {N} examples")
