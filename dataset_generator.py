# Generating a 10,000-line JSONL dataset of SQL training examples.
# Each line is a JSON object with "prompt" and "response".
# The dataset aims for variety (SELECT, WHERE, JOIN, GROUP BY, ORDER BY, LIMIT).
# The SQL should match the prompt.
# The file will be saved to /mnt/data/sql_dataset_10000.jsonl
# After this runs, a download link will be provided in the assistant message.

import json
import random
from datetime import datetime, timedelta

random.seed(42)

# Define realistic tables and columns
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

# Helpers to build queries and matching prompts
def sample_columns(table, minc=1, maxc=None):
    cols = tables[table]
    if maxc is None: maxc = min(len(cols), 4)
    k = random.randint(minc, maxc)
    return random.sample(cols, k)

def format_cols(cols):
    return ", ".join(cols)

def random_date(start_year=2018, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    d = start + timedelta(days=random.randint(0, delta.days))
    return d.strftime("%Y-%m-%d")

def make_where_clause(table):
    col = random.choice(tables[table])
    if "date" in col or "created_at" in col or "join" in col or "order_date" in col or "release_date" in col or "transaction_date" in col or "hire_date" in col:
        d = random_date()
        op = random.choice([">", "<", ">=", "<="])
        return f"WHERE {col} {op} '{d}'", f"{col} {op} {d}"
    if col in ("country", "status", "payment_method", "carrier", "name"):
        val = random.choice(["'US'", "'UK'", "'IN'", "'pending'", "'completed'", "'credit_card'", "'UPS'"])
        return f"WHERE {col} = {val}", f"{col} = {val}"
    if col in ("age", "price", "stock", "salary", "quantity", "rating", "views", "total_amount", "amount", "duration", "budget"):
        num = random.randint(1, 500)
        op = random.choice([">", "<", ">=", "<=", "="])
        return f"WHERE {col} {op} {num}", f"{col} {op} {num}"
    # fallback
    return f"WHERE {col} IS NOT NULL", f"{col} IS NOT NULL"

def make_join(table_a, table_b):
    # attempt plausible FK naming
    # common pattern: table_b has table_a_id or vice versa
    fk_candidates = []
    for a_col in tables[table_a]:
        if a_col.endswith("_id") and a_col[:-3] in table_b:
            fk_candidates.append((table_a, a_col, table_b, "id"))
    for b_col in tables[table_b]:
        if b_col.endswith("_id") and b_col[:-3] in table_a:
            fk_candidates.append((table_b, b_col, table_a, "id"))
    # generic heuristics
    if not fk_candidates:
        # use common names: user_id, product_id, order_id, department_id, category_id
        common = ["user_id","product_id","order_id","department_id","category_id","employee_id","customer_id"]
        for c in common:
            if c in tables[table_a] and "id" in tables[table_b]:
                fk_candidates.append((table_a,c,table_b,"id"))
            if c in tables[table_b] and "id" in tables[table_a]:
                fk_candidates.append((table_b,c,table_a,"id"))
    if not fk_candidates:
        # fallback join on id columns
        return f"JOIN {table_b} ON {table_a}.id = {table_b}.{table_a}_id", f"{table_a}.id = {table_b}.{table_a}_id"
    t_from, col_from, t_to, col_to = random.choice(fk_candidates)
    return f"JOIN {t_to} ON {t_from}.{col_from} = {t_to}.{col_to}", f"{t_from}.{col_from} = {t_to}.{col_to}"

def make_aggregate_clause(table):
    agg = random.choice(["COUNT", "SUM", "AVG", "MAX", "MIN"])
    col = random.choice([c for c in tables[table] if c not in ("id","name","title")])
    if col is None:
        col = "id"
    return f"{agg}({col}) AS {agg.lower()}_{col}", agg, col

def generate_example(i):
    # choose a pattern
    patterns = [
        "simple_select", "select_where", "select_order_limit", "join_select", "aggregate_group", "aggregate_having", "multi_join", "subquery_in", "between_dates", "count_distinct", "select_specific_columns"
    ]
    weights = [0.08, 0.20, 0.12, 0.18, 0.10, 0.05, 0.07, 0.07, 0.06, 0.03, 0.04]
    pat = random.choices(patterns, weights)[0]

    if pat == "simple_select":
        t = random.choice(table_names)
        cols = sample_columns(t, 1, min(4, len(tables[t])))
        prompt = f"List {format_cols(cols)} from {t}"
        sql = f"SELECT {format_cols(cols)} FROM {t};"
    elif pat == "select_where":
        t = random.choice(table_names)
        cols = sample_columns(t, 1, min(4, len(tables[t])))
        where_clause, where_desc = make_where_clause(t)
        prompt = f"Get {format_cols(cols)} from {t} where {where_desc}"
        sql = f"SELECT {format_cols(cols)} FROM {t} {where_clause};"
    elif pat == "select_order_limit":
        t = random.choice(table_names)
        cols = sample_columns(t, 1, min(4, len(tables[t])))
        order_col = random.choice(tables[t])
        direction = random.choice(["ASC", "DESC"])
        limit = random.choice([5,10,20,50])
        prompt = f"Get {format_cols(cols)} from {t} ordered by {order_col} {direction} limit {limit}"
        sql = f"SELECT {format_cols(cols)} FROM {t} ORDER BY {order_col} {direction} LIMIT {limit};"
    elif pat == "join_select":
        a, b = random.sample(table_names, 2)
        cols_a = sample_columns(a, 1, min(3,len(tables[a])))
        cols_b = sample_columns(b, 1, min(3,len(tables[b])))
        join_clause, join_desc = make_join(a,b)
        prompt = f"Join {a} and {b} to get {', '.join(cols_a+cols_b)} where {join_desc}"
        sql = f"SELECT {a}.{format_cols(cols_a)}, {b}.{format_cols(cols_b)} FROM {a} {join_clause};"
    elif pat == "aggregate_group":
        t = random.choice(table_names)
        agg_expr, agg_name, agg_col = make_aggregate_clause(t)
        group_col = random.choice([c for c in tables[t] if c not in (agg_col, "id")] or ["id"])
        prompt = f"Get {agg_name} of {agg_col} grouped by {group_col} from {t}"
        sql = f"SELECT {group_col}, {agg_expr} FROM {t} GROUP BY {group_col};"
    elif pat == "aggregate_having":
        t = random.choice(table_names)
        agg_expr, agg_name, agg_col = make_aggregate_clause(t)
        group_col = random.choice([c for c in tables[t] if c not in (agg_col, "id")] or ["id"])
        comp = random.randint(1, 50)
        prompt = f"Get {group_col} having {agg_name}({agg_col}) > {comp} from {t}"
        sql = f"SELECT {group_col}, {agg_expr} FROM {t} GROUP BY {group_col} HAVING {agg_name}({agg_col}) > {comp};"
    elif pat == "multi_join":
        a, b, c = random.sample(table_names, 3)
        cols = sample_columns(a,1,2) + sample_columns(b,1,2) + sample_columns(c,1,2)
        join1, desc1 = make_join(a,b)
        join2, desc2 = make_join(a,c)
        prompt = f"Join {a}, {b}, and {c} to select {format_cols(cols)}"
        sql = f"SELECT {format_cols([f'{a}.{col}' for col in sample_columns(a,1,2)] + [f'{b}.{col}' for col in sample_columns(b,1,2)] + [f'{c}.{col}' for col in sample_columns(c,1,2)])} FROM {a} {join1} {join2};"
    elif pat == "subquery_in":
        # select rows where id IN (subquery)
        t = random.choice(table_names)
        # choose another table for subquery
        other = random.choice([x for x in table_names if x!=t])
        col = random.choice(tables[t])
        subcol = random.choice(tables[other])
        prompt = f"Select records from {t} where {col} is in {other} {subcol}"
        sql = f"SELECT * FROM {t} WHERE {col} IN (SELECT {subcol} FROM {other});"
    elif pat == "between_dates":
        t = random.choice(table_names)
        date_cols = [c for c in tables[t] if "date" in c or "created_at" in c or "hire_date" in c or "release_date" in c or "transaction_date" in c]
        if not date_cols:
            date_col = random.choice(tables[t])
        else:
            date_col = random.choice(date_cols)
        d1 = random_date(2019,2023)
        d2 = random_date(2024,2025)
        if d1 > d2:
            d1, d2 = d2, d1
        prompt = f"Get records from {t} where {date_col} between {d1} and {d2}"
        sql = f"SELECT * FROM {t} WHERE {date_col} BETWEEN '{d1}' AND '{d2}';"
    elif pat == "count_distinct":
        t = random.choice(table_names)
        col = random.choice([c for c in tables[t] if c not in ("id",)])
        prompt = f"Count distinct {col} in {t}"
        sql = f"SELECT COUNT(DISTINCT {col}) FROM {t};"
    elif pat == "select_specific_columns":
        t = random.choice(table_names)
        cols = sample_columns(t, 2, min(5, len(tables[t])))
        prompt = f"Get {format_cols(cols)} from {t}"
        sql = f"SELECT {format_cols(cols)} FROM {t};"
    else:
        # fallback simple
        t = random.choice(table_names)
        prompt = f"List all from {t}"
        sql = f"SELECT * FROM {t};"

    # Make sure SQL ends with semicolon
    if not sql.strip().endswith(";"):
        sql = sql.strip() + ";"

    return {"prompt": prompt, "response": sql}

# Generate exactly 10,000 examples
N = 2500000
out_path = "sql_dataset_10000.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for i in range(N):
        ex = generate_example(i)
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

out_path, N

