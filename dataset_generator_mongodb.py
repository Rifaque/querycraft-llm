import json
import random
from datetime import datetime, timedelta
import faker
from multiprocessing import Pool, cpu_count

fake = faker.Faker()
random.seed(42)

collections = {
    "users": ["_id", "name", "email", "registration_date", "country", "age", "status"],
    "products": ["_id", "name", "category_id", "price", "stock", "created_at", "rating"],
    "orders": ["_id", "user_id", "order_date", "status", "total_amount", "payment_method"],
    "reviews": ["_id", "product_id", "user_id", "rating", "comment", "created_at"],
    "employees": ["_id", "first_name", "last_name", "department_id", "hire_date", "salary"],
    "departments": ["_id", "name", "manager_id", "budget"]
}
col_names = list(collections.keys())

def random_date(start_year=2018, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    d = start + timedelta(days=random.randint(0, delta))
    return d.strftime("%Y-%m-%d")

def gen_value(field):
    if "date" in field or "created_at" in field or "order_date" in field or "hire_date" in field:
        return f"'{random_date()}'"
    if field in ["name", "first_name", "last_name"]:
        return f"'{fake.first_name()}'"
    if field == "email":
        return f"'{fake.email()}'"
    if field in ["status", "payment_method", "country"]:
        return f"'{random.choice(['US','UK','IN','pending','completed','credit_card','UPS'])}'"
    if field in ["age", "price", "stock", "salary", "rating", "total_amount", "budget"]:
        return random.randint(1, 500)
    if field == "comment":
        return f"'{fake.sentence()}'"
    return f"'{fake.word()}'"

# Different phrasing templates for natural language prompts
prompt_templates = {
    "find_all": [
        "List all documents from {col}",
        "Show me every record in {col}",
        "Retrieve all entries from {col}"
    ],
    "find_filter": [
        "Find documents in {col} where {field} equals {value}",
        "Get all {col} records with {field} = {value}",
        "Show {col} entries matching {field} = {value}"
    ],
    "projection": [
        "Get only {fields} from {col}",
        "List {fields} fields for all {col}",
        "Retrieve the {fields} attributes from {col} documents"
    ],
    "sort_limit": [
        "Get top {limit} {col} sorted by {field} ({order})",
        "List {limit} {col} in {'ascending' if order==1 else 'descending'} order by {field}",
        "Retrieve {limit} documents from {col} sorted on {field} {order_text}"
    ],
    "aggregate_match_group": [
        "Group {col} by {field} and count documents",
        "Show count of {col} records for each {field}",
        "Aggregate {col} entries by {field} and total them"
    ],
    "lookup": [
        "Join {col} with {other} using {local_field}",
        "Combine {col} and {other} on {local_field}",
        "Merge {col} with {other} documents through {local_field}"
    ],
    "insert": [
        "Insert document {data} into {col}",
        "Add a new record {data} to {col}",
        "Create a new entry {data} in {col}"
    ],
    "update_one": [
        "Update one document in {col} where {field1}={val1} to set {field2}={val2}",
        "Modify a single {col} record with {field1}={val1}, changing {field2} to {val2}",
        "Change {field2} to {val2} for one {col} entry matching {field1}={val1}"
    ],
    "update_many": [
        "Update all documents in {col} where {field1}={val1} to set {field2}={val2}",
        "Modify multiple {col} entries with {field1}={val1} and set {field2}={val2}",
        "Change {field2} to {val2} for all {col} records where {field1}={val1}"
    ],
    "delete_one": [
        "Delete one document in {col} where {field}={val}",
        "Remove a single {col} record with {field}={val}",
        "Delete one entry from {col} where {field}={val}"
    ],
    "delete_many": [
        "Delete all documents in {col} where {field}={val}",
        "Remove every {col} record where {field}={val}",
        "Delete all entries from {col} matching {field}={val}"
    ]
}

def generate_example(_):
    col = random.choice(col_names)
    fields = collections[col]
    pat = random.choice(list(prompt_templates.keys()))

    if pat == "find_all":
        query = f"db.{col}.find({{}})"
        prompt = random.choice(prompt_templates[pat]).format(col=col)
    
    elif pat == "find_filter":
        f = random.choice(fields)
        val = gen_value(f)
        query = f"db.{col}.find({{{f}: {val}}})"
        prompt = random.choice(prompt_templates[pat]).format(col=col, field=f, value=val)
    
    elif pat == "projection":
        proj = random.sample(fields, min(2, len(fields)))
        query = f"db.{col}.find({{}}, {{{', '.join([f'{p}:1' for p in proj])}}})"
        prompt = random.choice(prompt_templates[pat]).format(col=col, fields=", ".join(proj))
    
    elif pat == "sort_limit":
        f = random.choice(fields)
        order = random.choice([1, -1])
        limit = random.choice([5,10,20])
        query = f"db.{col}.find({{}}).sort({{{f}: {order}}}).limit({limit})"
        order_text = "ascending" if order == 1 else "descending"
        prompt = random.choice(prompt_templates[pat]).format(col=col, limit=limit, field=f, order=order, order_text=order_text)
    
    elif pat == "aggregate_match_group":
        f = random.choice(fields)
        query = f"db.{col}.aggregate([{{'$group': {{'_id': '${f}', 'count': {{'$sum': 1}}}}}}])"
        prompt = random.choice(prompt_templates[pat]).format(col=col, field=f)
    
    elif pat == "lookup":
        other = random.choice([x for x in col_names if x != col])
        local_field = "user_id" if "user_id" in fields else fields[0]
        query = f"db.{col}.aggregate([{{'$lookup': {{'from': '{other}', 'localField': '{local_field}', 'foreignField': '_id', 'as': '{other}_docs'}}}}])"
        prompt = random.choice(prompt_templates[pat]).format(col=col, other=other, local_field=local_field)
    
    elif pat == "insert":
        data = {f: gen_value(f) for f in random.sample(fields, min(3, len(fields)))}
        query = f"db.{col}.insertOne({data})"
        prompt = random.choice(prompt_templates[pat]).format(col=col, data=data)
    
    elif pat in ["update_one", "update_many"]:
        f1 = random.choice(fields)
        f2 = random.choice([f for f in fields if f != f1])
        val1 = gen_value(f1)
        val2 = gen_value(f2)
        query_type = "updateOne" if pat=="update_one" else "updateMany"
        query = f"db.{col}.{query_type}({{{f1}: {val1}}}, {{'$set': {{{f2}: {val2}}}}})"
        prompt = random.choice(prompt_templates[pat]).format(col=col, field1=f1, val1=val1, field2=f2, val2=val2)
    
    elif pat in ["delete_one", "delete_many"]:
        f = random.choice(fields)
        val = gen_value(f)
        query_type = "deleteOne" if pat=="delete_one" else "deleteMany"
        query = f"db.{col}.{query_type}({{{f}: {val}}})"
        prompt = random.choice(prompt_templates[pat]).format(col=col, field=f, val=val)

    return {"prompt": prompt, "response": query}


# ---- Multiprocessing generation ----
def generate_batch(batch_size):
    return [generate_example(i) for i in range(batch_size)]

if __name__ == "__main__":
    N = 1000000        # total examples
    BATCH = 10000      # examples per process
    workers = cpu_count() - 1
    out_path = "data/mongo_crud_dataset_natural.jsonl"

    with open(out_path, "w", encoding="utf-8") as f, Pool(workers) as pool:
        for batch in pool.imap(generate_batch, [BATCH]* (N // BATCH)):
            for ex in batch:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset generated: {out_path} with {N} examples")
