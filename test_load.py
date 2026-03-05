import time, joblib, os

MODEL_PATH = "saved_models/random_forest_tuned_best.pkl"

print("CWD:", os.getcwd())
print("Exists:", os.path.exists(MODEL_PATH))
print("Loading model...")

t0 = time.time()
obj = joblib.load(MODEL_PATH)
t1 = time.time()

print("Loaded in", round(t1 - t0, 2), "seconds")
print("Type:", type(obj))
print("Keys:", list(obj.keys()) if isinstance(obj, dict) else "-")