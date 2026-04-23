from pipeline import run_pipeline
from time import perf_counter

if __name__ == "__main__":
    start = perf_counter()

    run_pipeline()
    
    elapsed = perf_counter() - start
    print("\n=========================================")
    print(f"[PIPELINE] Pipeline complete in {elapsed:.2f} seconds")