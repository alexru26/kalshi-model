"""Draw the E1 to E3 figures from the saved results into results/figures/.

Reads only results/*.csv, so it redraws without rerunning (`kalshi_model.experiment_figures`
lists each figure and the prediction it shows).

Usage: uv run python experiments/figures.py
"""

from kalshi_model.experiment_figures import write_figures

if __name__ == "__main__":
    for path in write_figures():
        print(f"wrote {path}")
