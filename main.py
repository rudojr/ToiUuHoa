from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from itertools import combinations
import pandas as pd

app = FastAPI()

product_df = pd.read_csv("test/product.csv")
material_df = pd.read_csv("test/material.csv")

material_df.columns = [c.strip() for c in material_df.columns]
product_df.columns = [c.strip() for c in product_df.columns]

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

material_df["Density min"] = (
    material_df["Density min"].str.replace(",", ".", regex=False).astype(float)
)
material_df["Density max"] = (
    material_df["Density max"].str.replace(",", ".", regex=False).astype(float)
)
material_df["Unit Price (Yen/Kg)"] = (
    material_df["Unit Price (Yen/Kg)"].str.replace(",", ".", regex=False).astype(float)
)

DATA = {
    "material": material_df["Material Name"].tolist(),
    "density_min": material_df["Density min"].tolist(),
    "density_max": material_df["Density max"].tolist(),
    "price": material_df["Unit Price (Yen/Kg)"].tolist(),
}


class OptimizeInput(BaseModel):
    product_id: int
    total_weight: float

def optimize_concrete_mix(product_id: int, total_weight: float):

    if product_id not in product_df["Product ID"].values:
        raise HTTPException(status_code=404, detail="Product ID not found")

    row = product_df[product_df["Product ID"] == product_id].iloc[0]

    CONTAINER = {
        "volume": float(row["Volume"]),
        "shell_weight": float(row["Can Weight"]),
        "cap_weight": float(row["Cap Weight"]),
        "error_low": float(row["Lower Tolerance"]),
        "error_high": float(row["Upper Tolerance"])
    }

    # USER INPUT TOTAL WEIGHT (override)
    concrete_weight = total_weight - CONTAINER["shell_weight"] - CONTAINER["cap_weight"]

    if concrete_weight <= 0:
        raise HTTPException(status_code=400, detail="Total weight too small")

    M1_RATIO = 0.06
    M2_RATIO = 0.03
    REMAINING_RATIO = 0.91

    materials_pool = list(range(2, len(DATA["material"])))

    best_result = None
    min_cost = float("inf")

    for selected_indices in combinations(materials_pool, 4):

        all_indices = [0, 1] + list(selected_indices)

        def objective(x):
            ratios = np.array([M1_RATIO, M2_RATIO] + list(x))
            cost = sum(
                ratios[i] * concrete_weight * DATA["price"][all_indices[i]]
                for i in range(len(all_indices))
            )
            return cost

        def constraint_volume(x):
            ratios = np.array([M1_RATIO, M2_RATIO] + list(x))

            total_volume = 0
            for i, idx in enumerate(all_indices):
                w = ratios[i] * concrete_weight
                avg_d = (DATA["density_min"][idx] + DATA["density_max"][idx]) / 2
                total_volume += w / avg_d

            error_ml = (total_volume - CONTAINER["volume"]) * 1000

            return [
                error_ml - CONTAINER["error_low"],
                CONTAINER["error_high"] - error_ml,
            ]

        def constraint_density(x):
            ratios = np.array([M1_RATIO, M2_RATIO] + list(x))

            volumes = []
            for i, idx in enumerate(all_indices):
                w = ratios[i] * concrete_weight
                avg_d = (DATA["density_min"][idx] + DATA["density_max"][idx]) / 2
                volumes.append(w / avg_d)

            total_volume = sum(volumes)
            mix_density = concrete_weight / total_volume

            density_min_mix = min(DATA["density_min"][idx] for idx in all_indices)
            density_max_mix = max(DATA["density_max"][idx] for idx in all_indices)

            return [
                mix_density - density_min_mix,
                density_max_mix - mix_density - 0.01
            ]

        constraints = [
            {"type": "eq", "fun": lambda x: sum(x) - REMAINING_RATIO},
            {"type": "ineq", "fun": constraint_volume},
            {"type": "ineq", "fun": constraint_density},
        ]

        bounds = Bounds([0.09] * 4, [0.5] * 4)
        x0 = np.array([REMAINING_RATIO / 4] * 4)

        try:
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 800, "ftol": 1e-9},
            )
            if res.success and res.fun < min_cost:
                min_cost = res.fun
                best_result = {
                    "indices": all_indices,
                    "ratios": np.array([M1_RATIO, M2_RATIO] + list(res.x)),
                    "cost": res.fun,
                }
        except:
            continue

    if best_result is None:
        raise HTTPException(status_code=400, detail="No valid formula found")

    ratios = best_result["ratios"]
    indices = best_result["indices"]

    output = []

    for i, idx in enumerate(indices):
        avg_d = (DATA["density_min"][idx] + DATA["density_max"][idx]) / 2
        weight = ratios[i] * concrete_weight

        output.append({
            "material": DATA["material"][idx],
            "ratio": float(ratios[i]),
            "weight": float(weight),
            "density_min": DATA["density_min"][idx],
            "density_max": DATA["density_max"][idx],
            "density_avg": avg_d,
            "price": DATA["price"][idx],
        })

    total_volume = sum([
        (ratios[i] * concrete_weight) /
        ((DATA["density_min"][indices[i]] + DATA["density_max"][indices[i]]) / 2)
        for i in range(len(indices))
    ])

    print(output)

    return {
        "product_id": product_id,
        "input_total_weight": total_weight,
        "concrete_weight": float(concrete_weight),

        "materials": output,
        "total_volume": float(total_volume),
        "mix_density": float(concrete_weight / total_volume),
        "cost": float(best_result["cost"])
    }

@app.post("/api/optimize")
def api_optimize(data: OptimizeInput):
    return optimize_concrete_mix(data.product_id, data.total_weight)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)