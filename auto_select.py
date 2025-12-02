import itertools
import numpy as np
from scipy.optimize import linprog
import pandas as pd

DATA = {
    'material': ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13'],
    'density_min': [2.94, 1, 3.6, 2.42, 3.8, 3.59, 7.8, 5.7, 3.8, 4, 3, 7.8, 4.47],
    'density_max': [3.11, 1, 4, 2.75, 4.2, 3.72, 7.8, 6.2, 4.2, 4.4, 4, 7.8, 4.57],
    'price':       [18,   0, 10.7, 3.1, 0, 10.7, 51.1, 29, 15.7, 19.40, 5.5, 51.2, 18]
}
df = pd.DataFrame(DATA)

volume_L = 1647
shell_weight = 477
lid_weight = 7
total_weight = 7600
concrete_mass = total_weight - shell_weight - lid_weight
target_density = concrete_mass / volume_L

min_user = 0.10
max_user = 0.40
epsilon = 1e-6

materials = list(df['material'])

def solve_mix(selected):
    sub = df[df.material.isin(selected)].reset_index(drop=True)

    p_m1 = 0.06
    p_m2 = 0.03

    A_ub = []
    b_ub = []

    # ràng buộc density
    A_ub.append(sub.density_min.values)
    b_ub.append(target_density)

    A_ub.append(-sub.density_max.values)
    b_ub.append(-target_density)

    # tổng % = 1
    A_eq = [[1] * len(sub)]
    b_eq = [1]

    # bound cho từng vật liệu
    bounds = []
    for m in sub.material:
        if m == "m1": bounds.append((p_m1, p_m1))
        elif m == "m2": bounds.append((p_m2, p_m2))
        else: bounds.append((min_user, max_user))

    # tránh price=0
    epsilon = 1e-6
    prices_for_lp = np.array([p if p > 0 else epsilon for p in sub.price.values])

    res = linprog(prices_for_lp,
                  A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        return None

    sol = res.x

    # tạo dataframe kết quả
    result = pd.DataFrame({
        'material': sub.material,
        'proportion': sol * 100,
        'density_min': sub.density_min,
        'mass_kg': np.round(sol * concrete_mass).astype(int),
        'density_max': sub.density_max,
        'price': sub.price,
        'amount': np.round(sol * concrete_mass * sub.price, 2)
    })

    return result, result.amount.sum()

print("Auto searching best mix...\n")

best_result = None

base = ['m1', 'm2']
others = [m for m in materials if m not in base]

best_per_k = {}

for k in range(3, 7):
    best_cost = float("inf")
    best_solution = None
    bestMaterials = None
    ##
    for combo in itertools.combinations(others, k):
        ##
        selected = base + list(combo)
        r = solve_mix(selected)

        if r is None:
            continue
        ##
        result, total_cost = r
        if total_cost < best_cost:
            best_cost = total_cost
            best_solution = result
            bestMaterials = selected
    ##
    if best_solution is not None:
        best_per_k[k] = (bestMaterials, best_solution, best_cost)
        print(f"Tổ hợp tốt nhất cho k={k}: {bestMaterials}, Cost = {best_cost:,.2f}")
    else:
        print(f"Không tìm được tổ hợp hợp lệ cho k={k}")

for k, (materials, dfres, cost) in best_per_k.items():
    print(f"\n--------------------------")
    print(f"k = {k} → Cost = {cost:,.2f}")
    print(f"Materials: {materials}")
    print(dfres)
    print(f"Total mass = {dfres.mass_kg.sum()} kg")
    print(f"Cost per kg = {(cost / dfres.mass_kg.sum()):.4f}")