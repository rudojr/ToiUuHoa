import numpy as np
from scipy.optimize import linprog
import pandas as pd

DATA = {
    'material': ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13'],
    'density_min': [2.94, 1, 3.6, 2.42, 3.8, 3.59, 7.8, 5.7, 3.8, 4, 3, 7.8, 4.47],
    'density_max': [3.11, 1, 4, 2.75, 4.2, 3.72, 7.8, 6.2, 4.2, 4.4, 4, 7.8, 4.57],
    'price': [18, 0, 10.7, 3.1, 0, 10.7, 51.1, 29, 15.7, 19.40, 5.5, 51.2, 18]
}

# Thông tin lon
volume_L = 1647       # Lít
shell_weight = 477    # kg
lid_weight = 7        # kg
total_weight = 7600   # kg (bao gồm vỏ + nắp + bê tông)
error_margin = 38     # kg

FIXED_FRACTIONS = {'m1': 0.06, 'm2': 0.03}
NUM_EXTRA = 4
df = pd.DataFrame(DATA)
print(df)

price_map = dict(zip(df['material'], df['price']))
dmin_map = dict(zip(df['material'], df['density_min']))
dmax_map = dict(zip(df['material'], df['density_max']))

concrete_mass = total_weight - shell_weight - lid_weight
print(f"Concrete mass (kg): {concrete_mass}")

target_density = concrete_mass / volume_L
print(f"Target density: {target_density:.4f} kg/L")

print("Available materials besides m1, m2:")
print([m for m in df.material if m not in ['m1','m2']])

candidates = [f"m{i}" for i in range(3, 14)]

user_materials = []
while len(user_materials) < 4:
    x = input(f"Enter material {len(user_materials)+1}/4: ").strip()
    if x in df.material.values and x not in ['m1','m2'] and x not in user_materials:
        user_materials.append(x)
    else:
        print("Invalid or duplicate. Try again.")

print("User selected:", user_materials)

selected = ['m1','m2'] + user_materials
sub = df[df.material.isin(selected)].reset_index(drop=True)

p_m1 = 0.06
p_m2 = 0.03
fixed_total = p_m1 + p_m2

remaining = 1 - fixed_total

A_ub = []
b_ub = []

A_ub.append(sub.density_min.values)
b_ub.append(target_density)

A_ub.append(-sub.density_max.values)
b_ub.append(-target_density)

A_eq = [ [1]*len(selected) ]
b_eq = [1]

bounds = []
min_user = 0.1
max_user = 0.4
for m in sub.material:
    if m == 'm1': bounds.append((p_m1,p_m1))
    elif m == 'm2': bounds.append((p_m2,p_m2))
    else: bounds.append((min_user, max_user))

epsilon = 1e-6
prices_for_lp = np.array([p if p > 0 else epsilon for p in sub.price.values])
res = linprog(prices_for_lp, A_ub=A_ub, b_ub=b_ub,A_eq=A_eq, b_eq=b_eq,bounds=bounds, method='highs')


if not res.success:
    print("No feasible solution.")
    exit()

solution = res.x
result = pd.DataFrame({
            'material': sub.material,
            'proportion': solution * 100,
            'density_min': sub.density_min,
            'mass_kg': np.round(solution * concrete_mass).astype(int),
            'density_max': sub.density_max,
            'price': sub.price,
            'amount': np.round(solution * concrete_mass * sub.price, 2)
        })

result['proportion'] = (solution * 100).round(2).astype(str) + '%'
print("Optimal Mix:")
print(result)
print(f"Total mass = {result.mass_kg.sum().astype(int)} kg")
print(f"Total cost = {result.amount.sum():,.2f}")
print(f"Cost per kg = {(result.amount.sum() / result.mass_kg.sum()).round(2)}")