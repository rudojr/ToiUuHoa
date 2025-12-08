import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import pandas as pd
from itertools import combinations

# Dữ liệu đầu vào
DATA = {
    'material': ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13'],
    'density_min': [2.94, 1, 3.6, 2.42, 3.8, 3.59, 7.8, 5.7, 3.8, 4, 3, 7.8, 4.47],
    'density_max': [3.11, 1, 4, 2.75, 4.2, 3.72, 7.8, 6.2, 4.2, 4.4, 4, 7.8, 4.57],
    'price': [18, 0, 10.7, 3.1, 0, 10.7, 51.1, 29, 15.7, 19.40, 5.5, 51.2, 18]
}

# Thông số lon rỗng
CONTAINER = {
    'volume': 1647,  # Lít
    'shell_weight': 477,  # kg
    'cap_weight': 7,  # kg
    'error_low': -38,  # mL
    'error_high': 114  # mL
}


def optimize_concrete_mix(total_weight):

    concrete_weight = total_weight - CONTAINER['shell_weight'] - CONTAINER['cap_weight']
    concrete_density = concrete_weight / CONTAINER['volume']

    print(f"Khối lượng bê tông: {concrete_weight:.3f} kg")
    print(f"Khối lượng riêng bê tông: {concrete_density:.3f} kg/L")
    print("-" * 80)

    M1_RATIO = 0.06
    M2_RATIO = 0.03
    REMAINING_RATIO = 0.91

    materials_pool = list(range(2, len(DATA['material'])))  # index từ 2 đến 12

    best_result = None
    min_cost = float('inf')

    for selected_indices in combinations(materials_pool, 4):
        all_indices = [0, 1] + list(selected_indices)

        # Hàm mục tiêu: minimize cost
        def objective(x):
            # x là tỉ lệ của 4 vật liệu còn lại (tổng = 0.91)
            ratios = np.array([M1_RATIO, M2_RATIO] + list(x))
            cost = sum(ratios[i] * concrete_weight * DATA['price'][all_indices[i]]
                       for i in range(len(all_indices)))

            # Penalty nếu dồn vào vật liệu giá 0
            for i, idx in enumerate(all_indices):
                if DATA['price'][idx] == 0 and ratios[i] > 0.4:
                    cost += 1000 * (ratios[i] - 0.4)

            return cost

        # Hàm ràng buộc
        def constraint_volume(x):
            ratios = np.array([M1_RATIO, M2_RATIO] + list(x))
            volumes = []
            for i, idx in enumerate(all_indices):
                weight = ratios[i] * concrete_weight
                avg_density = (DATA['density_min'][idx] + DATA['density_max'][idx]) / 2
                volume = weight / avg_density
                volumes.append(volume)
            total_volume = sum(volumes)
            volume_error_ml = (total_volume - CONTAINER['volume']) * 1000
            # Ràng buộc: error_low <= volume_error <= error_high
            return [volume_error_ml - CONTAINER['error_low'],
                    CONTAINER['error_high'] - volume_error_ml]

        def constraint_density(x):
            ratios = np.array([M1_RATIO, M2_RATIO] + list(x))
            volumes = []
            for i, idx in enumerate(all_indices):
                weight = ratios[i] * concrete_weight
                avg_density = (DATA['density_min'][idx] + DATA['density_max'][idx]) / 2
                volume = weight / avg_density
                volumes.append(volume)
            total_volume = sum(volumes)
            mix_density = concrete_weight / total_volume

            # Tính density range của hỗn hợp
            density_min_mix = min(DATA['density_min'][idx] for idx in all_indices)
            density_max_mix = max(DATA['density_max'][idx] for idx in all_indices)

            # Ràng buộc: density_min_mix <= mix_density < density_max_mix
            return [mix_density - density_min_mix,
                    density_max_mix - mix_density - 0.01]

        constraints = [
            {'type': 'eq', 'fun': lambda x: sum(x) - REMAINING_RATIO},  # Tổng = 0.91
            {'type': 'ineq', 'fun': constraint_volume},
            {'type': 'ineq', 'fun': constraint_density}
        ]

        # Bounds cho 4 biến (tỉ lệ từ 0 đến 0.91)
        bounds = Bounds([0, 0, 0, 0], [REMAINING_RATIO] * 4)

        # Giá trị khởi tạo
        x0 = np.array([REMAINING_RATIO / 4] * 4)

        # Tối ưu hóa
        try:
            result = minimize(objective, x0, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 1000, 'ftol': 1e-9})

            if result.success and result.fun < min_cost:
                min_cost = result.fun
                best_result = {
                    'indices': all_indices,
                    'ratios': np.array([M1_RATIO, M2_RATIO] + list(result.x)),
                    'cost': result.fun
                }
        except:
            continue

    if best_result is None:
        print("Không tìm được công thức phù hợp!")
        return None

    ratios = best_result['ratios']
    indices = best_result['indices']

    results = []
    total_volume = 0

    for i, idx in enumerate(indices):
        weight = ratios[i] * concrete_weight
        avg_density = (DATA['density_min'][idx] + DATA['density_max'][idx]) / 2
        volume = weight / avg_density
        total_volume += volume
        amount = weight  # kg

        results.append({
            'Material': DATA['material'][idx],
            'Proportion (%)': f"{ratios[i] * 100:.2f}",
            'Density Min': DATA['density_min'][idx],
            'Density Max': DATA['density_max'][idx],
            'Density Avg': f"{avg_density:.2f}",
            'Weight (kg)': f"{weight:.3f}",
            'Price ($/kg)': DATA['price'][idx],
            'Amount (kg)': f"{amount:.3f}"
        })

    df = pd.DataFrame(results)

    total_mass = concrete_weight
    mix_density = total_mass / total_volume
    volume_error_ml = (total_volume - CONTAINER['volume']) * 1000
    total_cost = best_result['cost']

    # In kết quả
    print("\n" + "=" * 80)
    print("KẾT QUẢ TỐI ƯU HÓA ")
    print("=" * 80)
    print(df.to_string(index=False))
    print("-" * 80)
    print(f"Total Mass:                {total_mass:.3f} kg")
    print(f"Total Volume:              {total_volume:.6f} L ({total_volume * 1000:.2f} mL)")
    print(
        f"Volume Error:              {volume_error_ml:.2f} mL (Range: {CONTAINER['error_low']} to {CONTAINER['error_high']} mL)")
    print(f"Mix Density:               {mix_density:.3f} kg/L")
    print(f"Total Cost:                ${total_cost:.2f}")
    print("=" * 80)

    print("\nKIỂM TRA RÀNG BUỘC:")
    print(f"Tổng tỉ lệ = {sum(ratios) * 100:.2f}%")
    print(f"Số vật liệu = {len(indices)} (≤ 6)")
    print(f"M1 (cát) = {ratios[0] * 100:.2f}%")
    print(f"M2 (xi măng) = {ratios[1] * 100:.2f}%")

    return df


if __name__ == "__main__":
    print("=" * 80)

    total_weight = float(input("Nhập khối lượng tổng (kg): "))

    result = optimize_concrete_mix(total_weight)