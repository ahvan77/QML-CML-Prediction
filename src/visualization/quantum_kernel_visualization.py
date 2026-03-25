from qiskit.circuit.library import ZZFeatureMap

feature_cols = ["A_scaled", "B_scaled", "C_scaled", "D_scaled", "E_scaled"]
num_features = len(feature_cols)

# Choose the reps you want to display (e.g., reps = 2)
reps = 1
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=reps)

# Text diagram
print(feature_map.draw())

# Matplotlib figure
feature_map.draw(output="mpl", fold=20, filename=f"zzfeaturemap_{num_features}q_reps{reps}.png")

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=reps)
decomposed_map = feature_map.decompose()   # expand into basic gates

print(decomposed_map.draw())

decomposed_map.draw(
    output="mpl",
    fold=20,                 # wider diagrams before folding
    filename=f"zzfeaturemap_{num_features}q_reps{reps}_decomposed.png"
)

