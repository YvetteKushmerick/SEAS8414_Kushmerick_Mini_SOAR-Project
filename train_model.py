# train_model.py
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless for Docker/CI
# import matplotlib.pyplot as plt  # not needed directly

# ---- PyCaret (classification) ----
from pycaret.classification import (
    setup as cls_setup,
    compare_models,
    finalize_model as cls_finalize,
    plot_model as cls_plot_model,
    save_model as cls_save_model,
)

# ---- PyCaret (clustering) ----
from pycaret.clustering import (
    setup as u_setup,
    create_model as u_create_model,
    assign_model as u_assign_model,
    save_model as u_save_model,
)


def generate_synthetic_data(num_samples=500, seed=42):
    """Generates a synthetic dataset of phishing and benign URL features."""
    rng = np.random.default_rng(seed)

    # Feature list 
    features = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service',
        'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
        'having_Sub_Domain', 'SSLfinal_State', 'URL_of_Anchor', 'Links_in_tags',
        'SFH', 'Abnormal_URL', 'has_political_keyword'
    ]

    num_phishing = num_samples // 2
    num_benign = num_samples - num_phishing

    phishing_data = {
        'having_IP_Address': rng.choice([1, -1], num_phishing, p=[0.3, 0.7]),
        'URL_Length': rng.choice([1, 0, -1], num_phishing, p=[0.5, 0.4, 0.1]),
        'Shortining_Service': rng.choice([1, -1], num_phishing, p=[0.6, 0.4]),
        'having_At_Symbol': rng.choice([1, -1], num_phishing, p=[0.4, 0.6]),
        'double_slash_redirecting': rng.choice([1, -1], num_phishing, p=[0.3, 0.7]),
        'Prefix_Suffix': rng.choice([1, -1], num_phishing, p=[0.7, 0.3]),
        'having_Sub_Domain': rng.choice([1, 0, -1], num_phishing, p=[0.6, 0.3, 0.1]),
        'SSLfinal_State': rng.choice([-1, 0, 1], num_phishing, p=[0.6, 0.3, 0.1]),
        'URL_of_Anchor': rng.choice([-1, 0, 1], num_phishing, p=[0.5, 0.3, 0.2]),
        'Links_in_tags': rng.choice([-1, 0, 1], num_phishing, p=[0.4, 0.4, 0.2]),
        'SFH': rng.choice([-1, 0, 1], num_phishing, p=[0.7, 0.2, 0.1]),
        'Abnormal_URL': rng.choice([1, -1], num_phishing, p=[0.5, 0.5]),
        'has_political_keyword': np.ones(num_phishing, dtype=int)
    }

    benign_data = {
        'having_IP_Address': rng.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'URL_Length': rng.choice([1, 0, -1], num_benign, p=[0.1, 0.6, 0.3]),
        'Shortining_Service': rng.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_At_Symbol': rng.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'double_slash_redirecting': rng.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'Prefix_Suffix': rng.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_Sub_Domain': rng.choice([1, 0, -1], num_benign, p=[0.1, 0.4, 0.5]),
        'SSLfinal_State': rng.choice([-1, 0, 1], num_benign, p=[0.05, 0.15, 0.8]),
        'URL_of_Anchor': rng.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'Links_in_tags': rng.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'SFH': rng.choice([-1, 0, 1], num_benign, p=[0.1, 0.1, 0.8]),
        'Abnormal_URL': rng.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'has_political_keyword': np.ones(num_benign, dtype=int)
    }

    df_phishing = pd.DataFrame(phishing_data)
    df_benign = pd.DataFrame(benign_data)

    df_phishing['label'] = 1
    df_benign['label'] = 0

    final_df = pd.concat([df_phishing, df_benign], ignore_index=True)
    return final_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train():
    # ---- Paths
    models_dir = "models"
    data_dir = "data"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    cls_model_base = os.path.join(models_dir, "phishing_url_detector")  # PyCaret adds .pkl
    plot_path = os.path.join(models_dir, "feature_importance.png")

    # Generate & save dataset
    data = generate_synthetic_data()
    data_csv = os.path.join(data_dir, "phishing_synthetic.csv")
    data.to_csv(data_csv, index=False)

    # ---- Classification
    if not os.path.exists(cls_model_base + ".pkl"):
        print("Initializing PyCaret Classification setup...")
        s = cls_setup(data=data, target="label", session_id=42, verbose=False)

        print("Comparing models...")
        best_model = compare_models(n_select=1, include=['rf', 'et', 'lightgbm'])

        print("Finalizing model...")
        final_model = cls_finalize(best_model)

        print("Saving feature importance plot...")
        cls_plot_model(final_model, plot='feature', save=True)
        # PyCaret saves as "Feature Importance.png" in CWD by default
        default_plot = "Feature Importance.png"
        if os.path.exists(default_plot):
            os.replace(default_plot, plot_path)
        else:
            print("[Warn] Expected feature importance file not found; plot not saved.")

        print("Saving classification model...")
        cls_save_model(final_model, cls_model_base)
        print("[OK] Classification model saved.")
    else:
        print("[Skip] Classification model already exists.")

    # ---- Clustering
    clu_base = os.path.join(models_dir, "threat_actor_profiler")  # PyCaret adds .pkl
    map_path = os.path.join(models_dir, "threat_actor_profile_map.json")
    features_csv = os.path.join(data_dir, "phishing_features_only.csv")
    clustered_csv = os.path.join(data_dir, "clustered_assignments.csv")

    # Features only
    X = data.drop(columns=["label"])
    X.to_csv(features_csv, index=False)

    if not os.path.exists(clu_base + ".pkl"):
        print("=== [Clustering] PyCaret setup ===")
        u_setup(data=X, session_id=42, normalize=True, verbose=False)

        print("=== [Clustering] Create K-Means (k=3) ===")
        km3 = u_create_model('kmeans', num_clusters=3, init='k-means++', random_state=42)

        print("=== [Clustering] Assign clusters and save ===")
        clustered = u_assign_model(km3)  # adds a 'Cluster' column


        clustered["ClusterId"] = (
            clustered["Cluster"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
        )

        clustered.to_csv(clustered_csv, index=False)
        print(f"[OK] Cluster assignments saved to {clustered_csv}")

        # ---- Derive cluster → actor mapping using numeric ClusterId
        means = clustered.groupby("ClusterId").mean(numeric_only=True)
        mapping = {}
        for cid, row in means.iterrows():
            crime_score = (
                0.5 * row.get("Shortining_Service", 0) +
                0.3 * row.get("Abnormal_URL", 0) +
                0.2 * row.get("having_IP_Address", 0)
            )
            state_score = (
                0.5 * row.get("SSLfinal_State", 0) +
                0.3 * row.get("Prefix_Suffix", 0) -
                0.2 * row.get("Shortining_Service", 0)
            )
            hacktivist_score = (
                0.5 * row.get("has_political_keywork", 0) +
                0.3 * row.get("Links_in_tags", 0) +
                0.2 * row.get("URL_of_Anchor", 0)
            )
            label = max(
                [("Organized Cybercrime", crime_score),
                ("State-Linked Actor", state_score),
                ("Hacktivist", hacktivist_score)],
                key=lambda kv: kv[1]
            )[0]
            mapping[int(cid)] = label  

        with open(map_path, "w") as f:
            json.dump(mapping, f)
        print(f"[OK] Saved cluster→actor map → {map_path}")
               

        print("Saving clusterer…")
        u_save_model(km3, clu_base)
        print("[OK] Clusterer saved.")
    else:
        print("[Skip] Clusterer already present.")
        if not os.path.exists(map_path):
            print("[Warn] cluster→actor map missing; re-run clustering or rebuild map from assignments.")

    print("\nAll done.")


if __name__ == "__main__":
    train()
