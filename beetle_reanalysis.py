"""
Beetle-fragment re-analysis (run in Google Colab:  %run beetle_reanalysis.py)

Does five things, saving everything to OUT_DIR:
  A. Seven-classifier comparison under three CV protocols
       random     : StratifiedKFold (what the manuscript currently reports)
       group_name : StratifiedGroupKFold, groups = ImageName (all rows sharing a file
                    number are kept in the same fold; most conservative proxy)
       group_sp   : StratifiedGroupKFold, groups = Species + ImageName
                    (same file number within a species, e.g. mw/1.jpg for every fragment)
  B. Fragment-specific Extra Trees under random vs grouped CV
  C. Cross-fragment (leave-one-fragment-out) with ET, RF, SVM, kNN and chance baselines
  D. Can the descriptors predict the FRAGMENT? (evidence they encode fragment identity)
  E. Feature importance (impurity + permutation, 5-fold, 30 repeats) -> Table 7 / Figure 6
  F. Figure 2: descriptor distributions by species
Set QUICK=1 in the environment for a fast smoke test.
"""
import os, sys, json, platform, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import xgboost, catboost

warnings.filterwarnings("ignore")
QUICK = os.environ.get("QUICK") == "1"
OUT_DIR = os.environ.get("OUT_DIR", "/content/reanalysis_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
FEATURES = ["Skewness", "Kurtosis", "Entropy", "StandardDeviation"]
URLS = {"Curculionidae": "https://ndownloader.figshare.com/files/54647816",
        "Tenebrionidae": "https://ndownloader.figshare.com/files/54647813"}
LOCAL = {"Curculionidae": os.environ.get("CUR_CSV"), "Tenebrionidae": os.environ.get("TEN_CSV")}
N_TREES = 50 if QUICK else 400
N_REPEATS = 3 if QUICK else 30

def load(family):
    df = pd.read_csv(LOCAL[family] or URLS[family])
    df = df.rename(columns={"StdDev": "StandardDeviation", "Image Name": "ImageName",
                            "Standard Deviation": "StandardDeviation"})
    assert df[FEATURES + ["Species", "Fragment", "ImageName"]].notna().all().all()
    return df.reset_index(drop=True)

def factories():
    """Same settings as the executed notebook (scaling is inside the pipeline)."""
    return {
        "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=N_TREES, max_features="sqrt", random_state=SEED, n_jobs=-1),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=N_TREES, max_features="sqrt", random_state=SEED, n_jobs=-1),
        "SVM-RBF": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
        "k-NN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "CatBoost": lambda: CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05, loss_function="MultiClass",
                                               verbose=False, random_seed=SEED, thread_count=2, allow_writing_files=False),
        "Hist gradient boosting": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                                                         l2_regularization=1.0, random_state=SEED),
        "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                         objective="multi:softprob", eval_metric="mlogloss", random_state=SEED, n_jobs=2),
    }

def get_splits(protocol, df, y):
    X = df[FEATURES]
    if protocol == "random":
        return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    groups = df["ImageName"] if protocol == "group_name" else df["Species"].astype(str) + "_" + df["ImageName"].astype(str)
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))

def cv_scores(make, df, y, splits):
    X = df[FEATURES].to_numpy(float)
    acc, bal, f1 = [], [], []
    for tr, te in splits:
        m = make(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
        acc.append(accuracy_score(y[te], p)); bal.append(balanced_accuracy_score(y[te], p))
        f1.append(f1_score(y[te], p, average="macro"))
    sd = lambda a: float(np.std(a, ddof=1))
    return dict(Accuracy=np.mean(acc) * 100, Accuracy_SD=sd(acc) * 100, BalancedAcc=np.mean(bal) * 100,
                MacroF1=np.mean(f1) * 100, MacroF1_SD=sd(f1) * 100)

data = {f: load(f) for f in URLS}
PROTOCOLS = ["random", "group_name", "group_sp"]

# ---------------- audit ----------------
audit = []
for fam, df in data.items():
    audit.append(dict(Family=fam, Rows=len(df), UniqueImageNames=df.ImageName.nunique(),
                      DuplicateKeys_Species_Fragment_Image=int(df.duplicated(["Species", "Fragment", "ImageName"]).sum()),
                      ExactDuplicateFeatureRows=int(df.duplicated(FEATURES + ["Species", "Fragment"]).sum())))
pd.DataFrame(audit).to_csv(f"{OUT_DIR}/A0_audit.csv", index=False); print(pd.DataFrame(audit).to_string(index=False))

# ---------------- A. model comparison ----------------
rows = []
for fam, df in data.items():
    y = LabelEncoder().fit_transform(df["Species"].astype(str))
    for proto in PROTOCOLS:
        splits = get_splits(proto, df, y)
        for name, make in factories().items():
            r = cv_scores(make, df, y, splits); r.update(Family=fam, Protocol=proto, Model=name); rows.append(r)
            print(f"A | {fam[:5]} | {proto:10s} | {name:22s} acc {r['Accuracy']:.2f} ± {r['Accuracy_SD']:.2f}  F1 {r['MacroF1']:.2f}")
A = pd.DataFrame(rows); A.to_csv(f"{OUT_DIR}/A_model_comparison.csv", index=False)
print("\nACCURACY (%) by protocol\n", A.pivot_table(index=["Family", "Model"], columns="Protocol", values="Accuracy").round(2))

# ---------------- B. fragment-specific Extra Trees ----------------
rows = []
for fam, df in data.items():
    for frag, d in df.groupby("Fragment"):
        d = d.reset_index(drop=True); y = LabelEncoder().fit_transform(d["Species"].astype(str))
        for proto in ["random", "group_name"]:
            r = cv_scores(factories()["Extra Trees"], d, y, get_splits(proto, d, y))
            r.update(Family=fam, Fragment=frag, Protocol=proto, N=len(d)); rows.append(r)
B = pd.DataFrame(rows); B.to_csv(f"{OUT_DIR}/B_fragment_specific_ET.csv", index=False)
print("\nB | fragment-specific ET macro-F1 (%)\n", B.pivot_table(index=["Family", "Fragment"], columns="Protocol", values="MacroF1").round(2))

# ---------------- C. cross-fragment ----------------
def xfrag_models():
    return {"Extra Trees": factories()["Extra Trees"],
            "Random Forest": lambda: RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced", random_state=SEED, n_jobs=-1),
            "SVM-RBF": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
            "k-NN": factories()["k-NN"],
            "Chance (stratified guess)": lambda: DummyClassifier(strategy="stratified", random_state=SEED),
            "Majority class": lambda: DummyClassifier(strategy="prior")}
rows = []
for fam, df in data.items():
    for held in sorted(df.Fragment.unique()):
        tr, te = df[df.Fragment != held], df[df.Fragment == held]
        for variant in ["raw", "per-fragment z-score"]:
            Xtr, Xte = tr[FEATURES].copy(), te[FEATURES].copy()
            if variant != "raw":   # each fragment standardised with its own mean/SD (unsupervised)
                for f_, sub in tr.groupby("Fragment"):
                    Xtr.loc[sub.index] = (sub[FEATURES] - sub[FEATURES].mean()) / sub[FEATURES].std(ddof=0)
                Xte = (Xte - Xte.mean()) / Xte.std(ddof=0)
            for name, make in xfrag_models().items():
                m = make(); m.fit(Xtr.to_numpy(float), tr["Species"]); p = m.predict(Xte.to_numpy(float))
                rows.append(dict(Family=fam, HeldOut=held, Variant=variant, Model=name,
                                 Accuracy=accuracy_score(te.Species, p) * 100, BalancedAcc=balanced_accuracy_score(te.Species, p) * 100,
                                 MacroF1=f1_score(te.Species, p, average="macro") * 100))
C = pd.DataFrame(rows); C.to_csv(f"{OUT_DIR}/C_cross_fragment.csv", index=False)
print("\nC | cross-fragment macro-F1 (%), mean over held-out fragments\n",
      C.pivot_table(index=["Family", "Model"], columns="Variant", values="MacroF1").round(1))

# ---------------- D. predict the fragment ----------------
rows = []
for fam, df in data.items():
    y = LabelEncoder().fit_transform(df["Fragment"].astype(str))
    for proto in ["random", "group_name"]:
        splits = get_splits(proto, df, y)
        for name in ["Extra Trees", "k-NN", "SVM-RBF"]:
            r = cv_scores(factories()[name], df, y, splits); r.update(Family=fam, Protocol=proto, Model=name); rows.append(r)
D = pd.DataFrame(rows); D.to_csv(f"{OUT_DIR}/D_predict_fragment.csv", index=False)
print("\nD | fragment prediction accuracy (%)\n", D.pivot_table(index=["Family", "Model"], columns="Protocol", values="Accuracy").round(2))

# ---------------- E. feature importance (5-fold) ----------------
rows = []
for fam, df in data.items():
    y = LabelEncoder().fit_transform(df["Species"].astype(str)); X = df[FEATURES].to_numpy(float)
    for proto in ["random", "group_name"]:
        imp_fold, perm_fold, perm_sd_fold = [], [], []
        for tr, te in get_splits(proto, df, y):
            m = ExtraTreesClassifier(n_estimators=N_TREES, max_features="sqrt", random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
            pi = permutation_importance(m, X[te], y[te], scoring="f1_macro", n_repeats=N_REPEATS, random_state=SEED, n_jobs=1)
            imp_fold.append(m.feature_importances_); perm_fold.append(pi.importances_mean); perm_sd_fold.append(pi.importances_std)
        imp_fold, perm_fold = np.array(imp_fold), np.array(perm_fold)
        for i, f in enumerate(FEATURES):
            rows.append(dict(Family=fam, Protocol=proto, Feature=f, Impurity=imp_fold[:, i].mean(), Impurity_SD_folds=imp_fold[:, i].std(ddof=1),
                             Permutation=perm_fold[:, i].mean(), Permutation_SD_folds=perm_fold[:, i].std(ddof=1)))
E = pd.DataFrame(rows); E.to_csv(f"{OUT_DIR}/E_feature_importance.csv", index=False)
print("\nE | feature importance\n", E.round(4).to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for j, fam in enumerate(data):
    e = E[(E.Family == fam) & (E.Protocol == "random")].sort_values("Permutation")
    axes[0, j].barh(e.Feature, e.Impurity, xerr=e.Impurity_SD_folds, capsize=3); axes[0, j].set_title(f"{fam}: impurity importance")
    axes[1, j].barh(e.Feature, e.Permutation, xerr=e.Permutation_SD_folds, capsize=3); axes[1, j].set_title(f"{fam}: permutation importance (macro-F1 decrease)")
fig.suptitle("Feature importance, Extra Trees, mean ± SD across five folds"); fig.tight_layout()
fig.savefig(f"{OUT_DIR}/Figure6_feature_importance.png", dpi=300); plt.close(fig)

# ---------------- F. Figure 2 ----------------
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for i, (fam, df) in enumerate(data.items()):
    sp = sorted(df.Species.unique())
    for j, f in enumerate(FEATURES):
        axes[i, j].boxplot([df.loc[df.Species == s, f] for s in sp], showfliers=False)
        axes[i, j].set_xticks(range(1, len(sp) + 1)); axes[i, j].set_xticklabels(sp)
        axes[i, j].set_title(f"{fam}: {f}", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/Figure2_descriptor_distributions.png", dpi=300); plt.close(fig)

# ---------------- versions ----------------
ver = dict(python=sys.version, platform=platform.platform(), sklearn=sklearn.__version__, numpy=np.__version__,
           pandas=pd.__version__, matplotlib=matplotlib.__version__, xgboost=xgboost.__version__, catboost=catboost.__version__)
json.dump(ver, open(f"{OUT_DIR}/versions.json", "w"), indent=2); print("\n", json.dumps(ver, indent=2))
print("\nDone. Outputs in", OUT_DIR)
