"""
Generate the final PDF report for the Wine Dataset ML project.
Uses reportlab Platypus for a multi-page, plot-embedded report.
"""

import json
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, HRFlowable
)
from reportlab.lib import colors

BASE = os.path.dirname(__file__)
PLOTS = os.path.join(BASE, "..", "plots")
DATA = os.path.join(BASE, "..", "data")
OUT = os.path.join(BASE, "..", "Wine_Dataset_ML_Report.pdf")

with open(os.path.join(DATA, "results_summary.json")) as f:
    R = json.load(f)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=20))
styles.add(ParagraphStyle("SubTitle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=11, textColor=colors.grey))
styles.add(ParagraphStyle("SectionHeader", parent=styles["Heading1"], fontSize=14, spaceBefore=14, spaceAfter=6,
                           textColor=colors.HexColor("#1f3864")))
styles.add(ParagraphStyle("SubHeader", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4,
                           textColor=colors.HexColor("#2e5395")))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, alignment=TA_JUSTIFY, leading=14, spaceAfter=6))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.5, alignment=TA_CENTER,
                           textColor=colors.grey, spaceAfter=10))
styles.add(ParagraphStyle("BulletItem", parent=styles["Normal"], fontSize=10, leftIndent=18, leading=14, spaceAfter=3))

doc = SimpleDocTemplate(OUT, pagesize=A4,
                         topMargin=1.5*cm, bottomMargin=1.5*cm,
                         leftMargin=1.8*cm, rightMargin=1.8*cm)

story = []

def img(name, width=15*cm):
    path = os.path.join(PLOTS, name)
    im = Image(path)
    aspect = im.imageHeight / im.imageWidth
    im.drawWidth = width
    im.drawHeight = width * aspect
    return im

def bullet(text):
    return Paragraph(f"&bull;&nbsp;&nbsp;{text}", styles["BulletItem"])


# =====================================================================
# TITLE PAGE
# =====================================================================
story.append(Spacer(1, 2*cm))
story.append(Paragraph("Statistical Analysis, Classification and Clustering<br/>on the UCI Wine Recognition Dataset", styles["TitleCenter"]))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("Data Science Project Report", styles["SubTitle"]))
story.append(Spacer(1, 1.5*cm))
story.append(HRFlowable(width="100%", color=colors.HexColor("#1f3864"), thickness=1))
story.append(Spacer(1, 0.5*cm))

meta_table = Table([
    ["Dataset:", R["dataset"]["name"]],
    ["Source:", "UCI Machine Learning Repository (archive.ics.uci.edu/ml/datasets/Wine)"],
    ["Samples:", str(R["dataset"]["n_samples"])],
    ["Features:", str(R["dataset"]["n_features"]) + " (all numerical)"],
    ["Classes:", ", ".join(R["dataset"]["classes"]) + " (3 wine cultivars)"],
    ["Tools Used:", "Python, pandas, NumPy, scikit-learn, matplotlib, seaborn"],
], colWidths=[3.5*cm, 11.5*cm])
meta_table.setStyle(TableStyle([
    ("FONTSIZE", (0,0), (-1,-1), 10),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 6),
]))
story.append(meta_table)
story.append(Spacer(1, 0.5*cm))
story.append(HRFlowable(width="100%", color=colors.HexColor("#1f3864"), thickness=1))
story.append(PageBreak())


# =====================================================================
# SECTION 1: DATASET DESCRIPTION
# =====================================================================
story.append(Paragraph("1. Dataset Description", styles["SectionHeader"]))

story.append(Paragraph(
    "The dataset used for this project is the <b>Wine Recognition Dataset</b> "
    "from the UCI Machine Learning Repository. It is the result of a chemical "
    "analysis of wines grown in the same region of Italy but derived from three "
    "different cultivars (grape varieties). The analysis determined the quantities "
    "of 13 chemical constituents found in each of the three types of wine.",
    styles["Body"]))

story.append(Paragraph(
    f"The dataset contains <b>{R['dataset']['n_samples']} rows (samples)</b> and "
    f"<b>{R['dataset']['n_features']} numerical feature columns</b>, plus one target "
    f"column denoting the class (cultivar) of each wine sample. All 13 features are "
    f"continuous numeric measurements, which makes this dataset directly suitable for "
    f"statistical analysis, classification, and clustering without any categorical "
    f"encoding.",
    styles["Body"]))

story.append(Paragraph("1.1 Feature Columns", styles["SubHeader"]))
feature_desc = [
    ("alcohol", "Alcohol content (%)"),
    ("malic_acid", "Malic acid concentration"),
    ("ash", "Ash content"),
    ("alcalinity_of_ash", "Alkalinity of ash"),
    ("magnesium", "Magnesium content (mg)"),
    ("total_phenols", "Total phenolic compounds"),
    ("flavanoids", "Flavanoid content"),
    ("nonflavanoid_phenols", "Non-flavanoid phenols"),
    ("proanthocyanins", "Proanthocyanin content"),
    ("color_intensity", "Colour intensity"),
    ("hue", "Hue of the wine"),
    ("od280/od315_of_diluted_wines", "OD280/OD315 ratio (protein content proxy)"),
    ("proline", "Proline amino acid content"),
]
feat_table_data = [["#", "Feature", "Description"]]
for i, (f, d) in enumerate(feature_desc, 1):
    feat_table_data.append([str(i), f, d])

feat_table = Table(feat_table_data, colWidths=[1*cm, 5*cm, 9*cm])
feat_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f3864")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("GRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f5fa")]),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(feat_table)

story.append(Paragraph("1.2 Class Labels", styles["SubHeader"]))
class_table_data = [["Class", "Cultivar Name", "Number of Samples", "Proportion"]]
total = R["dataset"]["n_samples"]
for cname, ccount in R["dataset"]["class_counts"].items():
    class_table_data.append([cname, cname.replace("_", " ").title(), str(ccount), f"{ccount/total*100:.1f}%"])
class_table = Table(class_table_data, colWidths=[3*cm, 4*cm, 4*cm, 4*cm])
class_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f3864")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f5fa")]),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(class_table)
story.append(Spacer(1, 0.2*cm))
story.append(img("03_class_pie.png", width=8*cm))
story.append(Paragraph("Figure 1.1: Proportion of samples belonging to each cultivar class.", styles["Caption"]))
story.append(img("02_class_distribution.png", width=10*cm))
story.append(Paragraph("Figure 1.2: Count of samples per class. The dataset is reasonably balanced "
                        "across the three classes, which avoids the need for resampling techniques.", styles["Caption"]))

story.append(PageBreak())


# =====================================================================
# SECTION 2: DATA PREPROCESSING
# =====================================================================
story.append(Paragraph("2. Data Preprocessing", styles["SectionHeader"]))

story.append(Paragraph("2.1 Why Preprocessing Is Needed", styles["SubHeader"]))
story.append(Paragraph(
    "Before applying any statistical or machine learning technique, the raw data "
    "was examined and prepared. The following preprocessing steps were carried out, "
    "along with the reasoning behind each:",
    styles["Body"]))

story.append(bullet(
    "<b>Missing value check:</b> The dataset was checked for null/missing entries "
    "across all 13 feature columns. No missing values were found, so no imputation "
    "was required."))
story.append(bullet(
    "<b>Type verification:</b> All 13 feature columns were confirmed to be numeric "
    "(float/int), satisfying the requirement that the data used for this analysis "
    "be numerical."))
story.append(bullet(
    "<b>Feature scaling (Standardization):</b> The features are on very different "
    "scales &ndash; for example, <i>proline</i> ranges from 278 to 1680, while "
    "<i>hue</i> ranges from 0.48 to 1.71. Distance-based algorithms such as kNN, "
    "k-Means, DBSCAN, and SVM (with an RBF kernel) are sensitive to feature scale: "
    "a feature with a larger numeric range would dominate the distance calculation "
    "and bias the model. To address this, all features were standardized using "
    "<b>StandardScaler</b>, transforming each feature to have mean&nbsp;0 and "
    "standard deviation&nbsp;1."))
story.append(bullet(
    "<b>Train/Test split:</b> For the classification task, the data was split into "
    "70% training and 30% testing sets using stratified sampling, ensuring that "
    "the class proportions in both sets match the original dataset. This gives a "
    "fair, unbiased estimate of how well each classifier generalizes to unseen data."))

story.append(Paragraph("2.2 Effect of Standardization", styles["SubHeader"]))
story.append(Paragraph(
    "Before standardization, features like <i>proline</i> (mean &asymp; 747, std "
    "&asymp; 315) and <i>magnesium</i> (mean &asymp; 100, std &asymp; 14) had numeric "
    "ranges orders of magnitude larger than features like <i>hue</i> (mean &asymp; "
    "0.96, std &asymp; 0.23) or <i>nonflavanoid_phenols</i> (mean &asymp; 0.36). "
    "After standardization, every feature is on a common scale (mean 0, std 1), "
    "ensuring each of the 13 chemical measurements contributes proportionally to "
    "distance-based computations rather than being dominated by features measured "
    "in larger units.",
    styles["Body"]))

story.append(Paragraph("2.3 Descriptive Statistics (Before Scaling)", styles["SubHeader"]))
story.append(Paragraph(
    "The table below summarizes the mean, standard deviation, minimum and maximum "
    "values for each feature in the original (unscaled) dataset.",
    styles["Body"]))

# Descriptive stats table (selected columns to fit page)
import pandas as pd
desc = pd.read_csv(os.path.join(DATA, "descriptive_stats.csv"), index_col=0)
desc = desc.drop(index="target")
desc_table_data = [["Feature", "Mean", "Std Dev", "Min", "Max"]]
for feat in desc.index:
    row = desc.loc[feat]
    desc_table_data.append([feat, f"{row['mean']:.2f}", f"{row['std']:.2f}", f"{row['min']:.2f}", f"{row['max']:.2f}"])

desc_table = Table(desc_table_data, colWidths=[6*cm, 2.6*cm, 2.6*cm, 2.4*cm, 2.4*cm])
desc_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f3864")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("GRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f5fa")]),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
story.append(desc_table)

story.append(PageBreak())


# =====================================================================
# SECTION 3: TASK A - PRELIMINARY STATISTICAL ANALYSIS
# =====================================================================
story.append(Paragraph("3. Task A &mdash; Preliminary Statistical Analysis & Inferences", styles["SectionHeader"]))

story.append(Paragraph("3.1 Correlation Analysis", styles["SubHeader"]))
story.append(Paragraph(
    "A Pearson correlation matrix was computed across all 13 features to identify "
    "linear relationships between chemical properties. The heatmap below visualizes "
    "these correlations, with dark red indicating strong positive correlation and "
    "dark blue indicating strong negative correlation.",
    styles["Body"]))
story.append(img("01_correlation_heatmap.png", width=14*cm))
story.append(Paragraph("Figure 3.1: Correlation heatmap of all 13 chemical features.", styles["Caption"]))

story.append(Paragraph("<b>Inferences from the correlation analysis:</b>", styles["Body"]))
story.append(bullet(
    "<b>total_phenols and flavanoids</b> show a very strong positive correlation "
    "(r &asymp; 0.86). This makes chemical sense &ndash; flavanoids are a sub-class "
    "of phenolic compounds, so wines high in total phenols also tend to be high in "
    "flavanoids."))
story.append(bullet(
    "<b>flavanoids and od280/od315_of_diluted_wines</b> are also strongly correlated "
    "(r &asymp; 0.79), suggesting that the optical density measurement is heavily "
    "influenced by flavanoid content."))
story.append(bullet(
    "<b>alcohol and proline</b> show a moderate positive correlation (r &asymp; 0.64), "
    "indicating wines with higher alcohol content tend to also have higher proline "
    "levels &ndash; both of which are useful discriminating features between the "
    "three cultivars, as seen in Section 3.2."))
story.append(bullet(
    "<b>malic_acid and hue</b> show a moderate negative correlation (r &asymp; -0.56), "
    "meaning wines with higher malic acid content tend to have a lower hue value."))
story.append(bullet(
    "<b>flavanoids and nonflavanoid_phenols</b> are negatively correlated (r &asymp; "
    "-0.54), which is expected since these represent two competing categories of "
    "the same broader phenol family."))

story.append(Paragraph("3.2 Feature Distributions by Class", styles["SubHeader"]))
story.append(Paragraph(
    "Boxplots of four key chemical features were generated, grouped by wine class, "
    "to identify which features best separate the three cultivars.",
    styles["Body"]))
story.append(img("04_boxplots_by_class.png", width=14*cm))
story.append(Paragraph("Figure 3.2: Boxplots of alcohol, flavanoids, color_intensity and proline by class.", styles["Caption"]))

story.append(Paragraph("<b>Inferences:</b>", styles["Body"]))
story.append(bullet(
    "<b>Flavanoids</b> show the clearest class separation: class_0 and class_1 wines "
    "have noticeably higher flavanoid content than class_2, with relatively little "
    "overlap. This makes flavanoids a strong candidate feature for classification."))
story.append(bullet(
    "<b>Proline</b> is markedly higher for class_0 wines compared to class_1 and "
    "class_2, making it another strong discriminating feature."))
story.append(bullet(
    "<b>Color intensity</b> is highest for class_2 wines and lowest for class_1, "
    "providing a third useful axis of separation."))
story.append(bullet(
    "<b>Alcohol</b> shows class_0 generally having the highest alcohol content, "
    "though with more overlap between classes than the other three features."))
story.append(bullet(
    "Overall, these distributions explain why a classifier such as SVM achieves "
    "very high accuracy on this dataset (see Section 4) &ndash; the three classes "
    "are well-separated in the 13-dimensional feature space, even though no single "
    "feature perfectly separates all three on its own."))

story.append(Paragraph("3.3 Feature Distributions (Histograms)", styles["SubHeader"]))
story.append(Paragraph(
    "Histograms of all 13 features (across the entire dataset, all classes combined) "
    "were plotted to understand the overall shape of each feature's distribution.",
    styles["Body"]))
story.append(img("05_feature_histograms.png", width=15*cm))
story.append(Paragraph("Figure 3.3: Histograms of all 13 features.", styles["Caption"]))

story.append(Paragraph("<b>Inferences:</b>", styles["Body"]))
story.append(bullet(
    "Most features (alcohol, ash, alcalinity_of_ash, total_phenols, hue) approximate "
    "a roughly normal (bell-shaped) distribution, which is expected for natural "
    "chemical measurements across a sample population."))
story.append(bullet(
    "<b>Proline</b> and <b>color_intensity</b> are right-skewed, with a long tail "
    "toward higher values &ndash; a small number of wines have unusually high proline "
    "or colour intensity compared to the bulk of the samples."))
story.append(bullet(
    "<b>Malic_acid</b> is also right-skewed, with most wines clustered at lower "
    "values and a tail extending toward higher acid content."))
story.append(bullet(
    "<b>Flavanoids</b> shows a roughly bimodal pattern, which is consistent with the "
    "boxplot finding that flavanoid content differs systematically between the "
    "wine classes &ndash; the two 'modes' likely correspond to the "
    "class_0/class_1 group (high flavanoids) versus class_2 (low flavanoids)."))

story.append(PageBreak())


# =====================================================================
# SECTION 4: TASK B - CLASSIFICATION
# =====================================================================
story.append(Paragraph("4. Task B &mdash; Classification (SVM and kNN)", styles["SectionHeader"]))

story.append(Paragraph("4.1 Purpose and Methodology", styles["SubHeader"]))
story.append(Paragraph(
    "The goal of this task is to predict the wine cultivar (class_0, class_1, or "
    "class_2) from its 13 chemical measurements, and to compare two classic "
    "classification algorithms: <b>Support Vector Machine (SVM)</b> with an RBF "
    "kernel, and <b>k-Nearest Neighbours (kNN)</b> with k=5. Both algorithms were "
    "trained on the standardized features using a 70/30 stratified train-test split, "
    "and evaluated using three metrics:",
    styles["Body"]))
story.append(bullet("<b>Accuracy</b> &ndash; the proportion of test samples correctly classified."))
story.append(bullet(
    "<b>Cohen's Kappa</b> &ndash; a metric that compares observed accuracy to the "
    "accuracy expected by chance, correcting for class imbalance. A kappa close to "
    "1 indicates the classifier's predictions agree with the true labels far more "
    "than random chance would suggest."))
story.append(bullet(
    "<b>ROC-AUC (one-vs-rest)</b> &ndash; measures how well the classifier separates "
    "each class from the others across all probability thresholds, averaged over "
    "the three classes. A value of 1.0 indicates perfect separation."))

story.append(Paragraph("4.2 Results Comparison", styles["SubHeader"]))
results_table_data = [
    ["Metric", "SVM (RBF kernel)", "kNN (k=5)"],
    ["Accuracy", f"{R['classification']['SVM']['accuracy']:.4f}", f"{R['classification']['kNN']['accuracy']:.4f}"],
    ["Cohen's Kappa", f"{R['classification']['SVM']['kappa']:.4f}", f"{R['classification']['kNN']['kappa']:.4f}"],
    ["ROC-AUC", f"{R['classification']['SVM']['roc_auc']:.4f}", f"{R['classification']['kNN']['roc_auc']:.4f}"],
]
results_table = Table(results_table_data, colWidths=[5*cm, 5*cm, 5*cm])
results_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f3864")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 10),
    ("GRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f5fa")]),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
]))
story.append(results_table)
story.append(Spacer(1, 0.3*cm))
story.append(img("06_classification_comparison.png", width=13*cm))
story.append(Paragraph("Figure 4.1: Bar chart comparing Accuracy, Kappa and ROC-AUC for SVM vs kNN.", styles["Caption"]))

story.append(Paragraph("4.3 Confusion Matrices", styles["SubHeader"]))
story.append(img("07_confusion_matrices.png", width=15*cm))
story.append(Paragraph("Figure 4.2: Confusion matrices on the 54-sample test set for SVM (left) and kNN (right).", styles["Caption"]))

story.append(Paragraph("4.4 ROC Curves", styles["SubHeader"]))
story.append(Paragraph(
    "One-vs-rest ROC curves were plotted for each class, for both classifiers. "
    "The Area Under the Curve (AUC) for each class is shown in the legend.",
    styles["Body"]))
story.append(img("08_roc_curves.png", width=15*cm))
story.append(Paragraph("Figure 4.3: ROC curves (one-vs-rest) for SVM (left) and kNN (right).", styles["Caption"]))

story.append(Paragraph("4.5 Discussion and Inferences", styles["SubHeader"]))
story.append(bullet(
    f"<b>SVM outperforms kNN on all three metrics</b>: accuracy "
    f"({R['classification']['SVM']['accuracy']:.3f} vs {R['classification']['kNN']['accuracy']:.3f}), "
    f"kappa ({R['classification']['SVM']['kappa']:.3f} vs {R['classification']['kNN']['kappa']:.3f}), "
    f"and ROC-AUC ({R['classification']['SVM']['roc_auc']:.3f} vs {R['classification']['kNN']['roc_auc']:.3f}). "
    "This is consistent with the dataset characteristics observed in Task A: the "
    "classes are well-separated but the separation involves a combination of "
    "multiple features rather than simple axis-aligned boundaries, which the "
    "RBF kernel's non-linear decision boundary captures more effectively than "
    "kNN's purely local, distance-based voting."))
story.append(bullet(
    "<b>SVM achieved a perfect ROC-AUC of 1.0</b>, meaning that for every pairwise "
    "one-vs-rest comparison, there exists some threshold at which the SVM perfectly "
    "separates that class from the rest &ndash; even though one sample was still "
    "misclassified in the confusion matrix (a class_2 sample predicted as class_1). "
    "This shows AUC and accuracy capture different things: AUC reflects the ranking "
    "quality of predicted probabilities across all thresholds, not just the single "
    "threshold (0.5) used to produce the confusion matrix."))
story.append(bullet(
    "<b>kNN's main confusion is between class_1 and class_2</b> (3 samples of class_1 "
    "misclassified as class_2). Referring back to the boxplots in Figure 3.2, "
    "class_1 and class_2 have the most overlapping flavanoid and color_intensity "
    "ranges among the three classes, which likely causes kNN's nearest-neighbour "
    "votes to occasionally favour the wrong class near the decision boundary."))
story.append(bullet(
    "<b>Cohen's Kappa above 0.9 for both models</b> confirms that both classifiers "
    "are performing far better than random guessing (which would give kappa &asymp; 0), "
    "and the gap between SVM's kappa (0.972) and kNN's (0.917) reflects SVM's more "
    "consistent performance specifically on the harder-to-separate class_1/class_2 "
    "boundary."))
story.append(bullet(
    "<b>Practical takeaway:</b> for this dataset, a kernel-based method that can "
    "model curved decision boundaries (SVM) is preferable to a purely distance-based "
    "method (kNN), though both achieve strong (&gt;90%) accuracy, confirming that "
    "the 13 chemical features collectively carry strong discriminative signal for "
    "the three cultivars."))

story.append(PageBreak())


# =====================================================================
# SECTION 5: TASK C - CLUSTERING
# =====================================================================
story.append(Paragraph("5. Task C &mdash; Clustering (k-Means and DBSCAN)", styles["SectionHeader"]))

story.append(Paragraph("5.1 Purpose and Methodology", styles["SubHeader"]))
story.append(Paragraph(
    "Unlike Task B, clustering is <b>unsupervised</b>: the true class labels are "
    "not used to fit the model. The goal is to test whether the natural groupings "
    "(clusters) discovered purely from the 13 chemical features correspond to the "
    "three known wine cultivars. Two algorithms were compared:",
    styles["Body"]))
story.append(bullet(
    "<b>k-Means (k=3)</b> &ndash; partitions the data into exactly 3 clusters by "
    "iteratively assigning points to the nearest centroid and recomputing centroids. "
    "k=3 was chosen to match the known number of classes, and validated using the "
    "elbow method (Figure 5.1)."))
story.append(bullet(
    "<b>DBSCAN</b> &ndash; a density-based algorithm that groups together points "
    "that are closely packed, marking points in low-density regions as <i>noise</i> "
    "(outliers). It does not require the number of clusters to be specified in "
    "advance. Parameters used: eps=2.2, min_samples=3, selected using the k-distance "
    "plot (Figure 5.2)."))

story.append(Paragraph("Two metrics were used to evaluate and compare the clusterings:", styles["Body"]))
story.append(bullet(
    "<b>SSE (Sum of Squared Errors / Inertia)</b> &ndash; the total squared distance "
    "of each point from its cluster centre. Lower SSE means tighter, more compact "
    "clusters."))
story.append(bullet(
    "<b>Entropy</b> &ndash; for each cluster, the entropy of the true class labels "
    "within that cluster is computed (in bits), then averaged across clusters "
    "weighted by cluster size. Lower entropy means each cluster predominantly "
    "contains samples from a single true class (i.e., the unsupervised clusters "
    "align well with the real cultivars)."))

story.append(Paragraph("5.2 Choosing k for k-Means: Elbow Method", styles["SubHeader"]))
story.append(img("11_elbow_method.png", width=11*cm))
story.append(Paragraph("Figure 5.1: SSE vs number of clusters (k). The 'elbow' near k=3 "
                        "supports using 3 clusters, matching the 3 known wine classes.", styles["Caption"]))

story.append(Paragraph("5.3 Choosing eps for DBSCAN: k-Distance Plot", styles["SubHeader"]))
story.append(img("08b_kdistance_plot.png", width=11*cm))
story.append(Paragraph("Figure 5.2: 3rd-nearest-neighbour distance for each point, sorted ascending. "
                        "The 'knee' of this curve guides the choice of eps; eps=2.2 was chosen "
                        "to balance cluster count against noise points.", styles["Caption"]))

story.append(PageBreak())

story.append(Paragraph("5.4 Results Comparison", styles["SubHeader"]))
clust_table_data = [
    ["Metric", "k-Means (k=3)", "DBSCAN (eps=2.2, min_samples=3)"],
    ["SSE", f"{R['clustering']['KMeans']['sse']:.2f}", f"{R['clustering']['DBSCAN']['sse']:.2f}"],
    ["Entropy (bits)", f"{R['clustering']['KMeans']['entropy']:.4f}", f"{R['clustering']['DBSCAN']['entropy']:.4f}"],
    ["Silhouette Score", f"{R['clustering']['KMeans']['silhouette']:.4f}",
     f"{R['clustering']['DBSCAN']['silhouette']:.4f}" if R['clustering']['DBSCAN']['silhouette'] else "N/A"],
    ["Clusters Found", "3 (fixed)", str(R['clustering']['DBSCAN']['n_clusters'])],
    ["Noise Points", "0 (n/a)", str(R['clustering']['DBSCAN']['n_noise'])],
]
clust_table = Table(clust_table_data, colWidths=[4.5*cm, 4.5*cm, 6.5*cm])
clust_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f3864")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9.5),
    ("GRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f5fa")]),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
]))
story.append(clust_table)
story.append(Spacer(1, 0.3*cm))
story.append(img("09_clustering_comparison.png", width=14*cm))
story.append(Paragraph("Figure 5.3: SSE (left) and Entropy (right) for k-Means vs DBSCAN.", styles["Caption"]))

story.append(Paragraph("5.5 Visualizing the Clusters (PCA Projection)", styles["SubHeader"]))
story.append(Paragraph(
    "Since the data has 13 dimensions, Principal Component Analysis (PCA) was used "
    "to project it down to 2 dimensions (capturing the two directions of greatest "
    "variance) purely for visualization. The true class labels, k-Means clusters, "
    "and DBSCAN clusters are plotted side-by-side on the same 2D projection.",
    styles["Body"]))
story.append(img("10_pca_clusters.png", width=15.5*cm))
story.append(Paragraph("Figure 5.4: PCA projection (PC1 vs PC2) showing true class labels (left), "
                        "k-Means clusters (centre), and DBSCAN clusters with noise points in dark "
                        "blue (right).", styles["Caption"]))

story.append(PageBreak())

story.append(Paragraph("5.6 Discussion and Inferences", styles["SubHeader"]))
story.append(bullet(
    f"<b>k-Means produces purer clusters (lower entropy)</b>: with k=3, k-Means "
    f"achieved an entropy of only {R['clustering']['KMeans']['entropy']:.3f} bits, "
    f"meaning each k-Means cluster is dominated almost entirely by a single true "
    f"wine class. Comparing the left and centre panels of Figure 5.4, the k-Means "
    f"clusters visually overlay the true class regions very closely &ndash; this "
    f"confirms that the three cultivars form genuinely distinct, roughly "
    f"spherical/convex groups in the standardized feature space, which is exactly "
    f"the kind of structure k-Means is designed to find."))
story.append(bullet(
    f"<b>DBSCAN achieves lower SSE</b> ({R['clustering']['DBSCAN']['sse']:.1f} vs "
    f"{R['clustering']['KMeans']['sse']:.1f} for k-Means) <b>but much higher entropy</b> "
    f"({R['clustering']['DBSCAN']['entropy']:.3f} vs {R['clustering']['KMeans']['entropy']:.3f} "
    f"bits). The lower SSE is partly an artifact of DBSCAN excluding "
    f"{R['clustering']['DBSCAN']['n_noise']} boundary/outlier points as 'noise' "
    f"&ndash; removing these points naturally shrinks the within-cluster spread "
    f"of the remaining points. However, the much higher entropy shows that the "
    f"clusters DBSCAN <i>does</i> find mix samples from multiple true classes "
    f"more than k-Means does."))
story.append(bullet(
    f"<b>Why DBSCAN struggles here:</b> DBSCAN assumes clusters are separated by "
    f"regions of <i>low density</i>, and works best when clusters have roughly "
    f"uniform density and clear density 'gaps' between them. Looking at Figure 5.4 "
    f"(right panel), the three wine classes in PCA space form a connected, "
    f"continuously-varying region rather than being separated by sparse gaps "
    f"&ndash; especially class_0 and class_2, which sit close together. As a "
    f"result, DBSCAN with a single global eps value either merges multiple true "
    f"classes into one density-connected region, or splits a single true class "
    f"across multiple density-based clusters, depending on local density variation."))
story.append(bullet(
    f"<b>Trade-off between SSE and Entropy:</b> This comparison illustrates an "
    f"important general point &ndash; SSE alone is not a reliable indicator of "
    f"'good' clustering when ground-truth labels are available for validation. "
    f"DBSCAN's lower SSE looks favourable in isolation, but the entropy metric "
    f"(which uses the true labels) reveals that k-Means' clusters are far more "
    f"<i>meaningful</i> with respect to the actual wine cultivars."))
story.append(bullet(
    "<b>Practical takeaway:</b> for datasets where the true groups are convex, "
    "roughly equal-sized, and globular &ndash; as the three wine cultivars appear "
    "to be in this 13-dimensional standardized space &ndash; centroid-based methods "
    "like k-Means recover the true structure more faithfully than density-based "
    "methods like DBSCAN, which are better suited to clusters of irregular shape "
    "with clear density gaps between them."))

story.append(PageBreak())


# =====================================================================
# SECTION 6: CONCLUSION
# =====================================================================
story.append(Paragraph("6. Overall Conclusions", styles["SectionHeader"]))

story.append(bullet(
    "The Wine Recognition dataset (178 samples, 13 numeric chemical features, 3 "
    "balanced classes) required minimal cleaning &ndash; no missing values were "
    "present &ndash; but <b>standardization was essential</b> due to the widely "
    "differing scales of the 13 features."))
story.append(bullet(
    "Correlation analysis and class-wise boxplots showed that features such as "
    "<b>flavanoids</b>, <b>proline</b>, and <b>color_intensity</b> are strong "
    "natural discriminators between the three cultivars, explaining the high "
    "downstream classification accuracy."))
story.append(bullet(
    "For <b>classification</b>, <b>SVM (RBF kernel) outperformed kNN</b> on "
    "accuracy, Cohen's Kappa, and ROC-AUC, achieving near-perfect separation of "
    "the three classes (98.1% accuracy, kappa 0.972, AUC 1.000)."))
story.append(bullet(
    "For <b>clustering</b>, <b>k-Means (k=3) produced clusters that align much "
    "more closely with the true cultivars</b> (entropy 0.190 bits) compared to "
    "DBSCAN (entropy 0.806 bits), even though DBSCAN achieved a lower raw SSE by "
    "treating 44 points as noise. This highlights that no single metric (SSE) "
    "should be used in isolation to judge clustering quality."))
story.append(bullet(
    "Overall, the analysis confirms that the 13 chemical measurements collectively "
    "form a feature space in which the three wine cultivars are <b>well-separated "
    "and largely convex/globular</b> &ndash; a structure well-suited to both "
    "kernel-based classifiers (SVM) and centroid-based clustering (k-Means), "
    "but less suited to density-based clustering (DBSCAN) at a single global "
    "density threshold."))

story.append(Spacer(1, 1*cm))
story.append(HRFlowable(width="100%", color=colors.HexColor("#1f3864"), thickness=1))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "<i>All code, plots, and this report were generated using Python "
    "(pandas, NumPy, scikit-learn, matplotlib, seaborn, reportlab). "
    "The complete source code is available in the accompanying repository.</i>",
    styles["Caption"]))


doc.build(story)
print(f"Report generated: {OUT}")
print(f"Pages: approx. (built successfully)")
