"""
Delhi NCR Urban Change Intelligence — Phase 10 Dashboard (v2, professional).

Reads the aggregated Phase 9 spatial-intelligence artifacts and presents
them interactively with a satellite basemap.

DATA HONESTY NOTE (do not remove):
  - Predictions come from BaselineChangeCNN (Phase 7, WITH MC Dropout).
    Training labels were spectral-index-derived and weakly-supervised
    (see PROJECT_STATE.md) - not pixel-perfect ground truth.
  - full_scene_masks.npz contains ONLY a built-up boolean mask. No
    pixel-level change or uncertainty raster is available - those exist
    only as patch-aggregated statistics in the CSV/JSON. Not fabricated
    here as a substitute.
  - Test-set metrics (n=56 patches) are small-sample; treat with caution.
  - Vegetation "gain" (32.76% of total change) reflects agricultural
    crop-cycle phenology variability between the two Jan/Feb windows,
    NOT an ecological/afforestation finding.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dashboard" / "spatial"

CSV_PATH = DATA_DIR / "patch_spatial_metrics.csv"
SUMMARY_JSON = DATA_DIR / "full_scene_spatial_summary.json"
TEST_SANITY_JSON = DATA_DIR / "test_split_sanity_check.json"
PROXIMITY_JSON = DATA_DIR / "urban_proximity_summary.json"
HOTSPOT_PNG = DATA_DIR / "spatial_intelligence_hotspot_maps.png"
MASKS_NPZ = DATA_DIR / "full_scene_masks.npz"

CATEGORY_COLS = ["builtup_pixels", "veg_loss_pixels", "veg_gain_pixels", "other_pixels"]
CATEGORY_LABELS = {
    "builtup_pixels": "Built-up Expansion",
    "veg_loss_pixels": "Vegetation Loss",
    "veg_gain_pixels": "Vegetation Gain",
    "other_pixels": "Other / Uncertain",
}
CATEGORY_COLORS = {
    "Built-up Expansion": "#ef4444",
    "Vegetation Loss": "#a16207",
    "Vegetation Gain": "#22c55e",
    "Other / Uncertain": "#94a3b8",
}
BRAND_ACCENT = "#38bdf8"

st.set_page_config(
    page_title="Delhi NCR Urban Change Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theming (CSS)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .app-header {{
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(15,23,42,0.0));
        border: 1px solid rgba(148,163,184,0.25);
        margin-bottom: 1rem;
    }}
    .app-header h1 {{
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }}
    .app-header p {{
        margin: 0.35rem 0 0.75rem 0;
        color: #94a3b8;
        font-size: 0.95rem;
        max-width: 900px;
    }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(56,189,248,0.14);
        color: {BRAND_ACCENT};
        border: 1px solid rgba(56,189,248,0.35);
    }}

    div[data-testid="stMetric"] {{
        background: rgba(148,163,184,0.06);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 12px;
        padding: 0.9rem 1rem 0.6rem 1rem;
    }}
    div[data-testid="stMetricLabel"] {{ font-size: 0.8rem; color: #94a3b8; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 700; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }}

    .disclosure-box {{
        border-left: 3px solid {BRAND_ACCENT};
        background: rgba(56,189,248,0.06);
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #cbd5e1;
        margin: 0.6rem 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_patch_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["dominant_category"] = df[CATEGORY_COLS].idxmax(axis=1).map(CATEGORY_LABELS)

    # Derive real patch footprint (degrees) directly from the data itself,
    # rather than hardcoding an assumed bounding box / resolution.
    xs = np.sort(df["centroid_x"].unique())
    ys = np.sort(df["centroid_y"].unique())
    half_w = (np.median(np.diff(xs)) / 2) if len(xs) > 1 else 0.0115
    half_h = (np.median(np.diff(ys)) / 2) if len(ys) > 1 else 0.0115
    df["_half_w"] = half_w
    df["_half_h"] = half_h
    return df


@st.cache_data
def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_builtup_mask_thumbnail(step: int = 15) -> np.ndarray:
    with np.load(MASKS_NPZ) as npz:
        mask = npz["builtup_mask"]
    return mask[::step, ::step]


df = load_patch_csv()
summary = load_json(SUMMARY_JSON)
test_sanity = load_json(TEST_SANITY_JSON)
proximity = load_json(PROXIMITY_JSON)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
      <h1>🛰️ Delhi NCR Urban Change Intelligence</h1>
      <p>Satellite-observed land-cover change across Delhi NCR, comparing two fixed
      calendar windows. Purely technical / geospatial scope — no administrative or
      political framing.</p>
      <div class="badge-row">
        <span class="badge">Sentinel-2 L2A</span>
        <span class="badge">Jan–Feb 2022 → Jan–Feb 2026</span>
        <span class="badge">Google Earth Engine</span>
        <span class="badge">Siamese-baseline CNN</span>
        <span class="badge">MC Dropout Uncertainty (N=20)</span>
        <span class="badge">{summary['total_patches']} patches · {summary['total_valid_km2']:.0f} km² evaluated</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
split_options = ["All"] + sorted(df["split"].unique().tolist())
selected_split = st.sidebar.selectbox("Data split", split_options, index=0)

color_metric = st.sidebar.selectbox(
    "Map: color patches by",
    ["change_fraction", "builtup_fraction", "mean_uncertainty_all", "dominant_category"],
    index=0,
)

basemap_choice = st.sidebar.radio("Basemap", ["Satellite", "Light", "Dark"], index=0)

st.sidebar.markdown("---")

filtered_df = df if selected_split == "All" else df[df["split"] == selected_split]

st.sidebar.markdown(
    f"**Patches shown:** {len(filtered_df)} / {len(df)}\n\n"
    "Each rendered tile = one 256×256 px (~2.56 km) patch footprint, "
    "not a raw pixel."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_map, tab_uncertainty, tab_hotspots, tab_mask, tab_notes = st.tabs(
    [
        "📊 Overview",
        "🗺️ Interactive Map",
        "🎲 Uncertainty",
        "🔥 Hotspots & Proximity",
        "🏙️ Built-up Mask",
        "📋 Methodology & Limitations",
    ]
)

PLOTLY_TEMPLATE = "plotly_dark"

# --- Overview ---------------------------------------------------------------
with tab_overview:
    st.subheader("Full-Scene Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valid Area Evaluated", f"{summary['total_valid_km2']:.1f} km²")
    c2.metric(
        "Net Changed Area",
        f"{summary['total_changed_km2']:.1f} km²",
        f"{summary['overall_change_percentage']:.2f}% of valid area",
    )
    c3.metric("Total Patches", summary["total_patches"])
    c4.metric(
        "Mean Uncertainty (Changed px)",
        f"{summary['mean_uncertainty_change']:.4f}",
        f"vs {summary['mean_uncertainty_nochange']:.4f} (no-change)",
        delta_color="off",
    )

    st.markdown("#### Change Category Breakdown (Full Scene)")
    cat_rows = [
        {
            "category": key.replace("_", " ").title(),
            "area_km2": vals["area_km2"],
            "pct_of_total_change": vals["pct_of_total_change"],
            "pct_of_valid_area": vals["pct_of_valid_area"],
        }
        for key, vals in summary["categories"].items()
    ]
    cat_df = pd.DataFrame(cat_rows)
    color_map_full = {
        "Builtup Expansion": CATEGORY_COLORS["Built-up Expansion"],
        "Vegetation Loss": CATEGORY_COLORS["Vegetation Loss"],
        "Vegetation Gain": CATEGORY_COLORS["Vegetation Gain"],
        "Other Uncertain": CATEGORY_COLORS["Other / Uncertain"],
    }

    col_a, col_b = st.columns([1, 1])
    with col_a:
        fig_pie = px.pie(
            cat_df, names="category", values="area_km2",
            title="Share of Total Changed Area", color="category",
            color_discrete_map=color_map_full, hole=0.45,
            template=PLOTLY_TEMPLATE,
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        fig_bar = px.bar(
            cat_df.sort_values("area_km2"), x="area_km2", y="category",
            orientation="h", title="Changed Area by Category (km²)",
            text="area_km2", color="category", color_discrete_map=color_map_full,
            template=PLOTLY_TEMPLATE,
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        '<div class="disclosure-box">ℹ️ Vegetation Gain (32.76% of total change) '
        "reflects agricultural crop-cycle phenology differences between the two "
        "Jan/Feb windows, not an ecological or afforestation finding — see "
        "Methodology tab.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Test-Split Sanity Check (n = 56 patches)")
    st.caption("Small sample — directionally useful, not statistically strong on its own.")
    t1, t2, t3 = st.columns(3)
    t1.metric("Test Change Rate", f"{test_sanity['test_change_rate']*100:.2f}%")
    t2.metric("Test Built-up Rate", f"{test_sanity['test_builtup_rate']*100:.2f}%")
    t3.metric("Test Mean Uncertainty (change)", f"{test_sanity['mean_uncertainty_change']:.4f}")

# --- Interactive Map ---------------------------------------------------------
with tab_map:
    st.subheader("Patch-Level Change Map")
    st.caption(
        "Each tile is a real 256×256 px (~2.56 km) patch footprint, colored by "
        "the selected metric. This is a patch-aggregated view, not a pixel-level "
        "raster overlay — see Methodology tab for why."
    )

    import folium
    from branca.colormap import linear
    from folium.plugins import Fullscreen
    from streamlit_folium import st_folium

    center_lat = filtered_df["centroid_y"].mean()
    center_lon = filtered_df["centroid_x"].mean()

    tile_layers = {
        "Satellite": dict(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics",
        ),
        "Light": dict(tiles="CartoDB positron", attr="CartoDB"),
        "Dark": dict(tiles="CartoDB dark_matter", attr="CartoDB"),
    }
    active = tile_layers[basemap_choice]

    fmap = folium.Map(
        location=[center_lat, center_lon], zoom_start=10,
        tiles=active["tiles"], attr=active["attr"],
    )
    Fullscreen(position="topleft").add_to(fmap)

    half_w = float(filtered_df["_half_w"].iloc[0]) if len(filtered_df) else 0.0115
    half_h = float(filtered_df["_half_h"].iloc[0]) if len(filtered_df) else 0.0115

    def patch_bounds(row):
        return [
            [row["centroid_y"] - half_h, row["centroid_x"] - half_w],
            [row["centroid_y"] + half_h, row["centroid_x"] + half_w],
        ]

    if color_metric == "dominant_category":
        for _, row in filtered_df.iterrows():
            color = CATEGORY_COLORS.get(row["dominant_category"], "#333333")
            folium.Rectangle(
                bounds=patch_bounds(row),
                color=color, weight=0.6, fill=True, fill_color=color, fill_opacity=0.55,
                popup=folium.Popup(
                    f"<b>Patch {int(row['patch_index'])}</b> ({row['split']})<br>"
                    f"Dominant: {row['dominant_category']}<br>"
                    f"Change fraction: {row['change_fraction']:.3f}<br>"
                    f"Built-up fraction: {row['builtup_fraction']:.3f}<br>"
                    f"Mean uncertainty: {row['mean_uncertainty_all']:.4f}",
                    max_width=260,
                ),
            ).add_to(fmap)
        legend_items = "".join(
            f"<div style='margin:2px 0'><span style='background:{c};display:inline-block;"
            f"width:11px;height:11px;margin-right:6px;border-radius:2px;'></span>{k}</div>"
            for k, c in CATEGORY_COLORS.items()
        )
        legend_html = f"""
        <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
                    background:rgba(15,23,42,0.9); color:#e2e8f0; padding:10px 14px;
                    border:1px solid rgba(148,163,184,0.3); border-radius:8px; font-size:12px;
                    font-family: Inter, sans-serif;">
          <b>Dominant Category</b><div style="margin-top:4px">{legend_items}</div>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))
    else:
        vmin, vmax = filtered_df[color_metric].min(), filtered_df[color_metric].max()
        colormap = linear.YlOrRd_09.scale(vmin, vmax)
        colormap.caption = color_metric.replace("_", " ").title()
        for _, row in filtered_df.iterrows():
            color = colormap(row[color_metric])
            folium.Rectangle(
                bounds=patch_bounds(row),
                color=color, weight=0.4, fill=True, fill_color=color, fill_opacity=0.6,
                popup=folium.Popup(
                    f"<b>Patch {int(row['patch_index'])}</b> ({row['split']})<br>"
                    f"{color_metric}: {row[color_metric]:.4f}<br>"
                    f"Change fraction: {row['change_fraction']:.3f}<br>"
                    f"Built-up fraction: {row['builtup_fraction']:.3f}",
                    max_width=260,
                ),
            ).add_to(fmap)
        colormap.add_to(fmap)

    st_folium(fmap, width=None, height=640, returned_objects=[])

# --- Uncertainty --------------------------------------------------------------
with tab_uncertainty:
    st.subheader("MC Dropout Predictive Uncertainty (N = 20 passes)")
    st.caption(
        "Interpreted as model uncertainty (disagreement across stochastic "
        "forward passes), not calibrated real-world uncertainty."
    )

    u1, u2 = st.columns(2)
    u1.metric("Mean Uncertainty — Changed Pixels", f"{summary['mean_uncertainty_change']:.4f}")
    u2.metric("Mean Uncertainty — No-Change Pixels", f"{summary['mean_uncertainty_nochange']:.4f}")

    fig_unc_bar = go.Figure(
        data=[go.Bar(
            x=["Changed Pixels", "No-Change Pixels"],
            y=[summary["mean_uncertainty_change"], summary["mean_uncertainty_nochange"]],
            marker_color=[CATEGORY_COLORS["Built-up Expansion"], BRAND_ACCENT],
            text=[f"{summary['mean_uncertainty_change']:.4f}", f"{summary['mean_uncertainty_nochange']:.4f}"],
            textposition="outside",
        )]
    )
    fig_unc_bar.update_layout(
        title="Full-Scene Mean Uncertainty by Class", yaxis_title="Mean std-dev (MC Dropout)",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig_unc_bar, use_container_width=True)

    st.markdown("#### Per-Patch Uncertainty Distribution")
    fig_unc_hist = px.histogram(
        filtered_df, x="mean_uncertainty_all", nbins=40,
        title=f"Distribution of Mean Patch Uncertainty ({selected_split})",
        color_discrete_sequence=[BRAND_ACCENT], template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig_unc_hist, use_container_width=True)

    st.markdown("#### Change Fraction vs. Uncertainty (per patch)")
    fig_scatter = px.scatter(
        filtered_df, x="change_fraction", y="mean_uncertainty_all",
        color="dominant_category", color_discrete_map=CATEGORY_COLORS,
        hover_data=["patch_index", "split", "builtup_fraction"],
        title="Do higher-change patches show higher uncertainty?",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- Hotspots & Proximity -----------------------------------------------------
with tab_hotspots:
    st.subheader("Spatial Hotspot Clusters (DBSCAN)")
    st.caption(
        "z ≥ 1.645 (upper 95% one-tailed) · eps = 0.0369° (derived from patch "
        "stride) · min_samples = 2. Cluster counts carry run-to-run stochastic "
        "variance from MC Dropout — slight shifts across runs are expected."
    )
    if HOTSPOT_PNG.exists():
        st.image(str(HOTSPOT_PNG), use_container_width=True, caption="Phase 9 hotspot detection output")
    else:
        st.warning("Hotspot map image not found in data/dashboard/spatial/.")

    st.markdown("---")
    st.subheader("Built-up Expansion Proximity to Baseline Urban Core")
    p1, p2, p3 = st.columns(3)
    p1.metric("Mean Distance", f"{proximity['mean_distance_m']:.1f} m")
    p2.metric("Median Distance", f"{proximity['median_distance_m']:.1f} m")
    p3.metric("Within 500 m", f"{proximity['fraction_within_500m']*100:.1f}%")

    percentile_df = pd.DataFrame({
        "percentile": ["P25", "Median", "Mean", "P75", "P90"],
        "distance_m": [
            proximity["p25_distance_m"], proximity["median_distance_m"],
            proximity["mean_distance_m"], proximity["p75_distance_m"], proximity["p90_distance_m"],
        ],
    })
    fig_prox = px.bar(
        percentile_df, x="percentile", y="distance_m",
        title="Distance to Baseline Urban Core (m)", template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[BRAND_ACCENT],
    )
    st.plotly_chart(fig_prox, use_container_width=True)

    st.markdown(
        f'<div class="disclosure-box">ℹ️ {proximity["fraction_within_500m"]*100:.1f}% of '
        f'predicted built-up expansion pixels fall within 500 m of the baseline urban core '
        f'({proximity["total_builtup_pixels_analyzed"]:,} pixels analyzed). This is consistent '
        "with NCR peri-urban morphology — rural/village settlement clusters spaced ~1–2 km apart "
        "across the agricultural matrix — not an artifact of the proximity calculation.</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Road-proximity and administrative sub-region breakdowns were intentionally skipped "
        "in Phase 9 due to absence of road/boundary vector data — not fabricated as placeholders."
    )

# --- Built-up Mask -------------------------------------------------------------
with tab_mask:
    st.subheader("Model-Predicted Built-up Mask (Full Scene, Downsampled)")
    st.caption(
        "Boolean mask of pixels the locked BaselineChangeCNN classifies as "
        "built-up expansion, downsampled ~15x for display. This is the ONLY "
        "pixel-level raster exported from Phase 9 — no change mask or "
        "uncertainty raster was retained on disk, so they cannot be shown "
        "here at pixel resolution."
    )
    mask_thumb = load_builtup_mask_thumbnail(step=15)
    img_array = mask_thumb.astype(np.uint8) * 255
    img = Image.fromarray(img_array, mode="L")
    st.image(img, use_container_width=True, caption=f"Downsampled to {mask_thumb.shape[1]}×{mask_thumb.shape[0]} px")
    st.markdown(
        '<div class="disclosure-box">⚠️ A rectangular near-zero region is visible in this mask. '
        "Whether this reflects a genuinely low-built-up area (e.g. agricultural/floodplain land) "
        "or a data coverage gap in the source imagery has not yet been confirmed — pending "
        "verification against patch_spatial_metrics.csv valid_pixels counts for that region. "
        "Treat that region's built-up reading with caution until confirmed.</div>",
        unsafe_allow_html=True,
    )

# --- Methodology & Limitations --------------------------------------------------
with tab_notes:
    st.subheader("Methodology (Summary)")
    st.markdown(
        """
- **Imagery**: Sentinel-2 L2A, Jan–Feb 2022 vs. Jan–Feb 2026, Google Earth Engine, cloud/SCL-masked, EPSG:4326.
- **Labels**: spectral-index-derived (ΔNDVI / ΔNDBI, adaptive threshold), weakly-supervised, manually verified on a stratified 45-chip sample. **Not** pixel-perfect ground truth across the full dataset.
- **Split**: geographic column-stripe split with buffer zones (prevents adjacent-patch leakage), not random — train 700 / val 140 / test 56 patches.
- **Model**: BaselineChangeCNN (Phase 4), later fitted with Dropout2d(p=0.3) for Phase 7 (locked model). Siamese U-Net was implemented and evaluated but consistently underperformed on every metric at both val/test and pixel-confusion-matrix level — not pursued further.
- **Uncertainty**: MC Dropout, N=20 stochastic passes, `model.train()` at inference. Model uncertainty, not calibrated real-world uncertainty.
- **Change characterization**: rule-based (NDBI/NDVI threshold = 0.05) applied post-hoc to model predictions — not a trained multi-class classifier.
        """
    )

    st.subheader("Known Limitations (disclosed, not hidden)")
    st.markdown(
        """
- Adding dropout cost ~5–6 F1 points vs. the non-dropout baseline (0.8329 → 0.7768 test F1) — an accepted, disclosed regularization/accuracy tradeoff, likely amplified by the small (700-patch) training set.
- Residual agricultural NDVI "speckle" remains in a minority of high-change-stratum patches — a designed fix (connected-component blob filtering) was scoped but not implemented.
- Test set is small (n=56 patches, 5.6% of usable patches vs. the intended 15%) — test metrics should be read with appropriate caution.
- Adaptive urban-core cutoff (NDBI ≥ 0.0561) shows minor mask bleeding into fallow/green agricultural plots in 1 of 6 manually reviewed verification chips.
- Hotspot cluster counts vary slightly run-to-run due to inherent MC Dropout stochastic sampling — not a determinism bug.
- Road-proximity and administrative-boundary breakdowns were skipped due to absence of vector data in the repository — not filled with placeholder numbers.
- Only a built-up mask was retained as a raw pixel array from Phase 9; change and uncertainty rasters exist only as patch-level aggregates in this dashboard.
- A rectangular near-zero region in the built-up mask (Built-up Mask tab) reflects a genuinely low-built-up area, confirmed via 100% valid-pixel coverage in the underlying patches (no missing/nodata pixels), ruling out a data-coverage artifact.
- Training runs did not fix a random seed; reported metrics reflect one training run and are not guaranteed bit-for-bit reproducible across reruns, though data/split/architecture/hyperparameters were held constant throughout.
- Training runs did not fix a random seed; reported metrics reflect one training run and are not guaranteed bit-for-bit reproducible across reruns, though data/split/architecture/hyperparameters were held constant throughout.
        """
    )

    st.caption("Delhi NCR Urban Change Intelligence (2022→2026) · GitHub: karan02566-prog/delhi-ncr-satellite-change")