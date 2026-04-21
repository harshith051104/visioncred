"""
VisionCred — Gradio Web Application
======================================
Interactive frontend for the VisionCred credit assessment engine.

Features:
    - Select a store folder to analyze
    - View store images in a gallery
    - See predictions, confidence scores, risk flags
    - Explore feature breakdowns and key drivers
    - Download full JSON report
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from PIL import Image

from src.pipeline import VisionCredPipeline
from src.config import DATA_DIR, OUTPUT_DIR
from utils.logger import get_logger

logger = get_logger("gradio_app")


# ── Initialize Pipeline (singleton) ───────────────────────────────────────────
pipeline = None


def get_pipeline():
    """Lazy-load the pipeline to avoid slow startup."""
    global pipeline
    if pipeline is None:
        pipeline = VisionCredPipeline()
    return pipeline


# ── Helper Functions ──────────────────────────────────────────────────────────

def get_store_list():
    """Get list of available store folders."""
    if not DATA_DIR.exists():
        return []
    stores = sorted([
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith("store_")
    ])
    return stores


def load_store_images(store_name: str):
    """Load thumbnail images for gallery display."""
    store_path = DATA_DIR / store_name
    images = []
    if store_path.exists():
        for ext in [".webp", ".jpg", ".jpeg", ".png"]:
            for img_path in sorted(store_path.glob(f"*{ext}")):
                try:
                    img = Image.open(img_path)
                    images.append((img, img_path.stem))
                except Exception:
                    pass
    return images


def format_currency(value):
    """Format INR currency with commas."""
    return f"₹{value:,.0f}"


def format_range(values):
    """Format a [min, max] range as currency."""
    return f"{format_currency(values[0])} – {format_currency(values[1])}"


# ── Main Analysis Function ────────────────────────────────────────────────────

def analyze_store(store_name: str):
    """
    Run the full pipeline on a selected store and return
    formatted results for the Gradio UI.
    """
    if not store_name:
        return (
            "⚠️ Please select a store first.",
            [],   # gallery
            "",   # features
            "",   # risk
            "",   # drivers
            "",   # json
        )

    store_path = DATA_DIR / store_name

    if not store_path.exists():
        return (
            f"❌ Store directory not found: {store_path}",
            [],
            "",
            "",
            "",
            "",
        )

    try:
        # Run pipeline
        pipe = get_pipeline()
        result = pipe.process_single_store_path(store_path)

        # ── Format Summary ─────────────────────────────────────────
        summary = f"""
## 📊 Credit Assessment: {result['store_id']}

| Metric | Range |
|--------|-------|
| 💰 **Daily Sales** | {format_range(result['daily_sales_range'])} |
| 📈 **Monthly Revenue** | {format_range(result['monthly_revenue_range'])} |
| 💵 **Monthly Income** | {format_range(result['monthly_income_range'])} |

### Confidence: **{result['confidence_score']:.2f}** — {result.get('confidence_interpretation', '')}

### Risk Level: **{result['overall_risk_level']}**

> 📋 *{result['recommendation']}*

---
**Formula Used:**
`{result['formula']}`

⏱️ Processed in **{result['processing_time_seconds']:.2f}s** | 🖼️ **{result['num_images_processed']}** images analyzed
"""

        # ── Format Features ────────────────────────────────────────
        feats = result.get("features", {})
        geo = result.get("geo_info", {})
        features_md = f"""
### 🔍 Vision Features
| Feature | Value |
|---------|-------|
| Shelf Density Index | `{feats.get('shelf_density_index', 0):.4f}` |
| SKU Diversity Score | `{feats.get('sku_diversity_score', 0):.4f}` |
| Inventory Value | {format_range(feats.get('inventory_value_range', [0, 0]))} |
| Store Size Proxy | `{feats.get('store_size_proxy', 0):.4f}` |
| Store Viability Index | `{feats.get('store_viability_index', 0):.4f}` |

### 🌍 Geo Features
| Feature | Value |
|---------|-------|
| Location Type | **{geo.get('location_type', 'N/A')}** |
| Nearest Metro Center | {geo.get('nearest_metro', 'N/A')} ({geo.get('distance_km', 0):.1f} km) |
| Footfall Score | `{feats.get('geo_footfall_score', 0):.4f}` |
| Competition Density | `{feats.get('competition_density', 0):.4f}` |
| GPS Source | {geo.get('gps_source', 'N/A')} |
| Coordinates | {geo.get('coordinates', {}).get('latitude', 'N/A')}, {geo.get('coordinates', {}).get('longitude', 'N/A')} |
"""

        # ── Format Risk Flags ──────────────────────────────────────
        risk_details = result.get("risk_details", [])
        if risk_details:
            risk_lines = []
            for flag in risk_details:
                severity_emoji = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(flag["severity"], "⚪")
                risk_lines.append(
                    f"{severity_emoji} **{flag['type']}** "
                    f"({flag['severity']})\n\n"
                    f"  {flag['detail']}\n\n"
                    f"  💡 *{flag['recommendation']}*\n"
                )
            risk_md = "\n---\n".join(risk_lines)
        else:
            risk_md = "✅ No risk flags detected. Data appears consistent."

        # ── Format Key Drivers ─────────────────────────────────────
        drivers = result.get("key_drivers", [])
        drivers_md = "\n".join(
            [f"- {d}" for d in drivers]
        )

        # ── Full JSON ─────────────────────────────────────────────
        json_output = json.dumps(result, indent=2, ensure_ascii=False)

        # ── Gallery Images ─────────────────────────────────────────
        gallery = load_store_images(store_name)

        return (
            summary,
            gallery,
            features_md,
            risk_md,
            drivers_md,
            json_output,
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return (
            f"❌ Analysis failed: {str(e)}",
            [],
            "",
            "",
            "",
            "",
        )


# ── Build Gradio Interface ────────────────────────────────────────────────────

def create_app():
    """Create and configure the Gradio app."""

    store_list = get_store_list()

    with gr.Blocks(
        title="VisionCred — AI Credit Engine",
    ) as app:

        # ── Header ────────────────────────────────────────────────
        gr.HTML("""
            <div class="header-text">
                <h1>🏪 VisionCred</h1>
                <p>AI Credit Engine for Kirana Stores</p>
                <p style="font-size: 0.85em; color: #7aa3c4 !important;">
                    Vision + Geo Intelligence + Economic Modeling = Explainable Credit Assessment
                </p>
            </div>
        """)

        with gr.Row():
            # ── Left Panel: Store Selection ────────────────────────
            with gr.Column(scale=1):
                store_dropdown = gr.Dropdown(
                    choices=store_list,
                    label="🏬 Select Store",
                    info="Choose a store folder to analyze",
                    interactive=True,
                )
                analyze_btn = gr.Button(
                    "🚀 Run Analysis",
                    variant="primary",
                    size="lg",
                )
                gallery = gr.Gallery(
                    label="📷 Store Images",
                    columns=2,
                    rows=3,
                    height=400,
                    object_fit="contain",
                )

            # ── Right Panel: Results ───────────────────────────────
            with gr.Column(scale=2):
                summary_output = gr.Markdown(
                    label="Assessment Summary",
                    value="*Select a store and click 'Run Analysis' to begin.*",
                )

        # ── Detailed Tabs ──────────────────────────────────────────
        with gr.Tabs():
            with gr.TabItem("📊 Features"):
                features_output = gr.Markdown(
                    label="Feature Breakdown",
                )
            with gr.TabItem("⚠️ Risk Flags"):
                risk_output = gr.Markdown(
                    label="Risk Assessment",
                )
            with gr.TabItem("🔑 Key Drivers"):
                drivers_output = gr.Markdown(
                    label="Key Drivers",
                )
            with gr.TabItem("📄 Raw JSON"):
                json_output = gr.Code(
                    label="Full JSON Output",
                    language="json",
                    lines=30,
                )

        # ── Footer ─────────────────────────────────────────────────
        gr.HTML("""
            <div style="text-align: center; padding: 1rem; color: #666; font-size: 0.85em;">
                VisionCred v1.0 — Built with YOLOv8 + Gradio<br>
                All estimates are ranges, not predictions. Use as supplementary data for credit decisions.
            </div>
        """)

        # ── Event Handlers ────────────────────────────────────────
        analyze_btn.click(
            fn=analyze_store,
            inputs=[store_dropdown],
            outputs=[
                summary_output,
                gallery,
                features_output,
                risk_output,
                drivers_output,
                json_output,
            ],
        )

        # Auto-load gallery on store selection
        store_dropdown.change(
            fn=lambda s: load_store_images(s) if s else [],
            inputs=[store_dropdown],
            outputs=[gallery],
        )

    return app


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
            .gradio-container { max-width: 1200px !important; }
            .header-text {
                text-align: center;
                padding: 1rem;
                background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 100%);
                border-radius: 12px;
                color: white;
                margin-bottom: 1rem;
            }
            .header-text h1 { color: white !important; margin: 0; font-size: 2rem; }
            .header-text p { color: #94b8d4 !important; margin: 0.3rem 0 0 0; }
        """,
    )
