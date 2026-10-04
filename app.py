import streamlit as st

from config import APP_NAME, EMBEDDING_MODEL, SOURCE_XLSX_URL
from crew import run_prunely
from knowledge_base import KnowledgeBase


st.set_page_config(
    page_title="Prunely — AI Pruning Advisor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #0b3d2e 0%, #145a3d 55%, #1f7a52 100%);
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 18px 45px rgba(8, 54, 39, .18);
}
.hero h1 { font-size: 3rem; margin: 0; letter-spacing: -1px; }
.hero p { font-size: 1.1rem; opacity: .92; max-width: 760px; }
.badge {
    display: inline-block;
    padding: .35rem .7rem;
    border-radius: 999px;
    background: rgba(255,255,255,.14);
    margin-bottom: .8rem;
    font-size: .82rem;
}
.card {
    padding: 1.15rem 1.25rem;
    border: 1px solid rgba(49, 87, 69, .14);
    border-radius: 18px;
    background: rgba(255,255,255,.65);
    box-shadow: 0 8px 25px rgba(0,0,0,.04);
}
.small-muted {
    color: #64748b;
    font-size: .88rem;
}
div[data-testid="stTextArea"] textarea {
    border-radius: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="badge">🌱 Multi-agent horticulture assistant</div>
  <h1>Prunely</h1>
  <p>
    Describe your plant in text. Three specialist agents analyze the plant,
    build a pruning plan, and verify it against the Prunely knowledge base.
  </p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()


with st.sidebar:
    st.markdown("### 🌿 Prunely")
    st.markdown(
        "Evidence-first pruning guidance powered by a single verified "
        "horticulture workbook."
    )

    st.divider()

    st.markdown("**Pipeline**")
    st.markdown("1. 🔎 Plant Analysis")
    st.markdown("2. ✂️ Pruning Expert")
    st.markdown("3. 🛡️ Data Verifier")

    st.divider()

    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")
    st.caption("LLM: Groq · GPT-OSS 120B")
    st.caption("Vector store: ChromaDB")
    st.caption("Source: public Google Sheet XLSX export")


st.markdown("### Tell Prunely about your plant")

query = st.text_area(
    "Plant description",
    height=170,
    placeholder=(
        "Example: I have a bougainvillea in my home garden. It has become "
        "very bushy and long branches are growing over the walkway. "
        "I want to know what I should remove and when I should prune it."
    ),
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 3])

with col1:
    analyze = st.button(
        "🌿 Analyze & Prune",
        type="primary",
        use_container_width=True,
    )

with col2:
    st.markdown(
        '<div class="small-muted">'
        "Text only — no image analysis is used in this version."
        "</div>",
        unsafe_allow_html=True,
    )


if analyze:

    if not query.strip():
        st.warning("Please describe the plant and what you want to prune.")
        st.stop()

    try:

        with st.spinner("Preparing the Prunely knowledge base…"):
            kb = get_kb()

        with st.status(
            "🌿 Prunely agents are working…",
            expanded=True,
        ) as status:

            st.write("🔎 Plant Analysis Agent")
            st.write("✂️ Pruning Expert Agent")
            st.write("🛡️ Data Verifier Agent")

            result = run_prunely(
                query.strip(),
                kb,
            )

            status.update(
                label="✅ Verified recommendation ready",
                state="complete",
                expanded=False,
            )

        st.markdown("## 🌱 Your Prunely recommendation")

        st.markdown(result["final"])

        with st.expander("🔎 Plant Analysis Agent — details"):
            st.markdown(result["analysis"])

        with st.expander("✂️ Pruning Expert Agent — details"):
            st.markdown(result["pruning"])

        with st.expander("🛡️ Data Verifier Agent — audit"):
            st.markdown(result["verification"])

        with st.expander("📚 Knowledge-base source"):

            st.caption("Source workbook:")

            st.code(
                SOURCE_XLSX_URL,
                language="text",
            )

            st.caption(
                "The displayed source traceability is generated from the "
                "sheet/row metadata stored with the vector records."
            )

    except Exception as exc:

        st.error(
            "Prunely could not complete the analysis."
        )

        st.exception(exc)


else:

    st.markdown(
        """
<div class="card">
<b>💡 For better results</b><br><br>
Include the plant's common name if you know it, where it is growing,
its approximate age/size, what looks wrong, and what you want to achieve.
</div>
""",
        unsafe_allow_html=True,
    )
