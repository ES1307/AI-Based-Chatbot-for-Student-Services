"""CampusGuide - a document-grounded student support chatbot."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from rag_engine import RAGEngine, SourceChunk, load_documents


BASE_DIR = Path(__file__).parent
SAMPLE_DIR = BASE_DIR / "sample_documents"

st.set_page_config(
    page_title="CampusGuide | Student Support",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    """One visual system for the app - deliberately small and Streamlit-safe."""
    st.markdown(
        """
        <style>
        :root {
          --ink: #17203a;
          --muted: #69758f;
          --canvas: #f7f8fc;
          --surface: #ffffff;
          --line: #e2e7f1;
          --accent: #5266d8;
          --accent-hover: #4053be;
          --navy: #111a35;
          --navy-soft: #1a2850;
          --success: #7ec39f;
        }

        html, body, [class*="css"] {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .stApp {
          background:
            radial-gradient(56rem 30rem at 88% -7%, #e8efff 0%, transparent 60%),
            var(--canvas);
          color: var(--ink);
        }
        #MainMenu, footer { visibility: hidden; }

        /* Native Streamlit chrome */
        header[data-testid="stHeader"] {
          height: 3.4rem;
          min-height: 3.4rem;
          background: rgba(18, 31, 68, .78) !important;
          border-bottom: 1px solid rgba(184, 203, 255, .18);
          box-shadow: none;
          backdrop-filter: blur(22px) saturate(135%);
          -webkit-backdrop-filter: blur(22px) saturate(135%);
        }
        /* Keep Streamlit's toolbar visible: it owns the native sidebar reopen control. */
        header[data-testid="stHeader"] [data-testid="stToolbar"] { display: flex !important; color: #fff !important; }
        header[data-testid="stHeader"] [data-testid="stAppDeployButton"] { display: flex !important; color: #fff !important; }
        [data-testid="stSidebarCollapsedControl"] {
          display: flex !important;
          align-items: center;
          justify-content: center;
          padding: .2rem !important;
          border-radius: 9px;
          background: rgba(255, 255, 255, .12);
          border: 1px solid rgba(222, 230, 255, .30);
        }
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] button {
          color: #ffffff !important;
          background: #2b478f !important;
          border: 1px solid rgba(223, 231, 255, .42) !important;
          border-radius: 7px !important;
        }
        [data-testid="stSidebarCollapsedControl"] button svg,
        [data-testid="stSidebarCollapseButton"] button svg {
          fill: #ffffff !important;
          color: #ffffff !important;
          opacity: 1 !important;
        }
        [data-testid="stSidebarCollapsedControl"] button svg *,
        [data-testid="stSidebarCollapseButton"] button svg * {
          fill: #ffffff !important;
          stroke: #ffffff !important;
        }
        header[data-testid="stHeader"] [data-testid="stAppDeployButton"] button {
          color: #ffffff !important;
          background: rgba(255, 255, 255, .10) !important;
          border: 1px solid rgba(225, 233, 255, .24) !important;
          border-radius: 8px !important;
        }

        /* Main canvas */
        .block-container {
          max-width: 900px;
          padding: 5.15rem 1.5rem 3.5rem;
        }
        .hero { padding: .45rem 0 1.8rem; }
        .hero-row {
          display: flex;
          align-items: center;
          gap: .65rem;
          margin-bottom: .8rem;
        }
        .eyebrow {
          color: var(--accent);
          font-size: .7rem;
          font-weight: 700;
          letter-spacing: .15em;
        }
        .grounded-badge {
          color: #4b6671;
          background: #edf7f1;
          border: 1px solid #d8ebdf;
          border-radius: 999px;
          padding: .22rem .55rem;
          font-size: .72rem;
          font-weight: 600;
        }
        .hero h1 {
          max-width: 650px;
          margin: 0 0 .55rem;
          color: var(--ink);
          font-size: clamp(2.3rem, 5vw, 3.4rem);
          font-weight: 720;
          letter-spacing: -.06em;
          line-height: 1.03;
        }
        .hero p {
          max-width: 505px;
          margin: 0;
          color: var(--muted);
          font-size: 1rem;
          line-height: 1.6;
        }
        .section-title {
          margin: 1.5rem 0 .25rem;
          color: var(--ink);
          font-size: 1.12rem;
          font-weight: 700;
          letter-spacing: -.02em;
        }
        .muted { color: var(--muted); font-size: .91rem; }

        /* Reusable surfaces */
        div[data-testid="stExpander"],
        div[data-testid="stForm"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
          background: rgba(255, 255, 255, .86);
          border: 1px solid var(--line);
          border-radius: 14px;
          box-shadow: 0 10px 30px rgba(31, 45, 82, .04);
        }
        div[data-testid="stExpander"] { overflow: hidden; }
        div[data-testid="stExpander"] details summary {
          min-height: 3.45rem;
          display: flex;
          align-items: center;
          color: #314363;
          font-weight: 600;
        }
        div[data-testid="stForm"] { padding: .9rem; }
        div[data-testid="stVerticalBlockBorderWrapper"] { padding: .2rem .15rem; }

        /* Inputs and buttons */
        .stTextInput input {
          min-height: 3rem !important;
          background: #fff !important;
          border: 1px solid #cbd4e3 !important;
          border-radius: 10px !important;
          color: var(--ink) !important;
          box-shadow: none !important;
        }
        .stTextInput input::placeholder { color: #929cb0 !important; opacity: 1; }
        .stTextInput input:focus {
          border-color: var(--accent) !important;
          box-shadow: 0 0 0 3px rgba(82, 102, 216, .13) !important;
        }
        div[data-testid="stFormSubmitButton"] > button {
          min-height: 3rem;
          border: 0;
          border-radius: 10px;
          background: var(--accent) !important;
          color: #fff !important;
          font-weight: 650;
          transition: background .16s ease;
        }
        div[data-testid="stFormSubmitButton"] > button:hover { background: var(--accent-hover) !important; }
        .stButton > button {
          min-height: 2.45rem;
          border: 1px solid var(--line);
          border-radius: 10px;
          background: #fff;
          color: #40528c;
          box-shadow: none;
          transition: border-color .16s ease, background .16s ease;
        }
        .stButton > button:hover {
          border-color: #aebbe8;
          background: #f7f8ff;
          color: #344a9c;
        }

        /* Document and source components */
        [data-testid="stFileUploaderFile"] { display: none !important; }
        [data-testid="stFileUploader"] section {
          min-height: 6.25rem;
          background: #fafbfe !important;
          border: 1px dashed #b9c5e0 !important;
          border-radius: 10px !important;
        }
        [data-testid="stFileUploader"] button {
          background: #eef1ff !important;
          color: #4057bb !important;
          border-color: #d8dff6 !important;
        }
        .file-chip {
          margin: .45rem 0;
          padding: .55rem .7rem;
          overflow-wrap: anywhere;
          color: #40558f;
          background: #f1f4ff;
          border: 1px solid #dfe5fa;
          border-radius: 9px;
          font-size: .85rem;
          font-weight: 600;
        }
        .source-tag {
          color: #5368d8;
          font-size: .72rem;
          font-weight: 700;
          letter-spacing: .08em;
        }

        /* Sidebar as a quiet utility rail */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
          background: linear-gradient(165deg, var(--navy), var(--navy-soft)) !important;
        }
        [data-testid="stSidebar"][aria-expanded="true"] {
          width: 16.25rem !important;
          min-width: 16.25rem !important;
          border-right: 1px solid rgba(255, 255, 255, .08);
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
          width: 0 !important;
          min-width: 0 !important;
          border-right: 0 !important;
        }
        [data-testid="stSidebar"] > div:first-child { padding: 1.2rem 1rem; }
        [data-testid="stSidebar"] h2 {
          color: #f7f9ff !important;
          font-size: 1.2rem !important;
          letter-spacing: -.03em;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span { color: #eaf0ff !important; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #aab8d4 !important; }
        [data-testid="stSidebar"] hr {
          margin: 1.1rem 0;
          border-color: rgba(205, 219, 255, .16);
        }
        [data-testid="stSidebar"] [data-testid="stMetric"] {
          padding: .5rem .55rem;
          background: rgba(255, 255, 255, .07);
          border: 1px solid rgba(206, 220, 255, .14);
          border-radius: 10px;
        }
        [data-testid="stSidebar"] [data-testid="stMetric"] * { color: #eef3ff !important; }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #aab8d4 !important; font-size: .7rem; }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.45rem !important; }

        @media (max-width: 760px) {
          header[data-testid="stHeader"] { height: 3rem; min-height: 3rem; }
          .block-container { padding: 4.25rem 1rem 2rem; }
          .hero { padding-bottom: 1.3rem; }
          .hero h1 { font-size: 2.25rem; }
          .grounded-badge { display: none; }
          [data-testid="stSidebar"][aria-expanded="true"] { width: auto !important; min-width: auto !important; }
          div[data-testid="stVerticalBlockBorderWrapper"] { box-shadow: none; }
        }
        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after { transition: none !important; scroll-behavior: auto !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def build_engine(file_signatures: tuple[tuple[str, int], ...]) -> RAGEngine:
    paths = [Path(name) for name, _ in file_signatures]
    return RAGEngine(load_documents(paths))


def save_uploads(files: list[st.runtime.uploaded_file_manager.UploadedFile]) -> list[Path]:
    """Save uploads outside OneDrive, which can block app-created folders."""
    upload_dir = Path(tempfile.gettempdir()) / "campusguide_uploads"
    upload_dir.mkdir(exist_ok=True)
    saved: list[Path] = []
    for uploaded in files:
        target = upload_dir / uploaded.name
        target.write_bytes(uploaded.getvalue())
        saved.append(target)
    return saved


def render_sources(sources: list[SourceChunk]) -> None:
    st.markdown("<p class='section-title'>Sources</p>", unsafe_allow_html=True)
    st.caption("Open a passage to check the evidence used for this response.")
    for index, source in enumerate(sources, start=1):
        name = source.source_name if len(source.source_name) <= 54 else f"{source.source_name[:51]}..."
        with st.expander(f"{index}. {name} · relevance {source.score:.0%}"):
            st.markdown(f"<span class='source-tag'>SOURCE CHUNK {source.chunk_id + 1}</span>", unsafe_allow_html=True)
            st.write(source.text)


apply_styles()

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


def set_question(question: str) -> None:
    st.session_state.question_input = question


st.markdown(
    """
    <section class="hero">
      <div class="hero-row">
        <span class="eyebrow">CAMPUSGUIDE</span>
        <span class="grounded-badge">● Source-grounded</span>
      </div>
      <h1>Ask your documents.</h1>
      <p>Clear student-support answers, grounded in the handbook you provide.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🎓 CampusGuide")
    st.caption("Document-grounded student support")
    st.divider()
    use_sample = st.toggle("Use sample university handbook", value=True)

with st.expander("Documents", expanded=False):
    uploaded_files = st.file_uploader(
        "Upload PDF or text documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="knowledge_documents",
        help="Add policy documents, student handbooks, fee guides, or service directories.",
    )
    visible_files = [file.name for file in uploaded_files] if uploaded_files else []
    if use_sample:
        visible_files.insert(0, "northstar_university_handbook.txt (sample)")
    if visible_files:
        st.caption("Ready for search")
        for name in visible_files:
            st.markdown(f"<div class='file-chip'>📄 {name}</div>", unsafe_allow_html=True)
    else:
        st.caption("No documents selected yet.")

document_paths: list[Path] = []
if use_sample:
    document_paths.extend(SAMPLE_DIR.glob("*.txt"))
if uploaded_files:
    document_paths.extend(save_uploads(uploaded_files))

if not document_paths:
    st.info("Add at least one PDF or text document to start asking questions.")
    st.stop()

signatures = tuple((str(path), path.stat().st_mtime_ns) for path in document_paths)
try:
    with st.spinner("Preparing the document knowledge base..."):
        engine = build_engine(signatures)
except Exception as exc:
    st.error(f"The document index could not be created: {exc}")
    st.stop()

with st.sidebar:
    st.divider()
    st.caption("● Ready to search")
    metric_one, metric_two = st.columns(2)
    metric_one.metric("Documents", len(document_paths))
    metric_two.metric("Chunks", len(engine.chunks))

st.markdown("<p class='section-title'>Ask a question</p>", unsafe_allow_html=True)
st.markdown("<p class='muted'>Use the same wording you would use with a student-support advisor.</p>", unsafe_allow_html=True)

with st.form("question_form", clear_on_submit=False):
    st.text_input(
        "Question",
        key="question_input",
        placeholder="e.g. How do I apply for a scholarship?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Search the handbook", type="primary", use_container_width=True)

with st.expander("Try an example"):
    examples = [
        "What support is available for mental wellbeing?",
        "How do I request financial assistance?",
        "What is the attendance requirement?",
    ]
    for example in examples:
        st.button(example, key=example, use_container_width=True, on_click=set_question, args=(example,))

if submitted and st.session_state.question_input.strip():
    question = st.session_state.question_input.strip()
    with st.spinner("Searching the most relevant passages..."):
        result = engine.answer(question)

    st.markdown("<p class='section-title'>Response</p>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(result.answer)
        if not result.used_generator:
            st.markdown("**Suggested next step**")
            st.write(result.next_step)

    if not result.is_grounded:
        st.warning("No matching evidence was found, so CampusGuide did not answer from general knowledge.")
    elif result.used_generator:
        st.caption("Written from the retrieved passages and checked against the source evidence.")
    else:
        st.caption("This response was assembled directly from the highest-ranked source passages.")

    if result.sources:
        render_sources(result.sources)
