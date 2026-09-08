import os
import streamlit as st
from dotenv import load_dotenv

from utils import extract_pdf_text, candidate_name_from_filename
from rag import build_vector_store, retrieve_candidate_evidence
from agents import screen_candidate, generate_interview_questions


load_dotenv()


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI HR Recruitment Assistant",
    page_icon="🤖",
    layout="wide",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 AI HR Recruitment Assistant")

st.caption(
    "Agent + Tools + RAG for resume screening, matching and interview preparation"
)


# --------------------------------------------------
# API Key Check
# --------------------------------------------------

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.warning(
        "Add GOOGLE_API_KEY or GEMINI_API_KEY to Streamlit Secrets before screening."
    )


# --------------------------------------------------
# Inputs
# --------------------------------------------------

job = st.text_area(
    "Job Description",
    height=240,
    placeholder=(
        "Example: Full Stack Developer with Python, Django, "
        "React, REST APIs, SQL and Docker..."
    ),
)

files = st.file_uploader(
    "Upload candidate resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = {}

if "evidence" not in st.session_state:
    st.session_state.evidence = {}

if "questions" not in st.session_state:
    st.session_state.questions = None


# --------------------------------------------------
# Screen Candidates
# --------------------------------------------------

if st.button(
    "🚀 Screen Candidates",
    type="primary",
    use_container_width=True,
):

    if not job.strip():
        st.error("Please provide a Job Description.")
        st.stop()

    if not files:
        st.error("Please upload at least one PDF resume.")
        st.stop()

    if not api_key:
        st.error(
            "Gemini API key is missing. "
            "Add GOOGLE_API_KEY or GEMINI_API_KEY in Streamlit Secrets."
        )
        st.stop()

    # ----------------------------------------------
    # Extract Resume Text
    # ----------------------------------------------

    resumes = {}

    for file in files:

        try:
            text = extract_pdf_text(file.getvalue())

            if text.strip():
                candidate = candidate_name_from_filename(file.name)
                resumes[candidate] = text

        except Exception as exc:
            st.warning(
                f"Could not read {file.name}: {exc}"
            )

    if not resumes:
        st.error("No readable PDF text was found.")
        st.stop()

    # ----------------------------------------------
    # Build RAG + Screen Candidates
    # ----------------------------------------------

    try:

        with st.spinner("Building Gemini RAG index..."):
            store = build_vector_store(resumes)

        results = {}
        evidence_map = {}

        for candidate in resumes:

            with st.spinner(f"Screening {candidate}..."):

                evidence = retrieve_candidate_evidence(
                    store,
                    job,
                    candidate,
                )

                result = screen_candidate(
                    candidate,
                    job,
                    evidence,
                )

                results[candidate] = result
                evidence_map[candidate] = evidence

        st.session_state.results = results
        st.session_state.evidence = evidence_map

        # Clear old questions after new screening
        st.session_state.questions = None

        st.success("✅ Screening complete.")

    except Exception as exc:

        st.error(
            "Gemini API request failed. "
            "Check your Google API key and free-tier quota."
        )

        st.exception(exc)


# --------------------------------------------------
# Candidate Ranking
# --------------------------------------------------

if st.session_state.results:

    st.subheader("🏆 Candidate Ranking")

    ranked = sorted(
        st.session_state.results.values(),
        key=lambda x: x.overall_score,
        reverse=True,
    )

    for i, result in enumerate(ranked, 1):

        with st.container(border=True):

            col1, col2, col3 = st.columns([1, 5, 2])

            with col1:
                st.metric(
                    "Rank",
                    i,
                )

            with col2:
                st.markdown(
                    f"### {result.candidate_name}"
                )

                st.write(
                    result.summary
                )

            with col3:
                st.metric(
                    "Fit",
                    f"{result.overall_score:.0f}/100",
                )

                st.write(
                    f"**{result.recommendation}**"
                )

            # --------------------------------------
            # Strengths & Gaps
            # --------------------------------------

            strength_col, gap_col = st.columns(2)

            with strength_col:

                st.markdown("**💪 Strengths**")

                for strength in result.strengths:
                    st.write(
                        "• " + strength
                    )

            with gap_col:

                st.markdown("**⚠️ Gaps**")

                for gap in result.gaps:
                    st.write(
                        "• " + gap
                    )

            # --------------------------------------
            # Retrieved Evidence
            # --------------------------------------

            with st.expander("📚 Retrieved Evidence"):

                evidence_items = st.session_state.evidence.get(
                    result.candidate_name,
                    [],
                )

                if evidence_items:

                    for evidence in evidence_items:
                        st.info(evidence)

                else:

                    st.write(
                        "No retrieved evidence available."
                    )


    # ------------------------------------------------
    # Interview Question Generator
    # ------------------------------------------------

    st.subheader("🎤 Interview Question Generator")

    candidate_names = [
        result.candidate_name
        for result in ranked
    ]

    selected = st.selectbox(
        "Select Candidate",
        candidate_names,
    )


    if st.button(
        "🎯 Generate Interview Questions",
        use_container_width=True,
    ):

        # --------------------------------------------
        # Find Selected Candidate
        # --------------------------------------------

        selected_result = next(
            (
                result
                for result in ranked
                if result.candidate_name == selected
            ),
            None,
        )

        if selected_result is None:

            st.error(
                "Selected candidate result could not be found."
            )

        else:

            selected_evidence = (
                st.session_state.evidence.get(
                    selected,
                    [],
                )
            )

            try:

                with st.spinner(
                    "Generating interview questions with Gemini..."
                ):

                    questions = generate_interview_questions(
                        selected,
                        job,
                        selected_evidence,
                        selected_result,
                    )

                    st.session_state.questions = questions

            except Exception as exc:

                st.error(
                    "Gemini request failed. "
                    "Your free-tier quota may be temporarily exhausted."
                )

                st.exception(exc)


    # ------------------------------------------------
    # Display Interview Questions
    # ------------------------------------------------

    if st.session_state.questions is not None:

        questions = st.session_state.questions

        st.markdown("### Interview Questions")

        for i, question in enumerate(
            questions.questions,
            1,
        ):

            st.markdown(
                f"**{i}. {question}**"
            )

        st.markdown(
            "### Evaluation Focus"
        )

        for item in questions.evaluation_focus:

            st.write(
                "• " + item
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Human review is required for employment decisions. "
    "This demo does not use protected characteristics."
)
