"""Local Streamlit presentation layer; prediction logic lives in predict_fight_v2.py."""
from datetime import date
import html
import logging
from pathlib import Path
import runpy

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
SIDE_COLORS = {"A": "#F27581", "B": "#68B9F4", "history": "#B4BDCE"}
METHOD_LABELS = {"KO_TKO": "KO / TKO", "SUBMISSION": "Submission", "DECISION": "Decision"}


def probability_chart(rows, series, label):
    """Render engine probabilities on a fixed 0–100% scale; never rescale estimates.

    Each row is (label, probabilities). Visible values and an accessible text
    description keep the chart readable without relying on color or hover.
    """
    description = "; ".join(
        f"{name}: " + ", ".join(f"{series[side]} {value:.1%}" for side, value in values.items())
        for name, values in rows
    )
    parts = [f'<div class="prob-chart" role="img" aria-label="{html.escape(label + ": " + description, quote=True)}">']
    parts.append('<div class="chart-scale">Scale: 0% to 100%</div>')
    for name, values in rows:
        parts.append(f'<div class="chart-group"><div class="chart-label">{html.escape(name)}</div>')
        for side, value in values.items():
            parts.append(
                f'<div class="chart-row"><span class="series-label">{html.escape(series[side])}</span>'
                f'<div class="chart-track"><div class="chart-fill" '
                f'style="width:{value * 100:.8f}%;background:{SIDE_COLORS[side]}"></div></div>'
                f'<span class="chart-value">{value:.1%}</span></div>'
            )
        parts.append('</div>')
    parts.append('</div>')
    return "".join(parts)


def render_method_chart(result, key, title):
    series = {"A": f"A · {result['fighter_a']}", "B": f"B · {result['fighter_b']}"}
    rows = [(label, {side: result[key][side][method] for side in series})
            for method, label in METHOD_LABELS.items()]
    st.markdown(probability_chart(rows, series, title), unsafe_allow_html=True)


def render_styles():
    st.markdown("""<style>
    .stApp {background:radial-gradient(ellipse at top right,#241923 0%,#0b1018 58%);}
    .block-container {max-width:1160px;padding-top:2.8rem;padding-bottom:3rem;}
    h1 {letter-spacing:-.045em;font-weight:850;line-height:1.12;}
    h2,h3 {letter-spacing:-.025em;}
    .eyebrow {font-size:.72rem;font-weight:750;letter-spacing:.16em;color:#bcc8d9;}
    .hero-note {max-width:690px;color:#bcc8d9;font-size:1.05rem;line-height:1.7;margin:.3rem 0 1.4rem;}
    .status-strip {display:flex;gap:.65rem;flex-wrap:wrap;margin:1rem 0 1.5rem;}
    .status-chip {border:1px solid #354052;border-radius:6px;padding:.45rem .7rem;
        color:#d5deeb;background:#131c29;font-size:.78rem;}
    .setup-title {font-size:1.15rem;font-weight:750;margin-bottom:.4rem;}
    .corner-label {font-size:.75rem;font-weight:800;letter-spacing:.13em;margin:.5rem 0;}
    .red-corner {color:#f27581;} .blue-corner {color:#68b9f4;}
    .matchup-grid {display:grid;grid-template-columns:minmax(0,1fr) 38px minmax(0,1fr);
        align-items:center;gap:1rem;margin:1.2rem 0;}
    .fight-card {background:linear-gradient(135deg,#261b26,#131c29);border:1px solid #414052;
        border-top:3px solid #f27581;border-radius:12px;padding:1.6rem;min-width:0;}
    .fight-card.side-b {background:linear-gradient(225deg,#16283b,#131c29);border-top-color:#68b9f4;}
    .fight-card h2 {font-size:clamp(1.2rem,3vw,1.9rem);color:#f1f5f9;margin:.5rem 0;
        padding:0;overflow-wrap:anywhere;min-height:2.5em;}
    .probability {font-size:clamp(2.5rem,6vw,4rem);line-height:1.2;font-weight:850;letter-spacing:-.06em;}
    .side-a .probability {color:#f27581;} .side-b .probability {color:#68b9f4;}
    .card-note {font-size:.82rem;color:#bcc8d9;margin-top:.5rem;}
    .versus {text-align:center;font-size:.8rem;font-weight:800;color:#aab9cc;}
    .win-share {display:flex;height:12px;border-radius:6px;overflow:hidden;background:#263346;margin:.5rem 0 1.7rem;}
    .prob-chart {background:#111a27;border:1px solid #334155;border-radius:10px;padding:1.2rem;margin:.7rem 0 1rem;}
    .chart-scale {color:#aebed1;font-size:.78rem;line-height:1.5;margin-bottom:1rem;}
    .chart-group + .chart-group {margin-top:1.15rem;}
    .chart-label {color:#edf2fa;font-size:.9rem;font-weight:700;margin-bottom:.5rem;}
    .chart-row {display:grid;grid-template-columns:minmax(95px,1fr) minmax(0,2fr) 55px;gap:.8rem;align-items:center;margin:.5rem 0;}
    .series-label {color:#becbdd;font-size:.78rem;line-height:1.5;min-width:0;overflow-wrap:anywhere;}
    .chart-track {height:12px;border-radius:3px;background:repeating-linear-gradient(to right,#263449 0%,#263449 calc(25% - 1px),#405069 25%);overflow:hidden;}
    .chart-fill {height:100%;border-radius:3px;}
    .chart-value {color:#f1f5f9;font-size:.85rem;font-weight:700;text-align:right;font-variant-numeric:tabular-nums;}
    [data-testid="stAlert"] p {line-height:1.65;}
    [data-testid="stCaptionContainer"] {line-height:1.6;}
    button[kind="primary"] {font-weight:800;letter-spacing:.08em;min-height:3.1rem;}
    @media(max-width:640px){
        .block-container {padding:1.5rem 1rem 2rem;}
        h1 {font-size:2.2rem;}
        .matchup-grid {grid-template-columns:minmax(0,1fr);gap:.55rem;}
        .fight-card {padding:1.2rem;}
        .fight-card h2 {min-height:0;}
        .versus {padding:.15rem;}
        .chart-row {grid-template-columns:minmax(80px,1fr) minmax(0,1.5fr) 48px;gap:.45rem;}
        .prob-chart {padding:.9rem;}
    }
    </style>""", unsafe_allow_html=True)


def snapshot_signature():
    """Invalidate resources when local artifacts or engine code change."""
    paths = [ROOT / path for path in (
        "models/ufc_winner_d_v2_0_0.joblib", "models/ufc_winner_d_v2_0_0_metadata.json",
        "models/ufc_method_m_v1_0_0.joblib", "models/ufc_method_m_v1_0_0_metadata.json",
        "models/ufc_round_prior_v1_0_0.json", "src/prediction/round_prior.py",
        "src/prediction/build_v2_matchup_features.py",
        "data/processed/fights_dated.csv", "src/prediction/predict_fight_v2.py",
        "src/prediction/build_matchup_features.py", "src/data/fighter_catalog.py")]
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


@st.cache_resource(show_spinner=False)
def load_engine(signature):
    engine = runpy.run_path(str(ROOT / "src/prediction/predict_fight_v2.py"))
    models, prior = engine["load_artifacts"]()
    metadata = models["winner"][1]
    engine["BUILD"]["default_builder"]()
    return engine, metadata


@st.cache_data(show_spinner=False)
def fighter_names(signature):
    engine, _ = load_engine(signature)
    return sorted(engine["BUILD"]["default_builder"]().base.names.values(), key=str.casefold)


@st.cache_resource(show_spinner=False)
def load_catalog(signature):
    return runpy.run_path(str(ROOT / "src/data/fighter_catalog.py"))["FighterCatalog"]()


def clear_division_selection():
    st.session_state["fighter_a"] = None
    st.session_state["fighter_b"] = None
    clear_result()


def run_prediction(a, b, day, signature, scheduled_rounds):
    """Only validate UI inputs and delegate; no copied feature/model formulas."""
    if not a or not b:
        raise ValueError("Select both fighters before requesting probabilities.")
    if " ".join(a.split()).casefold() == " ".join(b.split()).casefold():
        raise ValueError("Choose two different fighters.")
    engine, _ = load_engine(signature)
    return engine["predict_fight_v2"](a, b, day, scheduled_rounds)


def clear_result():
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("prediction_inputs", None)


def swap_fighters():
    a, b = st.session_state.get("fighter_a"), st.session_state.get("fighter_b")
    st.session_state["fighter_a"], st.session_state["fighter_b"] = b, a
    clear_result()


def render_result(result):
    st.subheader("WIN PROBABILITY")
    st.caption(f"Prediction date: {result['prediction_date']} | {result['scheduled_rounds']} scheduled rounds | Supported-outcome estimate")
    st.info("Probabilities are uncertain estimates for supported outcomes: KO/TKO, submission and decision. "
            "Draws, no contests, DQs and overturned outcomes have no probability mass.")
    cards = ['<div class="matchup-grid">']
    for side, label in (("a", "A · RED CORNER"), ("b", "B · BLUE CORNER")):
        probability = result["winner_probabilities"][side.upper()]
        if side == "b":
            cards.append('<div class="versus">VS</div>')
        cards.append(f'<div class="fight-card side-{side}"><div class="eyebrow">{label}</div>'
                     f'<h2>{html.escape(result[f"fighter_{side}"])}</h2>'
                     f'<div class="probability">{probability:.1%}</div>'
                     '<div class="card-note">Estimated win probability</div></div>')
    cards.append('</div>')
    st.markdown("".join(cards), unsafe_allow_html=True)
    win = result["winner_probabilities"]
    st.markdown(
        '<div class="win-share" role="img" '
        f'aria-label="Fighter A {win["A"]:.1%}; Fighter B {win["B"]:.1%}">'
        f'<div style="width:{win["A"] * 100:.8f}%;background:{SIDE_COLORS["A"]}"></div>'
        f'<div style="width:{win["B"] * 100:.8f}%;background:{SIDE_COLORS["B"]}"></div></div>',
        unsafe_allow_html=True,
    )
    for title, key, explanation in (
        ("METHOD GIVEN WIN", "method_given_win", "Conditional on that fighter winning; each fighter's column sums to 100%."),
        ("OVERALL WIN-BY-METHOD", "joint_winner_method", "Win probability multiplied by method given win; all six outcomes sum to 100%."),
    ):
        st.subheader(title)
        st.caption(explanation)
        render_method_chart(result, key, title)
        with st.expander(f"{title.capitalize()} · probability table"):
            st.table({"Method": list(METHOD_LABELS.values()),
                      result["fighter_a"]: [f"{result[key]['A'][m]:.2%}" for m in METHOD_LABELS],
                      result["fighter_b"]: [f"{result[key]['B'][m]:.2%}" for m in METHOD_LABELS]})
    st.subheader("FINISH ROUND PRIOR")
    st.caption("Historical population estimates conditional on a KO/TKO or Submission finish and the selected schedule. Not ML predictions or fighter-specific estimates. Decisions have no finish-round estimate.")
    prior = result["round_given_finish"]
    displayed_rounds = range(1, result["scheduled_rounds"] + 1)
    rounds = [(f"Round {i}", {"history": prior["probabilities"][f"ROUND_{i}"]}) for i in displayed_rounds]
    st.markdown(probability_chart(rounds, {"history": "Historical finishes"}, "Finish round prior"), unsafe_allow_html=True)
    with st.expander("Finish round prior · probability table"):
        st.table({"Round": [f"R{i}" for i in displayed_rounds],
                  "Historical probability": [f"{prior['probabilities'][f'ROUND_{i}']:.2%}" for i in displayed_rounds]})
    st.caption(f"Based on {prior['sample_count']:,} historical finishes through {result['metadata']['cutoff']}. Display rounding can cause totals to differ slightly from 100%.")
    with st.expander("Scientific details & matchup warnings", expanded=False):
        st.caption("Read these alongside the estimates. Missing history and unsupported outcomes affect interpretation.")
        for message in result.get("warnings", []):
            st.warning(message)


def render_model_info(metadata):
    with st.expander("About the model"):
        st.write(f"Winner D · Logistic Regression · {metadata['feature_count']} historical/activity, Elo and defensive matchup features · symmetric fighter-order design.")
        st.write("METHOD-M estimates method conditional on each fighter winning. The round table is a separate historical population prior.")
        st.write("Fighter histories use only information strictly before the selected date. Draws and no contests are excluded from the binary probabilities.")
        st.write(f"The frozen model itself was trained through **{metadata['training_cutoff']}**. Earlier selected dates are historical sanity checks, not forecasts from a model trained at that time.")
        metrics = metadata["reference_backtest"]["aggregate"]["pooled"]
        labels = [("Accuracy", "accuracy", ".2%"), ("Balanced accuracy", "balanced_accuracy", ".2%"),
                  ("ROC-AUC", "roc_auc", ".4f"), ("Log loss", "log_loss", ".4f"), ("Brier score", "brier_score", ".4f")]
        for column, (label, key, fmt) in zip(st.columns(5), labels):
            column.metric(label, format(metrics[key], fmt))
        st.caption("Historical out-of-time evaluation of winner configuration D, not guarantees of future performance or an evaluation of the final all-history fit. Previously inspected periods are not pristine holdouts.")
    with st.expander("Limitations", expanded=False):
        st.markdown("""- The model learns only from observed UFC history; missing data can affect estimates.
- Fighter identity and name matching have limitations.
- A long gap does not establish injury, retirement, or its cause.
- Probabilities are estimates, not certainty.
- Age, injuries, betting odds, and external live information are not used.""")


def main():
    st.set_page_config(page_title="UFC Fight Predictor AI", page_icon=":material/sports_mma:", layout="wide")
    render_styles()
    st.markdown('<div class="eyebrow">THE MATCHUP LAB / HISTORICAL UFC ANALYTICS</div>', unsafe_allow_html=True)
    st.title("UFC Fight Predictor AI")
    st.markdown('<div class="hero-note">Build a matchup. Explore the probabilities. '
                'Compare win chances, paths to victory and historical finish timing from observed UFC history.</div>', unsafe_allow_html=True)
    try:
        signature = snapshot_signature()
        with st.spinner("Loading local fighter histories and models…"):
            _, metadata = load_engine(signature)
            names = fighter_names(signature)
            catalog = load_catalog(signature)
        if len(names) < 2:
            raise ValueError("Not enough supported fighters.")
    except FileNotFoundError:
        st.error("Required local model or history files are missing. Restore the winner D, METHOD-M, round prior, metadata, and processed history before using the app.")
        st.stop()
    except Exception:
        LOGGER.exception("Unable to initialize the local prediction engine")
        st.error("The local model and history could not be loaded or validated. Check that the matching versioned artifacts and dataset are available, then restart the app.")
        st.stop()
    st.markdown('<div class="status-strip">'
                '<span class="status-chip">Winner D / METHOD-M / Round prior</span>'
                f'<span class="status-chip">{len(names):,} supported names</span>'
                f'<span class="status-chip">Snapshot: {html.escape(metadata["training_cutoff"])}</span>'
                '<span class="status-chip">Local history · No live data</span></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="eyebrow">01 / BUILD YOUR MATCHUP</div>'
                    '<div class="setup-title">Choose a division, then your fighters</div>', unsafe_allow_html=True)
        # The date control remains after fighter selectors; its session value
        # limits memberships on each rerun, preventing future division history.
        cutoff = st.session_state.get("prediction_date") or date.today()
        divisions = catalog.get_weight_classes(as_of_date=cutoff)
        if st.session_state.get("weight_class") not in divisions:
            st.session_state["weight_class"] = None
        division = st.selectbox("Weight class", divisions, index=None, key="weight_class",
                                placeholder="Choose an observed division", on_change=clear_division_selection)
        st.caption("Historical division membership, not a current roster. Retirement filtering is pending verified status data; no fighter is classified from inactivity.")
        st.caption(f"Only division observations strictly before {cutoff} are shown. Undated observations and first-time division entrants are unavailable for that historical date.")
        if division == "Unspecified historical division":
            st.info("These historical labels do not identify a division; this group does not establish a shared weight class.")
        eligible = catalog.get_active_fighters_for_weight_class(division, as_of_date=cutoff) if division else []
        names = [name for name in eligible if name in names]
        for key in ("fighter_a", "fighter_b"):
            if st.session_state.get(key) not in names:
                st.session_state[key] = None
                clear_result()
        left, right = st.columns(2)
        a_options = [name for name in names if name != st.session_state.get("fighter_b")]
        b_options = [name for name in names if name != st.session_state.get("fighter_a")]
        with left:
            st.markdown('<div class="corner-label red-corner">A / RED CORNER</div>', unsafe_allow_html=True)
            a = st.selectbox("Fighter A", a_options, index=None, placeholder="Search for a fighter…", key="fighter_a", on_change=clear_result, disabled=not division)
        with right:
            st.markdown('<div class="corner-label blue-corner">B / BLUE CORNER</div>', unsafe_allow_html=True)
            b = st.selectbox("Fighter B", b_options, index=None, placeholder="Search for an opponent…", key="fighter_b", on_change=clear_result, disabled=not division)
        st.button("SWAP FIGHTERS", key="swap", on_click=swap_fighters, disabled=not(a and b))
        if division and len(names) < 2:
            st.info("Fewer than two supported fighters have observed membership in this division before the selected date. Choose another division or a later date.")
        timing, schedule = st.columns(2)
        with timing:
            day = st.date_input("Prediction date", value=date.today(), min_value=date(1900,1,1),
                                max_value=date(2100,12,31), key="prediction_date", on_change=clear_result)
        with schedule:
            scheduled_rounds = st.selectbox("Scheduled rounds", [3, 5], key="scheduled_rounds", on_change=clear_result,
                                           help="Select the announced schedule. This is not inferred from the ending round.")
        st.caption("Only fight information strictly before this date constructs fighter histories; same-day fights are excluded. The model itself is trained on all available eligible history.")
        st.caption("Use the announced schedule; it is not inferred from the fighters or ending round.")
        if day and str(day) <= metadata["training_cutoff"]:
            st.info("Historical inspection: these probabilities are sanity checks, not out-of-sample forecasts.")
        elif day:
            st.info(f"Local history ends {metadata['training_cutoff']}. More recent fights may be absent.")
        if st.button("PREDICT FIGHT", key="predict", type="primary", use_container_width=True, disabled=not(a and b and day)):
            clear_result()
            try:
                with st.spinner("Constructing pre-fight history…"):
                    result = run_prediction(a,b,day,signature,scheduled_rounds)
                st.session_state["prediction_result"] = result
                st.session_state["prediction_inputs"] = (a,b,str(day),scheduled_rounds,signature)
            except ValueError as exc:
                st.error(f"Unable to build this matchup: {exc}")
            except FileNotFoundError:
                st.error("A required local model or history file is missing. Restore it and restart the app.")
            except Exception:
                LOGGER.exception("Prediction failed")
                st.error("The probability estimate could not be produced. Check the local artifacts or try another matchup; no result was generated.")
    # Results are session-local and shown only for their exact input snapshot.
    if st.session_state.get("prediction_inputs") == (a,b,str(day),scheduled_rounds,signature):
        render_result(st.session_state["prediction_result"])
    else:
        st.info("Choose two fighters and a date, then select PREDICT FIGHT.")
    render_model_info(metadata)
    st.caption("Independent research project. No official UFC affiliation or branding assets.")


if __name__ == "__main__":
    main()
