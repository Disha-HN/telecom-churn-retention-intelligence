from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "app.py"


def load_app():
    return AppTest.from_file(str(APP_PATH))


def test_app_starts_without_exception():
    app = load_app().run(timeout=15)

    assert not app.exception


def test_app_has_navigation():
    app = load_app().run(timeout=15)

    assert len(app.sidebar.radio) == 1


def test_executive_dashboard_loads():
    app = load_app().run(timeout=15)

    assert app.title
    assert not app.exception


def test_customer_analysis_page_loads():
    app = load_app().run(timeout=15)

    app.sidebar.radio[0].set_value(
        "👤 Customer Analysis"
    ).run(timeout=15)

    assert not app.exception


def test_high_risk_page_loads():
    app = load_app().run(timeout=15)

    app.sidebar.radio[0].set_value(
        "🚨 High-Risk Customers"
    ).run(timeout=15)

    assert not app.exception


def test_retention_intelligence_page_loads():
    app = load_app().run(timeout=15)

    app.sidebar.radio[0].set_value(
        "🧠 Retention Intelligence"
    ).run(timeout=15)

    assert not app.exception


def test_ai_agent_page_loads():
    app = load_app().run(timeout=15)

    app.sidebar.radio[0].set_value(
        "🤖 AI Retention Agent"
    ).run(timeout=15)

    assert not app.exception


def test_retention_operations_page_loads():
    app = load_app().run(timeout=15)

    app.sidebar.radio[0].set_value(
        "📈 Retention Operations"
    ).run(timeout=15)

    assert not app.exception