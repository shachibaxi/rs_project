"""SkillBridge - Streamlit entry point.

Run with:  streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from src.data_loader import DATASET_URL, DatasetError
from ui import navigation
from ui.components import inject_css
from ui.state import get_engine
from ui.views import explore, home, learning_path_page, profile_page, recommendations, skill_gap_page

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="SkillBridge", page_icon=str(ASSETS / "icon.svg"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")
inject_css()

try:
    get_engine()
except DatasetError as error:
    st.error(f"The course dataset could not be loaded: {error}")
    st.markdown(f"Download it from [Kaggle]({DATASET_URL}) and place `Coursera.csv` in `data/raw/`.")
    st.stop()

pages = [
    navigation.page(home.render, title="Home", icon=None, url_path="overview", default=True),
    navigation.page(profile_page.render, title="1 · Profile", icon=None, url_path="profile"),
    navigation.page(skill_gap_page.render, title="2 · Skill gaps", icon=None, url_path="skill-gap"),
    navigation.page(recommendations.render, title="3 · Courses for you", icon=None, url_path="recommendations"),
    navigation.page(learning_path_page.render, title="4 · Learning path", icon=None, url_path="learning-path"),
    navigation.page(explore.render, title="Explore all courses", icon=None, url_path="explore"),
]
st.navigation(pages, position="top").run()
