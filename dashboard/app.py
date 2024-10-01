# This file generated by Quarto; do not edit by hand.
# shiny_mode: core

from __future__ import annotations

from pathlib import Path
from shiny import App, Inputs, Outputs, Session, ui

import plotly.express as px
from shiny import render, reactive, ui, req
# df = px.data.gapminder()

import seaborn as sns
penguins = sns.load_dataset("penguins")

# ========================================================================




def server(input: Inputs, output: Outputs, session: Session) -> None:
    from shiny.express import render, ui

    global n
    n = 10

    # ui.input_select("x", "Variable:",
    #                 choices=["bill_length_mm", "bill_depth_mm"])
    # ui.input_select("dist", "Distribution:", choices=["hist", "kde"])
    # ui.input_checkbox("rug", "Show rug marks", value = False)


    ui.input_numeric(id="selected_topic", 
                     label="label",
                     value=0, 
                     min=0, 
                     max=n, step=1)
                 
                 
    ui.input_action_button(id="prev_topic",
    label="Previous Topic")
    ui.input_action_button(id="next_topic", label="Next Topic")
    ui.input_action_button(id="clear_topic", label="Clear Topic")


    # ========================================================================

    import pickle, pyLDAvis
    import panel as pn
    with open('ldavis_prepared', 'rb') as f:
        prepared_data = pickle.load(f)
    

    # pyLDAvis.display(prepared_data)

    # ========================================================================

    @render.plot
    def displot():
        sns.displot(
            data=penguins, hue="species", multiple="stack",
            x=input.x(), rug=input.rug(), kind=input.dist())

    # ========================================================================

    ##| include: FALSE
    ui.input_slider("n", "Select maximum value n:", min=1, max=100, value=10)

    # ========================================================================



    return None


_static_assets = ["quarto_files"]
_static_assets = {"/" + sa: Path(__file__).parent / sa for sa in _static_assets}

app = App(
    Path(__file__).parent / "quarto.html",
    server,
    static_assets=_static_assets,
)
