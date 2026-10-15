# Figures

Install `requirements-lock.txt` in a separate plotting environment. Arial is required for the publication layout.

```text
python reproduction/code/plotting/render_all.py
python reproduction/code/plotting/render_all.py --figure fig-08
```

The renderer resolves each figure's bundled inputs through `reproduction/figures/index.json`. It writes PDF, SVG, PNG and plot tables to that figure's `analysis/rebuilt/` directory. Paths derive from the repository location; external market data and a local research configuration are not required for rendering. Publication assets remain unchanged.

`validate_regeneration.py` compares regenerated tables, text, dimensions and pixels with the publication assets. Install `requirements-validation.txt` before running it. Font and graphics-library differences can affect pixels without changing plotted values.
