<img src="assets/header.gif" width="100%">
<hr style="color:#808080;">

**Albus** is a lightweight package that implements a library of metrics for evaluating neural emulators and generative priors of dynamical systems, spanning physics and biogeochemistry, from ocean to atmosphere. It covers several kinds of assessment: the **raw predictive skill of a neural emulator**, the **calibration of a prior learned by a generative model**, and the **physical soundness of the dynamics a model has learned**, checked against known domain-specific indicators (e.g. physics, biogeochemistry).

<hr style="color:#808080;">
<p align="center"><b>T U T O R I A L</b></p>
<hr style="color:#808080;">

A self-contained tutorial is available as a Jupyter notebook. It loads a sample dataset (a 7-day, 4-member forecast of oxygen in the Black Sea, at the surface and at 9 m depth, along with the corresponding ground truth), then demonstrates every metric in the library.

➜ [`notebook/demo.ipynb`](notebook/demo.ipynb)

<hr style="color:#808080;">
<p align="center"><b>I N S T A L L A T I O N</b></p>
<hr style="color:#808080;">

- If you want the **latest version**, install it directly from GitHub:

    ```
    pip install git+https://github.com/VikVador/albus
    ```

- If you want a **local editable** install with all optional dependencies (notebooks, linting):

    ```
    conda create -n albus python=3.11
    conda activate albus
    ```

    then

    ```
    pip install --editable '.[all]' --extra-index-url https://download.pytorch.org/whl/cu121
    ```
    Optionally, install the pre-commit hooks to automatically detect code issues before each commit:

    ```
    pre-commit install --config pre-commit.yml
    ```