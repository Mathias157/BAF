# Agent Configuration for BAF (Balmorel-Antares Soft-Linking Framework)

## Build/Test Commands
- **Run tests**: `pytest tests/` or `python tests/balmorel_antares_comparison.py`
- **Type check**: `pyright src/`
- **Install deps**: `pixi install` (uses pixi.toml for environment management)
- **Run preprocessing**: `pixi run preprocessing`
- **Run Balmorel**: `pixi run balmorel --scenario_name=<name>`
- **Run Antares**: `pixi run antares`

## Code Style
- **Python version**: 3.12.9 (strict)
- **Imports**: Standard library first, then external packages (pandas, numpy, matplotlib, geopandas, click, gams), then local modules (pybalmorel, Workflow.Functions)
- **Headers**: Each file starts with docstring (title, description, date, @author: Mathias Berg Rosendal)
- **Structure**: Section headers with `#%% ------------------------------- ###` format
- **CLI patterns**: Use click for command-line interfaces with groups and commands
- **Naming**: CamelCase for classes/configs (Config, MainResults), snake_case for functions/variables
- **Type hints**: Use pyright for type checking
- **Error handling**: Use custom ErrorLog class from Workflow.Functions.GeneralHelperFunctions
- **File paths**: Use pathlib.Path for cross-platform compatibility
- **Config**: Use configparser with Config.ini for runtime configuration