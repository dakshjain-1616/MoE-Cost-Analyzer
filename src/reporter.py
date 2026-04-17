"""Rich terminal reporter for benchmark results."""

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

from .analyzer import ModelStats

console = Console()


def _color_pct(val: str) -> Text:
    """Color a percentage string: negative = green (improvement), positive = red (regression)."""
    if val == "N/A":
        return Text(val, style="dim")
    try:
        num = float(val.replace("%", "").replace("+", ""))
    except ValueError:
        return Text(val)
    if num < -5:
        return Text(val, style="bold green")
    elif num < 0:
        return Text(val, style="green")
    elif num < 5:
        return Text(val, style="yellow")
    else:
        return Text(val, style="bold red")


def print_matrix(matrix: pd.DataFrame, dense_model: str, moe_model: str) -> None:
    """Print the decision matrix as a Rich table."""
    table = Table(
        title="[bold]Dense vs MoE — Decision Matrix[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold", min_width=22)
    table.add_column(f"Dense\n{dense_model}", justify="right", min_width=18)
    table.add_column(f"MoE\n{moe_model}", justify="right", min_width=18)
    table.add_column("MoE vs Dense", justify="right", min_width=14)

    for _, row in matrix.iterrows():
        table.add_row(
            row["Metric"],
            row["Dense"],
            row["MoE"],
            _color_pct(row["MoE vs Dense"]),
        )

    console.print()
    console.print(table)


def print_recommendation(
    recommendation: str,
    dense_stats: ModelStats,
    moe_stats: ModelStats,
) -> None:
    """Print the recommendation with color coding."""
    console.print()

    # Summary line
    if dense_stats.avg_latency_ms > 0 and dense_stats.avg_cost_per_query > 0:
        lat_pct = (moe_stats.avg_latency_ms - dense_stats.avg_latency_ms) / dense_stats.avg_latency_ms * 100
        cost_pct = (moe_stats.avg_cost_per_query - dense_stats.avg_cost_per_query) / dense_stats.avg_cost_per_query * 100
        console.print(
            f"[dim]Summary:[/dim] MoE is [bold]{abs(cost_pct):.1f}% cheaper[/bold] and "
            f"[bold]{abs(lat_pct):.1f}% {'faster' if lat_pct < 0 else 'slower'}[/bold] than the dense model."
        )

    # Recommendation
    if recommendation.startswith("USE MoE"):
        style = "bold green"
        prefix = "[green]✓[/green]"
    elif recommendation.startswith("MARGINAL"):
        style = "bold yellow"
        prefix = "[yellow]~[/yellow]"
    else:
        style = "bold red"
        prefix = "[red]✗[/red]"

    console.print()
    console.print(f"{prefix} [bold]Recommendation:[/bold] [{style}]{recommendation}[/{style}]")
    console.print()
