from pathlib import Path
from cyclopts import App
import logging
import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

DATASET_PATH = Path("data")
TMS_SITES = {
    "M1": [{"simnibs_centre": "C3", "simnibs_ydir": "CP5"}],
    "DLPFC": [{"simnibs_centre": "F3", "simnibs_ydir": "FC5"}],
    "SMA": [{"simnibs_centre": "FCz", "simnibs_ydir": "FC6"}],
    "PPC": [
        {"simnibs_centre": "P3", "simnibs_ydir": "PO3"},
        {"simnibs_centre": "P4", "simnibs_ydir": "PO4"},
    ],
}

MAX_WORKERS = 4

console = Console(stderr=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)


def _silence_simnibs_console_logging() -> None:
    # FileHandler subclasses StreamHandler, so exclude it explicitly to keep
    # the per-run log file that SESSION._set_logger writes into pathfem.
    simnibs_logger = logging.getLogger("simnibs")
    for handler in list(simnibs_logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            simnibs_logger.removeHandler(handler)
    simnibs_logger.propagate = False


app = App()


def run_simnibs_worker(charm_dir_path: str, output_path: str) -> str:
    try:
        from simnibs import sim_struct, run_simnibs
    except ImportError:
        raise ImportError("Unable to import SimNIBS")

    _silence_simnibs_console_logging()

    if os.path.isdir(output_path) and any(os.listdir(output_path)):
        return "SKIPPED"

    os.makedirs(output_path, exist_ok=True)

    s = sim_struct.SESSION()
    s.subpath = charm_dir_path
    s.pathfem = output_path
    s.open_in_gmsh = False
    s.map_to_surf = True
    s.map_to_fsavg = True

    tmslist = s.add_tmslist()
    tmslist.fnamecoil = os.path.join("legacy_and_other", "Magstim_70mm_Fig8.ccd")

    for sim_site in itertools.chain.from_iterable(TMS_SITES.values()):
        pos = tmslist.add_position()
        pos.centre = sim_site["simnibs_centre"]
        pos.pos_ydir = sim_site["simnibs_ydir"]

    run_simnibs(s)

    return "SUCCESS"


@app.default
def main():
    subject_list = sorted(DATASET_PATH.glob("sub-*"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_subjects = progress.add_task(
            "[cyan]Processing subjects...", total=len(subject_list)
        )

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    run_simnibs_worker,
                    str(
                        DATASET_PATH
                        / "derivatives"
                        / "simnibs_charm"
                        / subject.name
                        / f"m2m_{subject.name}"
                    ),
                    str(DATASET_PATH / "derivatives" / "simnibs_fem" / subject.name),
                ): subject.name
                for subject in subject_list
            }

            for future in as_completed(futures):
                sub_id = futures[future]
                try:
                    logger.info(f"{future.result()}: {sub_id}")
                except Exception:
                    logger.exception(f"FAILED: {sub_id}")
                progress.advance(task_subjects)


if __name__ == "__main__":
    app()
