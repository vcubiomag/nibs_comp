from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict, cast
from cyclopts import App
import logging
import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn, SpinnerColumn

DATASET_PATH = Path("data")
TMS_SITES = {
    "M1": [
        {
            "simnibs_centre": "C3",
            "simnibs_ydir": "CP5",
            "s4l_electrode": "C3",
            "s4l_ydir": "CP5",
        }
    ],
    "DLPFC": [
        {
            "simnibs_centre": "F3",
            "simnibs_ydir": "FC5",
            "s4l_electrode": "F3",
            "s4l_ydir": "FC5",
        }
    ],
    "SMA": [
        {
            "simnibs_centre": "FCz",
            "simnibs_ydir": "FC6",
            "s4l_electrode": "FCz",
            "s4l_ydir": "FC6",
        }
    ],
    "PPC": [
        {
            "simnibs_centre": "P3",
            "simnibs_ydir": "PO3",
            "s4l_electrode": "P3",
            "s4l_ydir": "PO3",
        },
        {
            "simnibs_centre": "P4",
            "simnibs_ydir": "PO4",
            "s4l_electrode": "P4",
            "s4l_ydir": "PO4",
        },
    ],
}

MAX_WORKERS = 4

# Maps each tissue STL name to its corresponding material in Sim4Life's
# built-in IT'IS database, used by LinkMaterialWithDatabase().
TISSUE_MATERIALS: dict[str, str] = {
    "white_matter":    "Brain (White Matter)",
    "gray_matter":     "Brain (Grey Matter)",
    "csf":             "Cerebrospinal Fluid",
    "scalp":           "Scalp",
    "eye_balls":       "Eye (Vitreous Humor)",
    "cortical_bone":   "Skull (Cortical)",
    "cancellous_bone": "Skull (Cancellous)",
    "blood":           "Blood",
    "muscle":          "Muscle",
}

console = Console(stderr=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True, show_path=False)
    ],
)
logger = logging.getLogger(__name__)


def _silence_simnibs_console_logging() -> None:
    # FileHandler subclasses StreamHandler, so exclude it explicitly to keep
    # the per-run log file that SESSION._set_logger writes into pathfem.
    simnibs_logger = logging.getLogger("simnibs")
    for handler in list(simnibs_logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            simnibs_logger.removeHandler(handler)
    simnibs_logger.propagate = False


app = App()

class HeadmodelComponents(TypedDict):
    white_matter: Path
    gray_matter: Path
    csf: Path
    scalp: Path
    eye_balls: Path
    cortical_bone: Path
    cancellous_bone: Path
    blood: Path
    muscle: Path


def get_headmodel_component_paths(headmodel_comps_path: Path) -> HeadmodelComponents:
    paths_dict = {
        key: headmodel_comps_path / f"{key}_solid.stl"
        for key in HeadmodelComponents.__annotations__.keys()
    }

    return cast(HeadmodelComponents, paths_dict)


def _read_charm_eeg_positions(m2m_path: Path) -> "dict[str, Any]":
    """Read EEG 10-10 electrode positions from a CHARM output directory.

    CHARM writes electrode positions in mm in subject RAS space.
    Returns a dict mapping electrode label → numpy array([x, y, z]) in mm.
    """
    import numpy as np

    eeg_dir = m2m_path / "eeg_positions"
    csv_files = sorted(eeg_dir.glob("EEG10-10*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No EEG10-10*.csv found in {eeg_dir}")

    positions: dict[str, Any] = {}
    with open(csv_files[0]) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    positions[parts[0]] = np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )
                except ValueError:
                    pass  # Skip header rows or malformed entries

    return positions


def _create_figure8_coil(
    center_mm: "Any",
    scalp_normal: "Any",
    ydir_pos_mm: "Any",
    radius_mm: float = 35.0,
    n_points: int = 64,
) -> "tuple[Any, Any]":
    """Build a Magstim 70 mm Figure-8 coil as two circular polylines (mm).

    The coil lies tangent to the scalp surface at *center_mm*, with the coil
    handle axis pointing toward *ydir_pos_mm*.  Wing 1 carries positive
    current; Wing 2 carries negative current and should have
    ``IsDirectionReverted=True`` in its ``CurrentSourceSettings``.

    Args:
        center_mm: 3-element array — coil center on the scalp surface (mm).
        scalp_normal: Unit outward normal of the scalp at that point.
        ydir_pos_mm: 3-element array — ydir reference electrode position (mm).
        radius_mm: Loop radius (default 35 mm → 70 mm wing diameter).
        n_points: Points per circular loop (default 64).

    Returns:
        (loop1_entity, loop2_entity) — two ``model.PolyLine`` entities.
    """
    import numpy as np
    import s4l_v1.model as model
    from s4l_v1.model import Vec3

    # Coil coordinate frame
    z_hat = scalp_normal / np.linalg.norm(scalp_normal)           # coil normal
    y_raw = ydir_pos_mm - center_mm
    y_proj = y_raw - np.dot(y_raw, z_hat) * z_hat                 # onto tangent plane
    y_hat = y_proj / np.linalg.norm(y_proj)                       # handle direction
    x_hat = np.cross(y_hat, z_hat)
    x_hat = x_hat / np.linalg.norm(x_hat)

    # Wing centers: each displaced by one radius along the handle axis
    loop1_center = center_mm + radius_mm * x_hat
    loop2_center = center_mm - radius_mm * x_hat

    def _circle(lc: "Any") -> "list[Vec3]":
        ts = np.linspace(0, 2 * np.pi, n_points + 1)
        return [
            Vec3(
                float(lc[0] + radius_mm * (np.cos(t) * y_hat[0] + np.sin(t) * z_hat[0])),
                float(lc[1] + radius_mm * (np.cos(t) * y_hat[1] + np.sin(t) * z_hat[1])),
                float(lc[2] + radius_mm * (np.cos(t) * y_hat[2] + np.sin(t) * z_hat[2])),
            )
            for t in ts
        ]

    loop1 = model.CreatePolyLine(_circle(loop1_center))
    loop2 = model.CreatePolyLine(_circle(loop2_center))
    return loop1, loop2


def run_simnibs_worker(subject_id: str, charm_dir_path: str, output_path: str) -> dict:
    try:
        from simnibs import sim_struct, run_simnibs
    except ImportError:
        raise ImportError("Unable to import SimNIBS")

    _silence_simnibs_console_logging()

    if os.path.isdir(output_path) and any(os.listdir(output_path)):
        return {"subject": subject_id, "status": "SKIPPED"}

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

    return {"subject": subject_id, "status": "SUCCESS"}


def run_sim4life_simulation(
    subject_id: str,
    m2m_path: Path,
    headmodel_comps_path: Path,
    output_path: Path,
    site_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run Sim4Life MQS TMS simulations for all TMS sites for one subject.

    Each TMS site config produces a separate ``MagnetoQuasiStaticSimulation``
    inside a single Sim4Life project (.smash) file.  Coil positions are read
    from the CHARM EEG position files so that the electrode locations exactly
    match those used by SimNIBS, enabling a direct comparison.

    Materials are assigned from the Sim4Life IT'IS database by name, so
    frequency-appropriate conductivity values are resolved automatically.

    The coil is modelled as two circular polylines (Magstim 70 mm Figure-8):
    - Wing 1: 35 mm-radius circle, positive current (1 A normalised).
    - Wing 2: 35 mm-radius circle, negative current (IsDirectionReverted=True).
    Both wings are positioned tangent to the scalp at the target electrode.

    Args:
        subject_id: Subject identifier, e.g. ``"sub-01"``.
        m2m_path: Path to the ``m2m_{subject_id}/`` CHARM output directory.
        headmodel_comps_path: Directory containing ``*_solid.stl`` tissue files.
        output_path: Directory where the ``.smash`` project file is written.
        site_callback: Optional callable invoked with the site name after each
            site's simulation completes (used to advance a progress bar).

    Returns:
        ``{"subject": subject_id, "status": "SUCCESS" | "SKIPPED"}``
    """
    try:
        import numpy as np
        import pyvista as pv
        import s4l_v1.document as document
        import s4l_v1.model as model
        import s4l_v1.simulation.emlf as emlf
        import s4l_v1.units as units
        from s4l_v1 import Unit  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(f"Unable to import Sim4Life API or pyvista: {exc}") from exc

    if os.path.isdir(output_path) and any(os.listdir(output_path)):
        return {"subject": subject_id, "status": "SKIPPED"}

    os.makedirs(output_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Project setup
    # ------------------------------------------------------------------
    document.NewDocument()
    model.SetLengthUnits(Unit("mm"))  # CHARM STL files use millimetres

    # ------------------------------------------------------------------
    # Import tissue solid bodies (shared by all site simulations)
    # ------------------------------------------------------------------
    components = get_headmodel_component_paths(headmodel_comps_path)
    tissue_entities: dict[str, Any] = {}
    for tissue_name, stl_path in components.items():
        entity: Any = model.Import(str(stl_path))
        entity.Name = tissue_name
        tissue_entities[tissue_name] = entity

    # ------------------------------------------------------------------
    # CHARM EEG electrode positions (mm, same coordinate frame as STLs)
    # ------------------------------------------------------------------
    eeg_positions = _read_charm_eeg_positions(m2m_path)

    # ------------------------------------------------------------------
    # Scalp mesh for outward surface normal computation at each electrode
    # ------------------------------------------------------------------
    scalp_mesh = pv.read(str(headmodel_comps_path / "scalp_solid.stl"))
    scalp_mesh_n = scalp_mesh.compute_normals(
        consistent_normals=True, flip_normals=False
    )

    # ------------------------------------------------------------------
    # One MQS simulation per TMS site config
    # ------------------------------------------------------------------
    for site_name, site_configs in TMS_SITES.items():
        for config in site_configs:
            electrode = config["s4l_electrode"]
            ydir_label = config["s4l_ydir"]

            center_mm: Any = eeg_positions[electrode]
            ydir_mm: Any = eeg_positions[ydir_label]

            # Outward scalp surface normal at the target electrode
            closest_idx = scalp_mesh_n.find_closest_point(center_mm)
            scalp_normal: Any = scalp_mesh_n.point_normals[closest_idx]

            # Figure-8 coil geometry: two circular polylines in mm
            loop1, loop2 = _create_figure8_coil(center_mm, scalp_normal, ydir_mm)
            sim_label = f"{subject_id}_{site_name}"
            # Disambiguate sites that have multiple configs (e.g. PPC → P3 and P4)
            if len(site_configs) > 1:
                sim_label += f"_{electrode}"
            loop1.Name = f"Coil_Wing1_{sim_label}"
            loop2.Name = f"Coil_Wing2_{sim_label}"

            # -- Simulation -----------------------------------------------
            sim = emlf.MagnetoQuasiStaticSimulation()
            sim.Name = sim_label
            document.AllSimulations.Add(sim)

            sim.SetupSettings.Frequency = 3200.0, units.Hz

            # -- Materials from IT'IS database ----------------------------
            for tissue_name, tissue_entity in tissue_entities.items():
                mat = sim.AddMaterialSettings([tissue_entity])
                mat.Name = tissue_name
                sim.LinkMaterialWithDatabase(mat, TISSUE_MATERIALS[tissue_name])

            # -- Coil current sources (Figure-8) --------------------------
            coil1 = sim.AddCurrentSourceSettings([loop1])
            coil1.Name = f"Wing1_{electrode}"
            coil1.Amplitude = 1.0, units.Amperes
            coil1.Radius = 1.0, Unit("mm")
            coil1.IsDirectionReverted = False

            coil2 = sim.AddCurrentSourceSettings([loop2])
            coil2.Name = f"Wing2_{electrode}"
            coil2.Amplitude = 1.0, units.Amperes
            coil2.Radius = 1.0, Unit("mm")
            coil2.IsDirectionReverted = True

            # -- E-field sensor -------------------------------------------
            sensor = sim.AddOverallFieldSensorSettings()
            sensor.RecordEField = True
            sensor.RecordHField = False

            # -- Automatic voxelisation / grid ----------------------------
            sim.AddAutomaticGridSettings()

            # -- Solver ---------------------------------------------------
            sim.SolverSettings.PredefinedTolerances = "High"

            # -- Run ------------------------------------------------------
            sim.RunSimulation(wait=True)

            if site_callback is not None:
                site_callback(site_name)

    # ------------------------------------------------------------------
    # Save project
    # ------------------------------------------------------------------
    document.SaveDocumentAs(str(output_path / f"{subject_id}.smash"))

    return {"subject": subject_id, "status": "SUCCESS"}


@app.default
def main(framework: Literal["simnibs", "sim4life", "ansys"], max_workers: int = MAX_WORKERS):
    logger.info(f"Generating E-Fields using Framework: {framework}")

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
        total_site_configs = sum(len(configs) for configs in TMS_SITES.values())
        task_subjects = progress.add_task("[cyan]Processing subjects...", total=len(subject_list))
        task_sites = progress.add_task("[green]Simulating sites...", total=total_site_configs, visible=False)

        if framework == "simnibs":
            jobs = []
            for subject in subject_list:
                subject_id = subject.name
                charm_dir_path = DATASET_PATH / "derivatives" / "simnibs_charm" / subject_id / f"m2m_{subject_id}"
                output_path = DATASET_PATH / "derivatives" / "simnibs_fem" / subject_id
                jobs.append((subject_id, str(charm_dir_path), str(output_path)))

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_simnibs_worker, sub_id, charm_dir, out_dir): sub_id
                    for sub_id, charm_dir, out_dir in jobs
                }

                for future in as_completed(futures):
                    sub_id = futures[future]
                    try:
                        result = future.result()
                        logger.info(f"{result['status']}: {sub_id}")
                    except Exception:
                        logger.exception(f"FAILED: {sub_id}")
                    progress.advance(task_subjects)

        else:
            import XCore
            XCore.GetOrCreateConsoleApp()

            for subject in subject_list:
                subject_id = subject.name

                if framework == "sim4life":
                    progress.reset(
                        task_sites,
                        description=f"[green]Simulating sites for {subject_id}...",
                        total=total_site_configs,
                        visible=True,
                    )

                    m2m_path = (
                        DATASET_PATH / "derivatives" / "simnibs_charm"
                        / subject_id / f"m2m_{subject_id}"
                    )
                    headmodel_comps_path = (
                        DATASET_PATH / "derivatives" / "headmodel_components" / subject_id
                    )
                    output_path = DATASET_PATH / "derivatives" / "sim4life_fem" / subject_id

                    try:
                        result = run_sim4life_simulation(
                            subject_id,
                            m2m_path,
                            headmodel_comps_path,
                            output_path,
                            site_callback=lambda _: progress.advance(task_sites),
                        )
                        logger.info(f"{result['status']}: {subject_id}")
                    except Exception:
                        logger.exception(f"FAILED: {subject_id}")

                progress.advance(task_subjects)

        progress.update(task_sites, visible=False)


if __name__ == "__main__":
    app()
