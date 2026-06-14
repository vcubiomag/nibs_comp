import os
import numpy as np
from pathlib import Path
from typing import Any, TypedDict, cast

from s4l_v1 import model
from s4l_v1.simulation import emlf
from s4l_v1 import units
import XCore
import XCoreModeling
import s4l_v1.document as document
import pyvista as pv

HEADMODEL_PATH = Path("data/derivatives/headmodel_components")
CHARM_PATH = Path("data/derivatives/simnibs_charm")
SIM4LIFE_OUT = Path("data/derivatives/sim4life_magneto_quasistatic")


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


TMS_SITES = {
    "M1": [
        {
            "s4l_centre": "C3",
            "s4l_ydir": "CP5",
        }
    ],
    "DLPFC": [
        {
            "s4l_centre": "F3",
            "s4l_ydir": "FC5",
        }
    ],
    "SMA": [
        {
            "s4l_centre": "FCz",
            "s4l_ydir": "FC6",
        }
    ],
    "PPC": [
        {
            "s4l_centre": "P3",
            "s4l_ydir": "PO3",
        },
        {
            "s4l_centre": "P4",
            "s4l_ydir": "PO4",
        },
    ],
}

N_TURNS_PER_WING = 9
MEAN_WING_RADIUS_MM = 35.0
WIRE_WIDTH_MM = 1.9
WIRE_HEIGHT_MM = 6.0
EQUIV_WIRE_RADIUS_MM = float(np.sqrt(WIRE_WIDTH_MM * WIRE_HEIGHT_MM / np.pi))
COIL_STANDOFF_MM = 4.0
WING_GAP_MM = 4.0  # midline gap between outer-turn centerlines (Magstim D70, per SimNIBS Magstim_70mm_Fig8.ccd reference: wing centers at +/-45 mm, outer turn radius ~43 mm)
AMPLITUDE_PER_TURN_A = 1.0

MATERIAL_DATABASE_NAME = "IT'IS LF 5.0"
TISSUE_MATERIALS: dict[str, str] = {
    "white_matter": "Brain (White Matter)",
    "gray_matter": "Brain (Grey Matter)",
    "csf": "Cerebrospinal Fluid",
    "scalp": "Scalp",
    "eye_balls": "Eye (Vitreous Humor)",
    "cortical_bone": "Skull (Cortical)",
    "cancellous_bone": "Skull (Cancellous)",
    "blood": "Blood",
    "muscle": "Muscle",
}


def get_headmodel_component_paths(headmodel_comps_path: Path) -> HeadmodelComponents:
    paths_dict = {
        key: headmodel_comps_path / f"{key}_solid.stl"
        for key in HeadmodelComponents.__annotations__.keys()
    }
    return cast(HeadmodelComponents, paths_dict)


def read_charm_eeg_positions(m2m_path: Path) -> dict[str, np.ndarray]:
    eeg_dir = m2m_path / "eeg_positions"

    try:
        csv_file = sorted(eeg_dir.glob("EEG10-10*.csv"))[0]
    except IndexError:
        raise FileNotFoundError(f"No EEG10-10*.csv found in {eeg_dir}")

    positions: dict[str, np.ndarray] = {}
    with open(csv_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]

            if len(parts) >= 5:
                try:
                    # 3. Use slicing for cleaner float conversion
                    positions[parts[4]] = np.array([float(x) for x in parts[1:4]])
                except ValueError:
                    pass

    return positions


def create_figure8_coil(
    center_mm: Any,
    scalp_normal: Any,
    ydir_pos_mm: Any,
    mean_radius_mm: float = MEAN_WING_RADIUS_MM,
    n_turns: int = N_TURNS_PER_WING,
    wire_width_mm: float = WIRE_WIDTH_MM,
    standoff_mm: float = COIL_STANDOFF_MM,
    wing_gap_mm: float = WING_GAP_MM,
    n_points: int = 64,
) -> tuple[list[Any], list[Any]]:
    z_hat = scalp_normal / np.linalg.norm(scalp_normal)
    y_raw = ydir_pos_mm - center_mm
    y_proj = y_raw - np.dot(y_raw, z_hat) * z_hat
    y_hat = y_proj / np.linalg.norm(y_proj)
    x_hat = np.cross(y_hat, z_hat)
    x_hat = x_hat / np.linalg.norm(x_hat)

    # Lift whole coil off the scalp along the outward normal before laying
    # out the two wings, so every turn sits at the same stand-off height.
    coil_origin = center_mm + standoff_mm * z_hat

    radii = mean_radius_mm + (np.arange(n_turns) - (n_turns - 1) / 2.0) * wire_width_mm
    # Push wing centers apart by the outer-turn radius plus half the midline
    # gap, so the outermost turns of the two wings clear each other instead
    # of being tangent (a tangent layout would model a short between wings).
    # The ~4 mm default matches the SimNIBS Magstim_70mm_Fig8.ccd reference
    # model (wing centers at +/-45 mm, outer turn radius ~43 mm).
    wing_offset_mm = float(radii.max()) + wing_gap_mm / 2.0

    loop1_center = coil_origin + wing_offset_mm * x_hat
    loop2_center = coil_origin - wing_offset_mm * x_hat

    def _circle(lc: Any, r: float) -> list[model.Vec3]:
        # Wings lie in the scalp tangent plane (x_hat, y_hat); z_hat is the
        # scalp normal and must NOT appear in the circle parameterization or
        # the loops stand perpendicular to the scalp like fins.
        ts = np.linspace(0, 2 * np.pi, n_points + 1)
        return [
            model.Vec3(
                float(lc[0] + r * (np.cos(t) * x_hat[0] + np.sin(t) * y_hat[0])),
                float(lc[1] + r * (np.cos(t) * x_hat[1] + np.sin(t) * y_hat[1])),
                float(lc[2] + r * (np.cos(t) * x_hat[2] + np.sin(t) * y_hat[2])),
            )
            for t in ts
        ]

    wing1_turns = [model.CreatePolyLine(_circle(loop1_center, float(r))) for r in radii]
    wing2_turns = [model.CreatePolyLine(_circle(loop2_center, float(r))) for r in radii]
    return wing1_turns, wing2_turns


def import_headmodel(
    subject_id: str, headmodel_components: HeadmodelComponents
) -> None:
    model.SetLengthUnits(units.MilliMeter)
    subject_model_group = model.CreateGroup(subject_id)

    # model.Import yields a TriangleMesh (surface) for STL; the rectilinear MQS
    # voxeler only tags surface-intersecting voxels for surfaces, leaving tissue
    # interiors at background material. Convert each mesh to an ACIS Body so the
    # voxeler fills the volume. ConvertToSolidOptions has no public constructor;
    # rely on the built-in defaults (MergeCoincidentNodes=True).
    for tissue_name, component_path in headmodel_components.items():
        component_path = str(cast(Path, component_path).absolute())
        tri_mesh = model.Import(component_path)[0]
        bodies = XCoreModeling.ConvertToSolid(cast(Any, tri_mesh))
        if not bodies:
            raise RuntimeError(f"ConvertToSolid produced no bodies for {tissue_name}")
        bodies[0].Name = tissue_name
        subject_model_group.Add(bodies[0])
        for i, extra in enumerate(bodies[1:], start=1):
            extra.Name = f"{tissue_name}_part{i}"
            subject_model_group.Add(extra)
        tri_mesh.Delete()
        print(f"Finished importing: {tissue_name}")


def setup_simulation(sim_label: str, wing1_turns, wing2_turns) -> None:
    print(f"Setting up Simulation: {sim_label}")
    import s4l_v1.materials.database as mat_db

    simulation = emlf.MagnetoQuasiStaticSimulation()
    simulation.Name = sim_label

    simulation.SetupSettings.Frequency = 3200, units.Hz

    simulation_entities = []

    lf_db = mat_db.ITISLF
    if lf_db is None:
        raise RuntimeError("IT'IS LF material database not available")

    for tissue_name, tissue_material in TISSUE_MATERIALS.items():
        material_settings = emlf.MaterialSettings()
        comp = model.AllEntities()[tissue_name]
        mat = lf_db[tissue_material]
        simulation.LinkMaterialWithDatabase(material_settings, mat)
        simulation.Add(material_settings, [comp])

        simulation_entities.append(comp)

    for i, turn in enumerate(wing1_turns):
        cs = simulation.AddCurrentSourceSettings([turn])
        cs.Name = f"Wing1_Turn{i + 1:02d}"
        cs.Amplitude = AMPLITUDE_PER_TURN_A, units.Ampere
        cs.Radius = EQUIV_WIRE_RADIUS_MM, units.MilliMeter
        cs.IsDirectionReverted = False

        simulation_entities.append(turn)

    for i, turn in enumerate(wing2_turns):
        cs = simulation.AddCurrentSourceSettings([turn])
        cs.Name = f"Wing2_Turn{i + 1:02d}"
        cs.Amplitude = AMPLITUDE_PER_TURN_A, units.Ampere
        cs.Radius = EQUIV_WIRE_RADIUS_MM, units.MilliMeter
        cs.IsDirectionReverted = True

        simulation_entities.append(turn)

    sensor_settings = simulation.AddOverallFieldSensorSettings()
    sensor_settings.RecordEField = True
    sensor_settings.RecordHField = False
    sensor_settings.RecordVectorPotentialField = False

    manual_grid_settings = simulation.AddManualGridSettings(simulation_entities)
    # TODO: tune with new GPU
    # manual_grid_settings.MaxStep = np.array([0.5, 0.5, 0.5]), units.MilliMeters
    # manual_grid_settings.Resolution = np.array([0.5, 0.5, 0.5]), units.MilliMeters
    manual_grid_settings.MaxStep = np.array([1.0, 1.0, 1.0]), units.MilliMeters
    manual_grid_settings.Resolution = np.array([3.0, 3.0, 3.0]), units.MilliMeters

    solver_settings = simulation.SolverSettings
    solver_settings.PredefinedTolerances = (
        solver_settings.PredefinedTolerances.enum.High
    )
    solver_settings.NumberOfProcesses = 1
    solver_settings.NumberOfThreads = 1

    auto_voxeler = next(
        s
        for s in simulation.AllSettings
        if isinstance(s, emlf.AutomaticVoxelerSettings)
    )
    simulation.Add(auto_voxeler, simulation_entities)

    print("Updating Simulation Materials")
    simulation.UpdateAllMaterials()

    print("Updating Simulation Grid")
    simulation.UpdateGrid()

    document.AllSimulations.Add(simulation)


def main():
    XCore.GetOrCreateConsoleApp()

    subject_list = sorted(HEADMODEL_PATH.glob("sub-*"))
    for subject in subject_list:
        subject_id = subject.name
        output_path = SIM4LIFE_OUT / f"{subject_id}_1"

        if output_path.is_dir() and any(output_path.iterdir()):
            print(f"SKIPPED: {subject_id} (output exists)")
            continue
        os.makedirs(output_path, exist_ok=True)

        smash_path = str((output_path / f"{subject_id}.smash").absolute())

        document.New()

        headmodel_comps = get_headmodel_component_paths(subject)

        m2m_path = CHARM_PATH / subject_id / f"m2m_{subject_id}"
        eeg_positions = read_charm_eeg_positions(m2m_path)
        scalp_mesh_n = pv.read(str(headmodel_comps["scalp"])).compute_normals(
            consistent_normals=True, flip_normals=False
        )

        import_headmodel(subject_id, headmodel_comps)

        document.SaveDocumentAs(smash_path)

        for site_name, site_configs in TMS_SITES.items():
            for site_config in site_configs:
                electrode = site_config["s4l_centre"]
                ydir_label = site_config["s4l_ydir"]

                center_mm: Any = eeg_positions[electrode]
                ydir_mm: Any = eeg_positions[ydir_label]

                closest_idx = scalp_mesh_n.find_closest_point(center_mm)
                scalp_normal: Any = scalp_mesh_n.point_normals[closest_idx]

                wing1_turns, wing2_turns = create_figure8_coil(
                    center_mm, scalp_normal, ydir_mm
                )

                sim_label = f"{subject_id}_{site_name}"
                if len(site_configs) > 1:
                    sim_label += f"_{electrode}"

                coil_group = model.CreateGroup(f"Coil_{sim_label}")
                wing1_group = model.CreateGroup("Wing1")
                wing2_group = model.CreateGroup("Wing2")
                coil_group.Add(wing1_group)
                coil_group.Add(wing2_group)
                for i, turn in enumerate(wing1_turns):
                    turn.Name = f"Wing1_Turn{i + 1:02d}"
                    wing1_group.Add(turn)
                for i, turn in enumerate(wing2_turns):
                    turn.Name = f"Wing2_Turn{i + 1:02d}"
                    wing2_group.Add(turn)

                setup_simulation(sim_label, wing1_turns, wing2_turns)

                document.SaveDocument()

            break

        # # License allows only one concurrent simulation, so run sequentially.
        # # wait=True blocks until each sim finishes and results are on disk.
        # sims = list(document.AllSimulations)
        # for i, sim in enumerate(sims, start=1):
        #     print(f"RUNNING: {sim.Name} ({i}/{len(sims)})")
        #     sim.RunSimulation(wait=True)
        #     document.SaveDocument()

        # print(f"DONE: {subject_id} ({len(sims)} simulations)")


if __name__ == "__main__":
    main()
