from pathlib import Path
import numpy as np
import pymeshfix
import pyvista
from rich.progress import track

HEADMODELS_DIR = Path("data/derivatives/simnibs_charm")
OUTPUT_DIR = Path("data/derivatives/headmodel_components")

SURFACE_KEY_TO_LABEL = {
    1001: "white_matter",
    1002: "gray_matter",
    1003: "csf",
    1005: "scalp",
    1006: "eye_balls",
    1007: "cortical_bone",
    1008: "cancellous_bone",
    1009: "blood",
    1010: "muscle",
    1099: "internal_air",
}

VOLUME_KEY_TO_LABEL = {
    1: "white_matter",
    2: "gray_matter",
    3: "csf",
    5: "scalp",
    6: "eye_balls",
    7: "cortical_bone",
    8: "cancellous_bone",
    9: "blood",
    10: "muscle",
}

VTK_TRIANGLE = 5
VTK_TETRA = 10


def extract_surfaces(mesh: pyvista.UnstructuredGrid) -> dict[str, pyvista.PolyData]:
    phys = mesh.cell_data["gmsh:physical"]
    tri_mask = mesh.celltypes == VTK_TRIANGLE

    surfaces = {}
    for tag, label in SURFACE_KEY_TO_LABEL.items():
        mask = tri_mask & (phys == tag)
        surface = mesh.extract_cells(mask).extract_surface(algorithm="dataset_surface")
        surfaces[label] = surface

    return surfaces


def extract_volumes(mesh: pyvista.UnstructuredGrid) -> dict[str, pyvista.UnstructuredGrid]:
    phys = mesh.cell_data["gmsh:physical"]
    tet_mask = mesh.celltypes == VTK_TETRA

    volumes = {}
    for tag, label in VOLUME_KEY_TO_LABEL.items():
        mask = tet_mask & (phys == tag)
        volume = mesh.extract_cells(mask)
        volumes[label] = volume

    return volumes


def repair_surface(surface: pyvista.PolyData) -> pyvista.PolyData:
    fixer = pymeshfix.MeshFix(surface)
    fixer.repair(joincomp=True, remove_smallest_components=False)
    return fixer.mesh


def volume_to_solid_surface(volume: pyvista.UnstructuredGrid) -> pyvista.PolyData:
    boundary = volume.extract_surface(algorithm="dataset_surface")
    if not boundary.is_manifold:
        boundary = repair_surface(boundary)
    return boundary


def main():
    headmodel_subjects = list(HEADMODELS_DIR.glob("sub-*"))
    for headmodel_sub_dir in track(headmodel_subjects):
        sub_id = headmodel_sub_dir.name
        headmodel_mesh_path = headmodel_sub_dir / f"m2m_{sub_id}" / f"{sub_id}.msh"
        output_path = OUTPUT_DIR / sub_id
        output_path.mkdir(exist_ok=True, parents=True)

        mesh = pyvista.read(headmodel_mesh_path)

        surfaces = extract_surfaces(mesh)
        for label, surface in surfaces.items():
            repaired_surface = surface
            if not surface.is_manifold:
                repaired_surface = repair_surface(surface)

            print(f"[surface] {label}: orig_manifold={surface.is_manifold}, repaired_manifold={repaired_surface.is_manifold}")
            surface.save(output_path / f"{label}_orig.stl")
            repaired_surface.save(output_path / f"{label}_repaired.stl")

        volumes = extract_volumes(mesh)
        for label, volume in volumes.items():
            solid_surface = volume_to_solid_surface(volume)
            print(f"[solid]   {label}: n_tets={volume.n_cells}, boundary_faces={solid_surface.n_cells}, manifold={solid_surface.is_manifold}")
            solid_surface.save(output_path / f"{label}_solid.stl")


if __name__ == "__main__":
    main()
