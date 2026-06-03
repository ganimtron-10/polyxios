from fury import actor, window

import numpy as np
import polyxios
from polyxios._element_types import ELEMENT_TYPES


# --- Element type code constants -----------------------------------------
_TRIANGLE = ELEMENT_TYPES["triangle"]
_QUAD = ELEMENT_TYPES["quad"]
_LINE = ELEMENT_TYPES["line"]
_POLY_LINE = ELEMENT_TYPES["poly_line"]
_VERTEX = ELEMENT_TYPES["vertex"]
_POLY_VERTEX = ELEMENT_TYPES["poly_vertex"]
_TRIANGLE_STRIP = ELEMENT_TYPES["triangle_strip"]
_POLYGON = ELEMENT_TYPES["polygon"]


def extract_triangles(poly: polyxios.PolyData) -> np.ndarray | None:
    """Return an (N, 3) int32 array of triangle indices from poly.

    Handles:
    - triangle cells   → taken directly
    - quad cells       → split into 2 triangles each
    - polygon cells    → fan-triangulated from the first vertex
    - triangle_strips  → converted to triangles
    """
    faces: list[np.ndarray] = []

    for i in range(len(poly.element_types)):
        etype = int(poly.element_types[i])
        s = int(poly.offsets[i])
        e = int(poly.offsets[i + 1])
        cell = poly.connectivity[s:e]

        if etype == _TRIANGLE:
            faces.append(cell.reshape(1, 3))

        elif etype == _QUAD:
            # Split quad ABCD → ABC + ACD
            a, b, c, d = cell[0], cell[1], cell[2], cell[3]
            faces.append(np.array([[a, b, c], [a, c, d]], dtype=np.int32))

        elif etype == _POLYGON:
            # Fan-triangulate: A-B-C, A-C-D, A-D-E, ...
            n = len(cell)
            if n >= 3:
                tris = np.stack(
                    [
                        np.full(n - 2, cell[0], dtype=np.int32),
                        cell[1 : n - 1].astype(np.int32),
                        cell[2:n].astype(np.int32),
                    ],
                    axis=1,
                )
                faces.append(tris)

        elif etype == _TRIANGLE_STRIP:
            # Convert strip to triangles: alternating winding
            n = len(cell)
            for j in range(n - 2):
                if j % 2 == 0:
                    faces.append(
                        np.array([[cell[j], cell[j + 1], cell[j + 2]]], dtype=np.int32)
                    )
                else:
                    faces.append(
                        np.array([[cell[j], cell[j + 2], cell[j + 1]]], dtype=np.int32)
                    )

    if not faces:
        return None
    return np.concatenate(faces, axis=0).astype(np.int32)


def extract_lines(poly: polyxios.PolyData) -> list[np.ndarray]:
    """Return a list of vertex-coordinate arrays, one per line/poly_line cell."""
    lines = []
    for i in range(len(poly.element_types)):
        etype = int(poly.element_types[i])
        if etype in (_LINE, _POLY_LINE):
            s = int(poly.offsets[i])
            e = int(poly.offsets[i + 1])
            indices = poly.connectivity[s:e]
            lines.append(poly.vertices[indices])
    return lines


def extract_points(poly: polyxios.PolyData) -> np.ndarray | None:
    """Return vertex coordinates for vertex/poly_vertex cells."""
    pts: list[np.ndarray] = []
    for i in range(len(poly.element_types)):
        etype = int(poly.element_types[i])
        if etype in (_VERTEX, _POLY_VERTEX):
            s = int(poly.offsets[i])
            e = int(poly.offsets[i + 1])
            indices = poly.connectivity[s:e]
            pts.append(poly.vertices[indices])
    if not pts:
        return None
    return np.concatenate(pts, axis=0)


def build_actors(poly: polyxios.PolyData) -> list:
    """Build FURY actors for a PolyData, choosing the right representation
    based on the element types present in the mesh."""
    actors = []

    # --- Surface (triangles / quads / polygons / strips) ---
    faces = extract_triangles(poly)
    if faces is not None and len(faces) > 0:
        print(f"  Surface: {len(faces)} triangles")
        mesh_actor = actor.surface(poly.vertices, faces, colors=(0.85, 0.85, 0.85))
        actors.append(mesh_actor)

    # --- Lines ---
    line_list = extract_lines(poly)
    if line_list:
        print(f"  Lines: {len(line_list)} line(s)")
        line_actor = actor.line(line_list, colors=(0.2, 0.8, 1.0))
        actors.append(line_actor)

    # --- Points ---
    pts = extract_points(poly)
    if pts is not None and len(pts) > 0:
        print(f"  Points: {len(pts)} point(s)")
        dot_actor = actor.dots(pts, color=(1.0, 0.5, 0.0), dot_size=5)
        actors.append(dot_actor)

    return actors


def main():
    """
    Example demonstrating how to use polyxios to fetch and read a remote VTK
    model, then render it using the fury library.

    Supports UNSTRUCTURED_GRID and POLYDATA datasets, including triangles,
    quads, polygons, triangle_strips, lines, and points.
    """
    model_name = "vtk.vtk"
    print(f"Fetching '{model_name}'...")
    model_path = polyxios.fetch(model_name)
    print(f"Model successfully fetched to: {model_path}")

    print("Loading the model using polyxios...")
    poly = polyxios.read(model_path)
    print(f"  Vertices : {poly.vertices.shape[0]}")
    print(f"  Cells    : {len(poly.element_types)}")

    print("Building scene actors...")
    scene_actors = build_actors(poly)
    if not scene_actors:
        print("No renderable geometry found in this model.")
        return

    print("Rendering the model...")
    window.show(scene_actors)

    print("Saving the model using polyxios...")
    polyxios.write(poly, f"polyxios_{model_name}")


if __name__ == "__main__":
    main()
