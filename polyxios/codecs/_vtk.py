"""High-performance, release-driven, dependency-free VTK codec for Polyxios."""

from pathlib import Path
from typing import Any

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    POLYXIOS_TO_VTK,
    VTK_TO_POLYXIOS,
)
from polyxios._types import PolyData
from polyxios.exceptions import (
    CodecError,
    IndexOverflowError,
    UnknownElementTypeError,
)
from polyxios.validate import validate_header

EXTENSION: str = ".vtk"

MAX_CONNECTIVITY_INDEX_V42: int = 2**31 - 1
MAX_CONNECTIVITY_INDEX_V51: int = 2**63 - 1

_VTK_DTYPE_MAP: dict[str, str] = {
    "float": "f4",
    "double": "f8",
    "int": "i4",
    "long": "i8",
    "unsigned_int": "u4",
    "unsigned_long": "u8",
    "short": "i2",
    "unsigned_short": "u2",
    "char": "i1",
    "unsigned_char": "u1",
}


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a VTK legacy file into an explicit PolyData instance."""
    path = Path(path)
    file_size = path.stat().st_size

    version, is_binary, dataset, offset = _parse_vtk_header(path)

    if is_binary:
        return _read_binary_body(path, offset, file_size, dataset, version)
    else:
        return _read_ascii_body(path, offset, file_size, dataset, version)


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to a VTK legacy unstructured grid file."""
    path = Path(path)
    binary: bool = bool(opts.get("binary", False))
    vtk_version: str = str(opts.get("vtk_version", "4.2"))

    max_allowed = (
        MAX_CONNECTIVITY_INDEX_V51
        if vtk_version == "5.1"
        else MAX_CONNECTIVITY_INDEX_V42
    )
    if poly.connectivity.size > 0 and int(poly.connectivity.max()) > max_allowed:
        raise IndexOverflowError("vtk", max_allowed, int(poly.connectivity.max()))

    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)

    with open(path, "wb") as fh:
        fh.write(f"# vtk DataFile Version {vtk_version}\n".encode())
        fh.write(b"Written by polyxios\n")
        fh.write(b"BINARY\n" if binary else b"ASCII\n")
        fh.write(b"DATASET UNSTRUCTURED_GRID\n")

        fh.write(f"POINTS {n_verts} double\n".encode())
        if binary:
            fh.write(poly.vertices.astype(np.dtype(">f8")).tobytes())
        else:
            for v in poly.vertices:
                fh.write(f"{v[0]:.10g} {v[1]:.10g} {v[2]:.10g}\n".encode())

        if vtk_version == "5.1":
            _write_cells_v51(poly, fh, binary)
        else:
            _write_cells_v42(poly, fh, binary)

        fh.write(f"CELL_TYPES {n_elems}\n".encode())
        vtk_types = np.array(
            [_polyxios_to_vtk_code(poly.element_types[i]) for i in range(n_elems)],
            dtype=np.int32,
        )
        if binary:
            fh.write(vtk_types.astype(np.dtype(">i4")).tobytes())
        else:
            fh.write((" ".join(str(t) for t in vtk_types) + "\n").encode())

        if poly.vertex_attrs:
            fh.write(f"POINT_DATA {n_verts}\n".encode())
            for name, arr in poly.vertex_attrs.items():
                _write_vtk_array(name, arr, fh, binary)

        if poly.element_attrs:
            fh.write(f"CELL_DATA {n_elems}\n".encode())
            for name, arr in poly.element_attrs.items():
                _write_vtk_array(name, arr, fh, binary)


def _parse_vtk_header(path: Path) -> tuple[str, bool, str, int]:
    version = "4.2"
    is_binary = False
    dataset = ""
    offset = 0

    with open(path, "rb") as fh:
        for _ in range(100):
            raw = fh.readline()
            if not raw:
                break
            offset += len(raw)
            line = raw.decode("ascii", errors="replace").strip().upper()
            if not line:
                continue
            if line.startswith("# VTK"):
                parts = line.split()
                if len(parts) >= 5:
                    version = parts[4]
            elif "ASCII" in line or "BINARY" in line:
                is_binary = "BINARY" in line
            elif line.startswith("DATASET"):
                dataset = line.split()[1]
                break
            elif line.startswith("FIELD"):
                dataset = "FIELD"
                offset -= len(raw)
                break

    if not dataset:
        raise CodecError(
            "Invalid VTK format. Header contains no DATASET or FIELD indicator."
        )

    return version, is_binary, dataset, offset


def _consume_optional_newline(fh) -> None:
    pos = fh.tell()
    b = fh.read(1)
    if b != b"\n":
        fh.seek(pos)


def _unpack_v42_cells(raw: np.ndarray, n_elems: int) -> tuple[np.ndarray, np.ndarray]:
    conn_list: list[int] = []
    off_list: list[int] = [0]
    idx = 0
    for _ in range(n_elems):
        cnt = int(raw[idx])
        idx += 1
        conn_list.extend(int(raw[idx + j]) for j in range(cnt))
        idx += cnt
        off_list.append(off_list[-1] + cnt)
    return np.array(conn_list, dtype=np.int32), np.array(off_list, dtype=np.int32)


def _build_structured_cells(
    nx: int, ny: int, nz: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nx > 1 and ny > 1 and nz > 1:
        K, J, I = np.meshgrid(
            np.arange(nz - 1), np.arange(ny - 1), np.arange(nx - 1), indexing="ij"
        )
        n0 = K * (nx * ny) + J * nx + I
        n1 = n0 + 1
        n2 = n0 + nx + 1
        n3 = n0 + nx
        n4 = n0 + (nx * ny)
        n5 = n4 + 1
        n6 = n4 + nx + 1
        n7 = n4 + nx
        conn = (
            np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=-1).ravel().astype(np.int32)
        )
        n_cells = (nx - 1) * (ny - 1) * (nz - 1)
        offsets = np.arange(0, n_cells * 8 + 1, 8, dtype=np.int32)
        types = np.full(n_cells, ELEMENT_TYPES["hexahedron"], dtype=np.uint8)
    elif nx > 1 and ny > 1:
        J, I = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
        n0 = J * nx + I
        n1 = n0 + 1
        n2 = n0 + nx + 1
        n3 = n0 + nx
        conn = np.stack([n0, n1, n2, n3], axis=-1).ravel().astype(np.int32)
        n_cells = (nx - 1) * (ny - 1)
        offsets = np.arange(0, n_cells * 4 + 1, 4, dtype=np.int32)
        types = np.full(n_cells, ELEMENT_TYPES["quad"], dtype=np.uint8)
    else:
        conn = np.array([], dtype=np.int32)
        offsets = np.array([0], dtype=np.int32)
        types = np.array([], dtype=np.uint8)
    return conn, offsets, types


def _finalize_geometry(
    dataset: str,
    dims: list[int],
    origin: list[float],
    spacing: list[float],
    x_coords,
    y_coords,
    z_coords,
    vertices,
    conn_list,
    off_list,
    type_list,
):
    if dataset == "STRUCTURED_POINTS" and dims[0] > 0:
        nx, ny, nz = dims
        x = origin[0] + np.arange(nx) * spacing[0]
        y = origin[1] + np.arange(ny) * spacing[1]
        z = origin[2] + np.arange(nz) * spacing[2]
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        vertices = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        c, o, t = _build_structured_cells(nx, ny, nz)
        conn_list = [c]
        off_list = [0] + o[1:].tolist()
        type_list = [t]

    elif dataset == "RECTILINEAR_GRID" and dims[0] > 0:
        nx, ny, nz = dims
        if x_coords is None:
            x_coords = np.arange(nx, dtype=np.float64)
        if y_coords is None:
            y_coords = np.arange(ny, dtype=np.float64)
        if z_coords is None:
            z_coords = np.arange(nz, dtype=np.float64)
        Z, Y, X = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
        vertices = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        c, o, t = _build_structured_cells(nx, ny, nz)
        conn_list = [c]
        off_list = [0] + o[1:].tolist()
        type_list = [t]

    elif dataset == "STRUCTURED_GRID" and dims[0] > 0:
        nx, ny, nz = dims
        c, o, t = _build_structured_cells(nx, ny, nz)
        conn_list = [c]
        off_list = [0] + o[1:].tolist()
        type_list = [t]

    final_conn = (
        np.concatenate(conn_list) if conn_list else np.array([], dtype=np.int32)
    )
    final_off = np.array(off_list, dtype=np.int32)
    final_types = (
        np.concatenate(type_list) if type_list else np.array([], dtype=np.uint8)
    )

    return vertices, final_conn, final_off, final_types


def _read_binary_body(
    path: Path, offset: int, file_size: int, dataset: str, version: str
) -> PolyData:
    vertices = np.zeros((0, 3), dtype=np.float64)
    conn_list: list[np.ndarray] = []
    off_list: list[int] = [0]
    type_list: list[np.ndarray] = []
    v_attrs: dict[str, np.ndarray] = {}
    e_attrs: dict[str, np.ndarray] = {}

    dims = [0, 0, 0]
    origin = [0.0, 0.0, 0.0]
    spacing = [1.0, 1.0, 1.0]
    x_coords = y_coords = z_coords = None
    target_dict = v_attrs

    n_verts = 0
    n_elems = 0

    with open(path, "rb") as fh:
        fh.seek(offset)
        while fh.tell() < file_size:
            pos = fh.tell()
            raw_line = fh.readline()
            if not raw_line:
                break
            line = raw_line.decode("ascii", errors="replace").strip().upper()
            if not line:
                continue

            parts = line.split()
            kw = parts[0]

            if kw == "DIMENSIONS":
                dims = [int(parts[1]), int(parts[2]), int(parts[3])]
                nx, ny, nz = dims
                n_verts = nx * ny * nz
                if nx > 1 and ny > 1 and nz > 1:
                    n_elems = (nx - 1) * (ny - 1) * (nz - 1)
                elif nx > 1 and ny > 1:
                    n_elems = (nx - 1) * (ny - 1)
            elif kw == "ORIGIN":
                origin = [float(parts[1]), float(parts[2]), float(parts[3])]
            elif kw in ("SPACING", "ASPECT_RATIO"):
                spacing = [float(parts[1]), float(parts[2]), float(parts[3])]
            elif kw == "POINTS":
                n_verts = int(parts[1])
                dtype_str = parts[2].lower() if len(parts) > 2 else "double"
                np_dt = ">f8" if dtype_str == "double" else ">f4"
                n_bytes = n_verts * 3 * np.dtype(np_dt).itemsize
                raw = np.frombuffer(fh.read(n_bytes), dtype=np_dt)
                vertices = raw.astype(np.float64).reshape(n_verts, 3)
                _consume_optional_newline(fh)
            elif kw in ("X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"):
                n_c = int(parts[1])
                dtype_str = parts[2].lower() if len(parts) > 2 else "double"
                np_dt = ">f8" if dtype_str == "double" else ">f4"
                n_bytes = n_c * np.dtype(np_dt).itemsize
                raw = np.frombuffer(fh.read(n_bytes), dtype=np_dt).astype(np.float64)
                _consume_optional_newline(fh)
                if kw == "X_COORDINATES":
                    x_coords = raw
                elif kw == "Y_COORDINATES":
                    y_coords = raw
                elif kw == "Z_COORDINATES":
                    z_coords = raw
            elif kw in ("CELLS", "POLYGONS", "LINES", "VERTICES", "TRIANGLE_STRIPS"):
                n_cells = int(parts[1])
                n_elems += n_cells
                total_size = int(parts[2])

                if kw == "CELLS" and version >= "5.1":
                    pos2 = fh.tell()
                    l2 = fh.readline().decode("ascii").strip().upper()
                    if l2.startswith("OFFSETS"):
                        n_bytes = (n_cells + 1) * 8
                        o_raw = np.frombuffer(fh.read(n_bytes), dtype=">i8").astype(
                            np.int32
                        )
                        _consume_optional_newline(fh)
                        fh.readline()
                        conn_size = o_raw[-1]
                        n_bytes = conn_size * 8
                        c_raw = np.frombuffer(fh.read(n_bytes), dtype=">i8").astype(
                            np.int32
                        )
                        _consume_optional_newline(fh)
                        conn_list.append(c_raw)
                        off_list.extend((o_raw[1:] + off_list[-1]).tolist())
                    else:
                        fh.seek(pos2)
                        n_bytes = total_size * 4
                        raw = np.frombuffer(fh.read(n_bytes), dtype=">i4").astype(
                            np.int32
                        )
                        _consume_optional_newline(fh)
                        c, o = _unpack_v42_cells(raw, n_cells)
                        conn_list.append(c)
                        off_list.extend((o[1:] + off_list[-1]).tolist())
                else:
                    n_bytes = total_size * 4
                    raw = np.frombuffer(fh.read(n_bytes), dtype=">i4").astype(np.int32)
                    _consume_optional_newline(fh)
                    c, o = _unpack_v42_cells(raw, n_cells)
                    conn_list.append(c)
                    off_list.extend((o[1:] + off_list[-1]).tolist())

                    if kw == "POLYGONS":
                        face_sizes = np.diff(o)
                        types = np.full(
                            n_cells, ELEMENT_TYPES["polygon"], dtype=np.uint8
                        )
                        types[face_sizes == 3] = ELEMENT_TYPES["triangle"]
                        types[face_sizes == 4] = ELEMENT_TYPES["quad"]
                        type_list.append(types)
                    elif kw == "LINES":
                        face_sizes = np.diff(o)
                        types = np.full(
                            n_cells, ELEMENT_TYPES["poly_line"], dtype=np.uint8
                        )
                        types[face_sizes == 2] = ELEMENT_TYPES["line"]
                        type_list.append(types)
                    elif kw == "VERTICES":
                        face_sizes = np.diff(o)
                        types = np.full(
                            n_cells, ELEMENT_TYPES["poly_vertex"], dtype=np.uint8
                        )
                        types[face_sizes == 1] = ELEMENT_TYPES["vertex"]
                        type_list.append(types)
                    elif kw == "TRIANGLE_STRIPS":
                        type_list.append(
                            np.full(
                                n_cells, ELEMENT_TYPES["triangle_strip"], dtype=np.uint8
                            )
                        )

            elif kw == "CELL_TYPES":
                n_ct = int(parts[1])
                n_bytes = n_ct * 4
                raw = np.frombuffer(fh.read(n_bytes), dtype=">i4").astype(np.int32)
                _consume_optional_newline(fh)
                mapped = np.array(
                    [
                        ELEMENT_TYPES.get(
                            VTK_TO_POLYXIOS.get(v, "polygon"), ELEMENT_TYPES["polygon"]
                        )
                        for v in raw
                    ],
                    dtype=np.uint8,
                )
                type_list.append(mapped)

            elif kw in ("POINT_DATA", "CELL_DATA"):
                n_items = int(parts[1])
                target_dict = v_attrs if kw == "POINT_DATA" else e_attrs

            elif kw == "SCALARS":
                name = parts[1]
                vtk_dt = parts[2].lower() if len(parts) > 2 else "double"
                n_comp = int(parts[3]) if len(parts) > 3 else 1
                np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f8")
                lt_pos = fh.tell()
                lt_line = (
                    fh.readline().decode("ascii", errors="replace").strip().upper()
                )
                if not lt_line.startswith("LOOKUP_TABLE"):
                    fh.seek(lt_pos)
                n_bytes = n_items * n_comp * np.dtype(np_dt).itemsize
                raw = np.frombuffer(fh.read(n_bytes), dtype=np_dt).astype(np.float64)
                _consume_optional_newline(fh)
                arr = raw.reshape(n_items, n_comp) if n_comp > 1 else raw

                # Strict routing fallback based on lengths to prevent validation panics
                if target_dict is v_attrs and n_items == n_verts:
                    v_attrs[name] = arr
                elif target_dict is e_attrs and n_items == n_elems:
                    e_attrs[name] = arr

            elif kw == "VECTORS":
                name = parts[1]
                vtk_dt = parts[2].lower() if len(parts) > 2 else "double"
                np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f8")
                n_bytes = n_items * 3 * np.dtype(np_dt).itemsize
                raw = np.frombuffer(fh.read(n_bytes), dtype=np_dt).astype(np.float64)
                _consume_optional_newline(fh)
                arr = raw.reshape(n_items, 3)
                if target_dict is v_attrs and n_items == n_verts:
                    v_attrs[name] = arr
                elif target_dict is e_attrs and n_items == n_elems:
                    e_attrs[name] = arr

            elif kw == "TENSORS":
                name = parts[1]
                vtk_dt = parts[2].lower() if len(parts) > 2 else "double"
                np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f8")
                n_bytes = n_items * 9 * np.dtype(np_dt).itemsize
                raw = np.frombuffer(fh.read(n_bytes), dtype=np_dt).astype(np.float64)
                _consume_optional_newline(fh)
                arr = raw.reshape(n_items, 3, 3)
                if target_dict is v_attrs and n_items == n_verts:
                    v_attrs[name] = arr
                elif target_dict is e_attrs and n_items == n_elems:
                    e_attrs[name] = arr

            elif kw == "FIELD":
                num_arrays = int(parts[2])
                for _ in range(num_arrays):
                    while True:
                        line = fh.readline().decode("ascii", errors="replace").strip()
                        if line:
                            break
                    fparts = line.split()
                    fname = fparts[0]
                    n_comp = int(fparts[1])
                    n_tuples = int(fparts[2])
                    vtk_dt = fparts[3].lower() if len(fparts) > 3 else "double"

                    if vtk_dt == "string":
                        # Skip string blocks in binary mode, which technically violate VTK spec anyway
                        continue

                    np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f8")
                    n_bytes = n_tuples * n_comp * np.dtype(np_dt).itemsize
                    raw = np.frombuffer(fh.read(n_bytes), dtype=np_dt).astype(
                        np.float64
                    )
                    _consume_optional_newline(fh)

                    arr = raw.reshape(n_tuples, n_comp) if n_comp > 1 else raw
                    # Intelligent length-based routing to satisfy rigid PolyData model
                    if n_tuples == n_verts and n_tuples != n_elems:
                        v_attrs[fname] = arr
                    elif n_tuples == n_elems and n_tuples != n_verts:
                        e_attrs[fname] = arr
                    elif n_tuples == n_verts and n_tuples == n_elems:
                        target_dict[fname] = arr

    vertices, final_conn, final_off, final_types = _finalize_geometry(
        dataset,
        dims,
        origin,
        spacing,
        x_coords,
        y_coords,
        z_coords,
        vertices,
        conn_list,
        off_list,
        type_list,
    )

    return PolyData(
        vertices=vertices,
        connectivity=final_conn,
        offsets=final_off,
        element_types=final_types,
        vertex_attrs=v_attrs,
        element_attrs=e_attrs,
    )


def _read_ascii_body(
    path: Path, offset: int, file_size: int, dataset: str, version: str
) -> PolyData:
    vertices = np.zeros((0, 3), dtype=np.float64)
    conn_list: list[np.ndarray] = []
    off_list: list[int] = [0]
    type_list: list[np.ndarray] = []
    v_attrs: dict[str, np.ndarray] = {}
    e_attrs: dict[str, np.ndarray] = {}

    dims = [0, 0, 0]
    origin = [0.0, 0.0, 0.0]
    spacing = [1.0, 1.0, 1.0]
    x_coords = y_coords = z_coords = None
    target_dict = v_attrs

    n_verts = 0
    n_elems = 0

    with open(path, "rb") as fh:
        fh.seek(offset)
        content = fh.read().decode("ascii", errors="replace")

    tokens = content.split()
    idx = 0
    n_tokens = len(tokens)

    while idx < n_tokens:
        kw = tokens[idx].upper()
        idx += 1

        if kw == "DIMENSIONS":
            dims = [int(tokens[idx]), int(tokens[idx + 1]), int(tokens[idx + 2])]
            idx += 3
            nx, ny, nz = dims
            n_verts = nx * ny * nz
            if nx > 1 and ny > 1 and nz > 1:
                n_elems = (nx - 1) * (ny - 1) * (nz - 1)
            elif nx > 1 and ny > 1:
                n_elems = (nx - 1) * (ny - 1)
        elif kw == "ORIGIN":
            origin = [
                float(tokens[idx]),
                float(tokens[idx + 1]),
                float(tokens[idx + 2]),
            ]
            idx += 3
        elif kw in ("SPACING", "ASPECT_RATIO"):
            spacing = [
                float(tokens[idx]),
                float(tokens[idx + 1]),
                float(tokens[idx + 2]),
            ]
            idx += 3
        elif kw == "POINTS":
            n_verts = int(tokens[idx])
            idx += 2
            count = n_verts * 3
            vertices = np.array(tokens[idx : idx + count], dtype=np.float64).reshape(
                n_verts, 3
            )
            idx += count
        elif kw in ("X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"):
            n_c = int(tokens[idx])
            idx += 2
            raw = np.array(tokens[idx : idx + n_c], dtype=np.float64)
            idx += n_c
            if kw == "X_COORDINATES":
                x_coords = raw
            elif kw == "Y_COORDINATES":
                y_coords = raw
            elif kw == "Z_COORDINATES":
                z_coords = raw
        elif kw in ("CELLS", "POLYGONS", "LINES", "VERTICES", "TRIANGLE_STRIPS"):
            n_cells = int(tokens[idx])
            n_elems += n_cells
            total_size = int(tokens[idx + 1])
            idx += 2

            if (
                kw == "CELLS"
                and version >= "5.1"
                and idx < n_tokens
                and tokens[idx].upper() == "OFFSETS"
            ):
                idx += 2
                off = np.array(tokens[idx : idx + n_cells + 1], dtype=np.int32)
                idx += n_cells + 1
                idx += 2
                conn_size = off[-1]
                conn = np.array(tokens[idx : idx + conn_size], dtype=np.int32)
                idx += conn_size
                conn_list.append(conn)
                off_list.extend((off[1:] + off_list[-1]).tolist())
            else:
                raw = np.array(tokens[idx : idx + total_size], dtype=np.int32)
                idx += total_size
                c, o = _unpack_v42_cells(raw, n_cells)
                conn_list.append(c)
                off_list.extend((o[1:] + off_list[-1]).tolist())

                if kw == "POLYGONS":
                    face_sizes = np.diff(o)
                    types = np.full(n_cells, ELEMENT_TYPES["polygon"], dtype=np.uint8)
                    types[face_sizes == 3] = ELEMENT_TYPES["triangle"]
                    types[face_sizes == 4] = ELEMENT_TYPES["quad"]
                    type_list.append(types)
                elif kw == "LINES":
                    face_sizes = np.diff(o)
                    types = np.full(n_cells, ELEMENT_TYPES["poly_line"], dtype=np.uint8)
                    types[face_sizes == 2] = ELEMENT_TYPES["line"]
                    type_list.append(types)
                elif kw == "VERTICES":
                    face_sizes = np.diff(o)
                    types = np.full(
                        n_cells, ELEMENT_TYPES["poly_vertex"], dtype=np.uint8
                    )
                    types[face_sizes == 1] = ELEMENT_TYPES["vertex"]
                    type_list.append(types)
                elif kw == "TRIANGLE_STRIPS":
                    type_list.append(
                        np.full(
                            n_cells, ELEMENT_TYPES["triangle_strip"], dtype=np.uint8
                        )
                    )

        elif kw == "CELL_TYPES":
            n_ct = int(tokens[idx])
            idx += 1
            raw = np.array(tokens[idx : idx + n_ct], dtype=np.int32)
            idx += n_ct
            mapped = np.array(
                [
                    ELEMENT_TYPES.get(
                        VTK_TO_POLYXIOS.get(v, "polygon"), ELEMENT_TYPES["polygon"]
                    )
                    for v in raw
                ],
                dtype=np.uint8,
            )
            type_list.append(mapped)

        elif kw in ("POINT_DATA", "CELL_DATA"):
            n_items = int(tokens[idx])
            idx += 1
            target_dict = v_attrs if kw == "POINT_DATA" else e_attrs

        elif kw == "SCALARS":
            name = tokens[idx]
            n_comp = 1
            if idx + 2 < n_tokens and tokens[idx + 2].isdigit():
                n_comp = int(tokens[idx + 2])
                idx += 3
            else:
                idx += 2
            if idx < n_tokens and tokens[idx].upper() == "LOOKUP_TABLE":
                idx += 2
            count = n_items * n_comp
            raw = np.array(tokens[idx : idx + count], dtype=np.float64)
            idx += count

            arr = raw.reshape(n_items, n_comp) if n_comp > 1 else raw
            if target_dict is v_attrs and n_items == n_verts:
                v_attrs[name] = arr
            elif target_dict is e_attrs and n_items == n_elems:
                e_attrs[name] = arr

        elif kw == "VECTORS":
            name = tokens[idx]
            idx += 2
            count = n_items * 3
            raw = np.array(tokens[idx : idx + count], dtype=np.float64)
            idx += count
            arr = raw.reshape(n_items, 3)
            if target_dict is v_attrs and n_items == n_verts:
                v_attrs[name] = arr
            elif target_dict is e_attrs and n_items == n_elems:
                e_attrs[name] = arr

        elif kw == "TENSORS":
            name = tokens[idx]
            idx += 2
            count = n_items * 9
            raw = np.array(tokens[idx : idx + count], dtype=np.float64)
            idx += count
            arr = raw.reshape(n_items, 3, 3)
            if target_dict is v_attrs and n_items == n_verts:
                v_attrs[name] = arr
            elif target_dict is e_attrs and n_items == n_elems:
                e_attrs[name] = arr

        elif kw == "FIELD":
            idx += 1
            num_arrays = int(tokens[idx])
            idx += 1
            for _ in range(num_arrays):
                fname = tokens[idx]
                n_comp = int(tokens[idx + 1])
                n_tuples = int(tokens[idx + 2])
                vtk_dt = tokens[idx + 3].lower() if idx + 3 < n_tokens else "double"
                idx += 4

                if vtk_dt == "string":
                    idx += n_tuples
                    continue

                count = n_tuples * n_comp
                raw = np.array(tokens[idx : idx + count], dtype=np.float64)
                idx += count

                arr = raw.reshape(n_tuples, n_comp) if n_comp > 1 else raw
                if n_tuples == n_verts and n_tuples != n_elems:
                    v_attrs[fname] = arr
                elif n_tuples == n_elems and n_tuples != n_verts:
                    e_attrs[fname] = arr
                elif n_tuples == n_verts and n_tuples == n_elems:
                    target_dict[fname] = arr

    vertices, final_conn, final_off, final_types = _finalize_geometry(
        dataset,
        dims,
        origin,
        spacing,
        x_coords,
        y_coords,
        z_coords,
        vertices,
        conn_list,
        off_list,
        type_list,
    )

    return PolyData(
        vertices=vertices,
        connectivity=final_conn,
        offsets=final_off,
        element_types=final_types,
        vertex_attrs=v_attrs,
        element_attrs=e_attrs,
    )


def _polyxios_to_vtk_code(type_code: int) -> int:
    name = ELEMENT_TYPES_INV.get(int(type_code))
    if name is None or name not in POLYXIOS_TO_VTK:
        return 7
    return POLYXIOS_TO_VTK[name]


def _write_cells_v42(poly: PolyData, fh: object, binary: bool) -> None:
    n_elems = len(poly.element_types)
    total_size = len(poly.connectivity) + n_elems
    fh.write(f"CELLS {n_elems} {total_size}\n".encode())

    if binary:
        parts: list[np.ndarray] = []
        for i in range(n_elems):
            s = int(poly.offsets[i])
            e = int(poly.offsets[i + 1])
            cnt = e - s
            parts.append(np.array([cnt], dtype=np.int32))
            parts.append(poly.connectivity[s:e].astype(np.int32))
        if parts:
            fh.write(np.concatenate(parts).astype(np.dtype(">i4")).tobytes())
    else:
        for i in range(n_elems):
            s = int(poly.offsets[i])
            e = int(poly.offsets[i + 1])
            face = poly.connectivity[s:e]
            fh.write(
                (str(e - s) + " " + " ".join(str(v) for v in face) + "\n").encode()
            )


def _write_cells_v51(poly: PolyData, fh: object, binary: bool) -> None:
    n_elems = len(poly.element_types)
    conn_size = len(poly.connectivity)

    fh.write(f"CELLS {n_elems} {conn_size}\n".encode())
    fh.write(b"OFFSETS vtktypeint64\n")
    offsets64 = poly.offsets.astype(np.int64)
    if binary:
        fh.write(offsets64.astype(np.dtype(">i8")).tobytes())
    else:
        fh.write((" ".join(str(x) for x in offsets64) + "\n").encode())

    fh.write(b"CONNECTIVITY vtktypeint64\n")
    conn64 = poly.connectivity.astype(np.int64)
    if binary:
        fh.write(conn64.astype(np.dtype(">i8")).tobytes())
    else:
        fh.write((" ".join(str(x) for x in conn64) + "\n").encode())


def _write_vtk_array(name: str, arr: np.ndarray, fh: object, binary: bool) -> None:
    if arr.ndim == 1:
        fh.write(f"SCALARS {name} double 1\n".encode())
        fh.write(b"LOOKUP_TABLE default\n")
        flat = arr.astype(np.float64)
        if binary:
            fh.write(flat.astype(np.dtype(">f8")).tobytes())
        else:
            fh.write((" ".join(f"{v:.10g}" for v in flat) + "\n").encode())
    elif arr.ndim == 2 and arr.shape[1] == 3:
        fh.write(f"VECTORS {name} double\n".encode())
        flat = arr.astype(np.float64).ravel()
        if binary:
            fh.write(flat.astype(np.dtype(">f8")).tobytes())
        else:
            for row in arr:
                fh.write(f"{row[0]:.10g} {row[1]:.10g} {row[2]:.10g}\n".encode())
    elif arr.ndim == 3 and arr.shape[1] == 3 and arr.shape[2] == 3:
        fh.write(f"TENSORS {name} double\n".encode())
        for mat in arr.astype(np.float64):
            for row in mat:
                fh.write(f"{row[0]:.10g} {row[1]:.10g} {row[2]:.10g}\n".encode())
    elif arr.ndim == 2 and arr.shape[1] == 6:
        fh.write(f"TENSORS {name} double\n".encode())
        for row in arr.astype(np.float64):
            mat = np.array(
                [
                    [row[0], row[3], row[4]],
                    [row[3], row[1], row[5]],
                    [row[4], row[5], row[2]],
                ]
            )
            for r in mat:
                fh.write(f"{r[0]:.10g} {r[1]:.10g} {r[2]:.10g}\n".encode())
    else:
        n_comp = arr.shape[1] if arr.ndim == 2 else 1
        fh.write(f"SCALARS {name} double {n_comp}\n".encode())
        fh.write(b"LOOKUP_TABLE default\n")
        flat = arr.astype(np.float64).ravel()
        if binary:
            fh.write(flat.astype(np.dtype(">f8")).tobytes())
        else:
            fh.write((" ".join(f"{v:.10g}" for v in flat) + "\n").encode())
